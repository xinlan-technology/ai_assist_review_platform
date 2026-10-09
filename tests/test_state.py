"""Workflow logic regressions without a Streamlit runtime or database."""
from __future__ import annotations


from features.workflow import state

A = state.STAGE_ABSTRACT
F = state.STAGE_FULLTEXT
HASH_A = state.criteria_hash("criteria A")
HASH_B = state.criteria_hash("criteria B")


def _hashes(abstract=HASH_A, fulltext=HASH_A):
    return {A: abstract, F: fulltext}


def _paper(**stage_fields) -> dict:
    p = state.new_paper("10.1/x", "T", "A")
    if stage_fields:
        p["stages"][A] = stage_fields
    return p


def _ai(paper, stage=A, verdict=state.VERDICT_INCLUDE, chash=HASH_A):
    state.set_ai_result(paper, stage, verdict, "r", "OpenAI", "gpt-4o", chash, "abstract-v2")


def test_v1_migration():
    v1 = {
        "topic": "Flood fusion",
        "threshold": 80,
        "rubric": [
            {"score": 100, "definition": "Perfect fit"},
            {"score": 80, "definition": "Good fit"},
            {"score": 50, "definition": "Off topic"},
        ],
        "papers": [
            {"doi": "10.1/a", "title": "P1", "abstract": "A1", "ai_score": 90,
             "ai_reason": "fits", "decision": "agree", "human_score": None},
            {"doi": "", "title": "P2", "abstract": "A2", "ai_score": 40,
             "ai_reason": "weak", "decision": "disagree", "human_score": 90},
            {"doi": "", "title": "P3", "abstract": "A3", "ai_score": None,
             "ai_reason": "boom", "decision": None, "human_score": None},
            # Legacy disagree after an AI failure becomes a human-only decision.
            {"doi": "", "title": "P4", "abstract": "A4", "ai_score": None,
             "ai_reason": "boom", "decision": "disagree", "human_score": 100},
        ],
        "original_columns": ["Title"],
        "original_records": [{"Title": f"P{i}"} for i in range(1, 5)],
        "idx": 1,
    }
    m = state.migrate(v1)
    assert m["schema_version"] == state.SCHEMA_VERSION
    assert m["mode"] == state.MODE_PRISMA
    crit = m["config"]["abstract_criteria"]
    assert "Perfect fit" in crit and "Off topic" in crit
    chash = state.criteria_hash(crit)
    hashes = {A: chash, F: state.criteria_hash("")}

    p1, p2, p3, p4 = m["papers"]
    s1 = p1["stages"][A]
    assert s1["ai_verdict"] == state.VERDICT_INCLUDE
    assert s1["decision"] == state.DECISION_AGREE
    assert s1["criteria_hash"] == chash
    assert s1["prompt_version"] == "v1-rubric"
    assert state.current_final_verdict(p1, A, chash) == state.VERDICT_INCLUDE

    s2 = p2["stages"][A]
    assert s2["ai_verdict"] == state.VERDICT_EXCLUDE
    assert s2["human_verdict"] == state.VERDICT_INCLUDE
    assert state.current_final_verdict(p2, A, chash) == state.VERDICT_INCLUDE

    s3 = p3["stages"][A]
    assert s3["ai_verdict"] is None
    assert s3["ai_error_kind"] == state.ERROR_CALL_FAILED
    assert not state.is_reviewed(p3, A)
    assert state.display_label(p3, A, chash) == state.LABEL_ERROR

    s4 = p4["stages"][A]
    assert s4["decision"] == state.DECISION_HUMAN
    assert s4["review_criteria_hash"] == chash
    assert not state.is_stale(p4, A, chash)
    assert state.current_final_verdict(p4, A, chash) == state.VERDICT_INCLUDE

    ft = {i for i, _ in state.eligible_papers(m["papers"], state.MODE_PRISMA, F, hashes)}
    assert ft == {0, 1, 3}
    assert m["cursors"] == {A: 1}


def test_migrate_rejects_newer_schema():
    try:
        state.migrate({"schema_version": 99, "mode": "prisma", "papers": []})
    except state.UnsupportedSchemaError:
        pass
    else:
        raise AssertionError("newer schema must be refused, not mangled")


def test_migrate_normalizes_current_schema():
    m = state.migrate({"schema_version": state.SCHEMA_VERSION, "mode": "??",
                       "config": {"abstract_criteria": "x"}, "papers": []})
    assert m["mode"] == state.MODE_PRISMA
    assert m["config"]["abstract_criteria"] == "x"
    assert m["config"]["extraction_questions"] == []
    assert state.migrate({})["papers"] == []


def test_disagree_does_not_invent_a_verdict():
    p = _paper()
    _ai(p, verdict=state.VERDICT_EXCLUDE)
    state.record_disagree(p, A)
    assert state.final_verdict(p, A) is None
    assert not state.is_reviewed(p, A)
    assert state.display_label(p, A, HASH_A) == state.LABEL_PENDING
    state.set_human_verdict(p, A, state.VERDICT_INCLUDE, review_hash=HASH_A)
    assert p["stages"][A]["decision"] == state.DECISION_DISAGREE
    assert state.current_final_verdict(p, A, HASH_A) == state.VERDICT_INCLUDE


def test_reopening_disagree_clears_old_confirmation():
    p = _paper()
    _ai(p, verdict=state.VERDICT_INCLUDE)
    state.record_disagree(p, A)
    state.set_human_verdict(p, A, state.VERDICT_EXCLUDE, review_hash=HASH_A)
    assert state.is_reviewed(p, A, HASH_A)
    nonce_before = p["stages"][A].get("hv_nonce", 0)

    state.record_agree(p, A)
    state.record_disagree(p, A)
    s = p["stages"][A]
    assert s["human_verdict"] is None
    assert not state.is_reviewed(p, A, HASH_A)
    assert s["hv_nonce"] == nonce_before + 1


def test_human_verdict_without_ai_is_human_only():
    p = _paper(ai_verdict=None, ai_error="boom", ai_error_kind=state.ERROR_CALL_FAILED)
    state.set_human_verdict(p, A, state.VERDICT_INCLUDE, review_hash=HASH_A)
    s = p["stages"][A]
    assert s["decision"] == state.DECISION_HUMAN
    assert s["review_criteria_hash"] == HASH_A
    assert state.is_reviewed(p, A, HASH_A)


def test_human_only_verdict_requires_review_hash():
    # Missing hashes would make new human-only decisions immediately stale.
    for review_hash in (None, ""):
        p = _paper()
        try:
            state.set_human_verdict(p, A, state.VERDICT_INCLUDE, review_hash=review_hash)
        except ValueError:
            pass
        else:
            raise AssertionError("human-only verdict without review_hash must be rejected")
        assert p["stages"].get(A, {}).get("decision") is None
    # Overrides can inherit the existing AI criteria hash.
    p = _paper()
    _ai(p, verdict=state.VERDICT_EXCLUDE)
    state.set_human_verdict(p, A, state.VERDICT_INCLUDE)
    assert p["stages"][A]["decision"] == state.DECISION_DISAGREE


def test_verdict_validation_at_state_layer():
    p = _paper()
    for bad_call in (
        lambda: state.set_human_verdict(p, F, state.VERDICT_UNSURE),
        lambda: state.set_human_verdict(p, A, "garbage"),
        lambda: state.set_human_verdict(p, state.STAGE_EXTRACTION, state.VERDICT_INCLUDE),
        lambda: state.set_ai_result(p, F, state.VERDICT_UNSURE, "r", "x", "m", HASH_A, "v"),
        lambda: state.set_ai_result(p, A, "garbage", "r", "x", "m", HASH_A, "v"),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("state layer must reject invalid verdicts")
    assert p["stages"].get(A, {}).get("human_verdict") is None


def test_agreement_measured_on_answers_not_buttons():
    a = _paper()
    _ai(a)
    state.record_agree(a, A)
    b = _paper()  # Disagree can still produce the same answer as the AI.
    _ai(b, verdict=state.VERDICT_INCLUDE)
    state.record_disagree(b, A)
    state.set_human_verdict(b, A, state.VERDICT_INCLUDE, review_hash=HASH_A)
    c = _paper()
    _ai(c, verdict=state.VERDICT_EXCLUDE)
    state.record_disagree(c, A)
    state.set_human_verdict(c, A, state.VERDICT_INCLUDE, review_hash=HASH_A)
    d = _paper(ai_verdict=None, ai_error="x", ai_error_kind=state.ERROR_CALL_FAILED)
    state.set_human_verdict(d, A, state.VERDICT_INCLUDE, review_hash=HASH_A)
    e = _paper()
    _ai(e, verdict=state.VERDICT_EXCLUDE)
    state.record_disagree(e, A)

    s = state.summarize([a, b, c, d, e], state.MODE_PRISMA, A, _hashes())
    assert s["agreed"] == 2
    assert s["disagreed"] == 1
    assert s["human_only"] == 1
    assert s["agreement_rate"] == 2 / 3
    assert s["reviewed"] == 4


def test_stale_results_do_not_advance_or_count():
    p = _paper()
    _ai(p, verdict=state.VERDICT_INCLUDE, chash=HASH_A)
    state.record_agree(p, A)
    fresh = _hashes(abstract=HASH_A)
    edited = _hashes(abstract=HASH_B)

    assert state.current_final_verdict(p, A, HASH_A) == state.VERDICT_INCLUDE
    assert len(state.eligible_papers([p], state.MODE_PRISMA, F, fresh)) == 1

    assert state.is_stale(p, A, HASH_B)
    assert state.current_final_verdict(p, A, HASH_B) is None
    assert state.eligible_papers([p], state.MODE_PRISMA, F, edited) == []
    s = state.summarize([p], state.MODE_PRISMA, A, edited)
    assert s["reviewed"] == 0 and s["stale"] == 1
    assert s["agreed"] == 0 and s["agreement_rate"] is None
    assert s["counts"][state.LABEL_STALE] == 1
    assert state.display_label(p, A, HASH_B) == state.LABEL_STALE

    # Historical outcomes remain available for exports.
    assert state.final_verdict(p, A) == state.VERDICT_INCLUDE


def test_human_only_decisions_go_stale_too():
    p = _paper(ai_verdict=None, ai_error="x", ai_error_kind=state.ERROR_CALL_FAILED)
    state.set_human_verdict(p, A, state.VERDICT_INCLUDE, review_hash=HASH_A)
    assert not state.is_stale(p, A, HASH_A)
    assert state.is_stale(p, A, HASH_B)
    assert state.current_final_verdict(p, A, HASH_B) is None
    assert state.eligible_papers([p], state.MODE_PRISMA, F, _hashes(abstract=HASH_B)) == []


def test_stale_detection_and_archive():
    p = _paper()
    _ai(p, chash=HASH_A)
    state.record_agree(p, A)
    stale = state.stale_of([p], state.MODE_PRISMA, A, _hashes(abstract=HASH_B))
    assert len(stale) == 1

    state.archive_ai_result(p, A)
    s = p["stages"][A]
    assert s.get("ai_verdict") is None and s.get("decision") is None
    assert len(s["history"]) == 1
    assert s["history"][0]["ai_verdict"] == state.VERDICT_INCLUDE
    assert s["history"][0]["decision"] == state.DECISION_AGREE
    assert state.pending_of([p], state.MODE_PRISMA, A, _hashes())


def test_archive_covers_human_only_decisions():
    p = _paper(ai_verdict=None, ai_error="x", ai_error_kind=state.ERROR_CALL_FAILED)
    state.set_human_verdict(p, A, state.VERDICT_EXCLUDE, review_hash=HASH_A)
    state.archive_ai_result(p, A)
    s = p["stages"][A]
    assert s["decision"] is None and s["human_verdict"] is None
    assert s.get("review_criteria_hash") is None
    assert s["history"][0]["human_verdict"] == state.VERDICT_EXCLUDE
    assert s["history"][0]["review_criteria_hash"] == HASH_A


def test_replacing_pdf_archives_old_fulltext_outcome():
    p = _paper()
    state.set_ai_result(
        p, F, state.VERDICT_INCLUDE, "r", "OpenAI", "gpt-4o", HASH_A, "fulltext-v1"
    )
    state.record_agree(p, F)
    assert state.archive_if_source_changed(p, F, "old-pdf", "new-pdf")
    s = p["stages"][F]
    assert state.final_verdict(p, F) is None
    assert s.get("ai_verdict") is None
    assert s["history"][-1]["ai_verdict"] == state.VERDICT_INCLUDE
    assert not state.archive_if_source_changed(p, F, "new-pdf", "new-pdf")


def test_first_pdf_archives_a_verdict_recorded_without_source():
    p = _paper()
    state.set_human_verdict(
        p, F, state.VERDICT_EXCLUDE, review_hash=HASH_A
    )
    assert state.archive_if_source_changed(p, F, None, "first-pdf")
    assert state.final_verdict(p, F) is None
    assert p["stages"][F]["history"][-1]["human_verdict"] == state.VERDICT_EXCLUDE

    untouched = _paper()
    assert not state.archive_if_source_changed(untouched, F, None, "first-pdf")
    assert untouched["stages"][F] == {}


def test_pending_excludes_invalid_and_decided():
    fresh = _paper()
    failed = _paper(ai_verdict=None, ai_error="net", ai_error_kind=state.ERROR_CALL_FAILED)
    invalid = _paper(ai_verdict=None, ai_error="bad json",
                     ai_error_kind=state.ERROR_INVALID_RESPONSE)
    decided = _paper(ai_verdict=None, ai_error="net", ai_error_kind=state.ERROR_CALL_FAILED)
    state.set_human_verdict(decided, A, state.VERDICT_EXCLUDE, review_hash=HASH_A)
    papers = [fresh, failed, invalid, decided]

    pending = {i for i, _ in state.pending_of(papers, state.MODE_PRISMA, A, _hashes())}
    assert pending == {0, 1}
    inv = {i for i, _ in state.invalid_of(papers, state.MODE_PRISMA, A, _hashes())}
    assert inv == {2}

    s = state.summarize(papers, state.MODE_PRISMA, A, _hashes())
    assert s["ai_failed"] == 2 and s["ai_invalid"] == 1


def test_stage_eligibility_flow():
    def decided(verdict):
        p = _paper()
        _ai(p, verdict=verdict)
        state.record_agree(p, A)
        return p

    inc, uns, exc = decided(state.VERDICT_INCLUDE), decided(state.VERDICT_UNSURE), decided(state.VERDICT_EXCLUDE)
    pen = _paper()
    _ai(pen)  # screened but not reviewed
    papers = [inc, uns, exc, pen]

    ft = {i for i, _ in state.eligible_papers(papers, state.MODE_PRISMA, F, _hashes())}
    assert ft == {0, 1}

    ft_direct = state.eligible_papers(papers, state.MODE_DIRECT, F, _hashes())
    assert len(ft_direct) == 4

    state.set_ai_result(inc, F, state.VERDICT_INCLUDE, "r", "x", "m", HASH_A, "v")
    state.record_agree(inc, F)
    state.set_ai_result(uns, F, state.VERDICT_EXCLUDE, "r", "x", "m", HASH_A, "v")
    state.record_agree(uns, F)
    ex = {i for i, _ in state.eligible_papers(papers, state.MODE_PRISMA,
                                              state.STAGE_EXTRACTION, _hashes())}
    assert ex == {0}
    ex2 = state.eligible_papers(papers, state.MODE_PRISMA, state.STAGE_EXTRACTION,
                                _hashes(fulltext=HASH_B))
    assert ex2 == []


def test_fulltext_has_no_unsure():
    assert state.VERDICT_UNSURE in state.STAGE_VERDICTS[A]
    assert state.VERDICT_UNSURE not in state.STAGE_VERDICTS[F]


def test_export_shows_pending_fulltext_before_screening():
    # Eligibility must not depend on stage_state() initializing the stage dict.
    p = _paper()
    _ai(p, verdict=state.VERDICT_INCLUDE)
    state.record_agree(p, A)
    base = state.pd.DataFrame({"Title": ["T"]})
    state._stage_columns(base, [p], state.MODE_PRISMA, F, "Full-text", _hashes())
    assert base.loc[0, "Full-text: Final"] == state.LABEL_PENDING
