"""Extraction lifecycle, audit history, and current source checks."""
from copy import deepcopy

import pytest


from core.llm import InvalidModelResponse
from features.extraction import state
from features.extraction.schema import build_spec


SPEC = build_spec("Extract this study only", [{
    "id": "q1", "text": "Ecosystems?", "type": "multiple_choice", "options": ["Forest", "Wetland"],
}, {"id": "q2", "text": "Region?", "type": "open_text"}])
SOURCE = "pdf-sha256"


def answers():
    return {
        "q1": {"values": ["Forest", "Wetland"], "other_text": "", "page": 1, "quote": "Forests and wetlands were assessed.", "issue": ""},
        "q2": {"values": ["Not reported"], "other_text": "", "page": None, "quote": "", "issue": ""},
    }


def run(paper, result=None):
    state.set_ai_result(paper, SPEC, SOURCE, {"answers": result if result is not None else answers()}, "Provider", "Model", 5)


def test_form_revision_advances_once_per_result_error_and_archive_but_not_draft():
    paper = {}
    assert state.form_nonce(state.get(paper)) == 0
    run(paper)
    assert state.form_nonce(state.get(paper)) == 1
    state.save_review(paper, SPEC, SOURCE, answers(), page_count=5)
    assert state.form_nonce(state.get(paper)) == 1
    state.set_error(paper, SPEC, SOURCE, "Unavailable", "call_failed")
    assert state.form_nonce(state.get(paper)) == 2
    state.archive(paper)
    assert state.form_nonce(state.get(paper)) == 3


def test_ai_then_review_uses_actual_answers_and_multiselect_order():
    paper = {}
    assert state.pending(paper, SPEC, SOURCE)
    run(paper)
    assert state.status(paper, SPEC, SOURCE) == "Needs review"
    assert not state.pending(paper, SPEC, SOURCE)
    final = answers()
    final["q1"]["values"].reverse()
    final["q2"] = {"values": ["Brazil"], "other_text": "", "page": 2, "quote": "The site was in Brazil.", "issue": ""}
    state.save_review(paper, SPEC, SOURCE, final, page_count=5, confirm=True)
    assert state.status(paper, SPEC, SOURCE) == "Confirmed"
    assert state.get(paper)["decisions"] == {"q1": "accepted", "q2": "edited"}
    assert state.confirmed_answers(paper, SPEC, SOURCE) == final


def test_draft_can_be_incomplete_but_never_completes_or_auto_retries():
    paper = {}
    incomplete = {"q1": {"values": []}}
    state.save_review(paper, SPEC, SOURCE, incomplete)
    assert state.status(paper, SPEC, SOURCE) == "Draft"
    assert state.get(paper)["final_answers"] == incomplete
    assert not state.pending(paper, SPEC, SOURCE)
    assert not state.confirmed_answers(paper, SPEC, SOURCE)
    before = deepcopy(paper)
    with pytest.raises(ValueError):
        state.save_review(paper, SPEC, SOURCE, incomplete, confirm=True)
    assert paper == before


def test_large_draft_preserves_question_and_answer_objects():
    paper = {}
    draft = answers()
    draft["q1"]["quote"] = "Evidence " * 1300
    draft["q1"]["other_text"] = "Explanation " * 800
    draft["q2"]["quote"] = "More evidence " * 1500
    state.save_review(paper, SPEC, SOURCE, draft)
    saved = state.get(paper)["final_answers"]
    assert isinstance(saved, dict)
    assert isinstance(saved["q1"], dict)
    assert saved["q1"]["quote"] == draft["q1"]["quote"]
    assert saved["q1"]["values"] == draft["q1"]["values"]
    assert len(saved["q2"]["quote"]) <= 12000


def test_human_only_review_is_bound_to_spec_and_source():
    paper = {}
    state.save_review(paper, SPEC, SOURCE, answers(), page_count=5, confirm=True)
    assert set(state.get(paper)["decisions"].values()) == {"human"}
    changed = deepcopy(SPEC)
    changed["instructions"] = "Different instructions"
    for spec, source in ((changed, SOURCE), (SPEC, "new-pdf"), (SPEC, "")):
        assert state.is_stale(paper, spec, source)
        assert state.status(paper, spec, source) == "Outdated"
        assert not state.confirmed_answers(paper, spec, source)
        assert not state.pending(paper, spec, source)
    with pytest.raises(ValueError, match="Archive outdated"):
        state.save_review(paper, changed, SOURCE, answers(), confirm=True)


def test_partial_invalid_field_is_saved_for_audit_and_repaired_by_human():
    paper = {}
    result = answers()
    result["q1"]["values"] = ["Savanna"]
    run(paper, result)
    assert state.status(paper, SPEC, SOURCE) == "Invalid"
    assert not state.pending(paper, SPEC, SOURCE)
    record = state.get(paper)
    assert list(record["ai_answers"]) == ["q2"]
    assert record["invalid_answers"]["q1"]["values"] == ["Savanna"]
    state.save_review(paper, SPEC, SOURCE, answers(), page_count=5, confirm=True)
    assert state.get(paper)["decisions"] == {"q1": "human", "q2": "accepted"}


def test_errors_retry_only_transient_failures_and_allow_human_review():
    paper = {}
    state.set_error(paper, SPEC, SOURCE, "Timed out", "call_failed")
    assert state.status(paper, SPEC, SOURCE) == "Error"
    assert state.pending(paper, SPEC, SOURCE)
    state.set_error(paper, SPEC, SOURCE, "Wrong envelope", "invalid_response")
    assert state.status(paper, SPEC, SOURCE) == "Invalid"
    assert not state.pending(paper, SPEC, SOURCE)
    state.save_review(paper, SPEC, SOURCE, answers(), confirm=True)
    assert state.status(paper, SPEC, SOURCE) == "Confirmed"
    assert not state.pending(paper, SPEC, SOURCE)


def test_archive_releases_stale_state_and_keeps_every_earlier_version():
    paper = {}
    for _ in range(7):
        run(paper)
    state.save_review(paper, SPEC, SOURCE, answers(), confirm=True)
    state.archive(paper)
    record = state.get(paper)
    assert set(record) - {"form_nonce"} == {"history"}
    assert len(record["history"]) == 7
    assert record["history"][-1]["spec"] == SPEC
    assert record["history"][-1]["review_state"] == "confirmed"
    assert "history" not in record["history"][-1]
    assert state.status(paper, SPEC, "new-pdf") == "Pending"
    assert state.pending(paper, SPEC, "new-pdf")


def test_reopening_confirmed_review_archives_previous_final_and_retains_ai():
    paper = {}
    run(paper)
    state.save_review(paper, SPEC, SOURCE, answers(), confirm=True)
    changed = answers()
    changed["q1"]["values"] = ["Forest"]
    state.save_review(paper, SPEC, SOURCE, changed)
    record = state.get(paper)
    assert record["history"][-1]["final_answers"] == answers()
    assert record["ai_answers"] == answers()
    assert record["review_state"] == "draft"
    assert not state.confirmed_answers(paper, SPEC, SOURCE)


def test_replacing_results_archives_review_and_does_not_keep_confirmation():
    paper = {}
    run(paper)
    state.save_review(paper, SPEC, SOURCE, answers(), confirm=True)
    run(paper)
    record = state.get(paper)
    assert record["history"][-1]["review_state"] == "confirmed"
    assert "final_answers" not in record
    assert not state.confirmed_answers(paper, SPEC, SOURCE)


def test_invalid_mutations_leave_existing_state_intact():
    paper = {}
    run(paper)
    before = deepcopy(paper)
    with pytest.raises(InvalidModelResponse):
        state.set_ai_result(paper, SPEC, SOURCE, {"wrong": {}}, "P", "M")
    for source in (None, "", " "):
        with pytest.raises(ValueError):
            state.save_review(paper, SPEC, source, answers(), confirm=True)
    assert paper == before


def test_stored_spec_and_answers_cannot_be_changed_by_input_mutation():
    paper, spec, final = {}, deepcopy(SPEC), answers()
    state.save_review(paper, spec, SOURCE, final, confirm=True)
    spec["questions"][0]["text"] = "Changed"
    final["q1"]["values"].clear()
    result = state.confirmed_answers(paper, SPEC, SOURCE)
    assert result == answers()
    result["q1"]["values"].clear()
    assert state.confirmed_answers(paper, SPEC, SOURCE) == answers()


def test_out_of_bounds_stored_confirmation_is_not_current():
    paper = {}
    state.save_review(paper, SPEC, SOURCE, answers(), page_count=5, confirm=True)
    state.get(paper)["final_answers"]["q1"]["page"] = 9
    assert not state.confirmed_answers(paper, SPEC, SOURCE)
