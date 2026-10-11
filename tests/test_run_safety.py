"""No model call is repeated unless the reviewer asks for that exact kind of repeat."""
from copy import deepcopy
import time

import pytest

from core import db
from core.llm import PROVIDERS
from features.screening import prompts
from features.workflow import runs, state
from run_fixtures import install_run_store
from test_extraction_ui import ANSWER, element as extraction_element, record as extraction_record, ui  # noqa: F401
from test_screening_ui import PROJECT_ID, STAGES, element, record, run_button, screening_ui, store  # noqa: F401

USER = "screening@example.invalid"
NEWER = "A different, newer criterion."
SELECTED = PROVIDERS["OpenAI"][0]  # the sidebar's default model


def _ready(app):
    element(app, "text_input", "API key").input("offline-key").run()
    return app


def _labels(app, prefix):
    return [button.label for button in app.button if button.label.startswith(prefix)]


def _spec(stage, criteria):
    version = prompts.ABSTRACT_PROMPT_VERSION if stage == state.STAGE_ABSTRACT else prompts.FULLTEXT_PROMPT_VERSION
    return {"criteria": criteria, "prompt_version": version}


def _attempt(app, stage, spec, *, model=SELECTED, finish="succeeded", metadata=None):
    """Record an attempt for the first paper as an earlier run would have."""
    paper = store(app)["papers"][0]
    run = db.start_ai_run(
        USER, PROJECT_ID, paper_uid=paper["uid"], stage=stage, batch_id="earlier",
        provider="OpenAI", model=model, config_hash=runs.config_hash(stage, spec),
        source_hash=runs.source_hash(stage, paper, metadata), prompt_version=spec["prompt_version"],
        prompt_snapshot=spec,
    )
    if finish == "succeeded":
        result = {"ai_verdict": "exclude", "ai_reason": "Saved earlier.", "provider": "OpenAI",
                  "model": model, "criteria_hash": runs.config_hash(stage, spec),
                  "prompt_version": spec["prompt_version"]}
        run = db.finish_ai_run(USER, PROJECT_ID, run["id"], status="succeeded", result=result,
                               duration_seconds=1.0)
    elif finish:
        run = db.finish_ai_run(USER, PROJECT_ID, run["id"], status=finish,
                               result={"ai_error": "Earlier failure.", "ai_error_kind": finish},
                               duration_seconds=1.0)
    return run


def _metadata(app, stage):
    return None if stage == state.STAGE_ABSTRACT else db.load_fulltexts(USER, PROJECT_ID)


@pytest.mark.parametrize("stage", STAGES)
def test_failed_call_gets_its_own_retry_and_is_never_reported_as_new_work(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    rows = install_run_store(monkeypatch).rows
    app = create(stage)
    store(app)["papers"] = store(app)["papers"][:1]
    _ready(app.run())
    model.side_effect = RuntimeError("429 rate limit")
    run_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 1 and rows[0]["status"] == "call_failed"
    assert run_button(app, stage).disabled and "0 calls" in run_button(app, stage).label
    retry = "↻ Retry failed calls (1 paper · 1 call, may be billed again)"
    model.side_effect = None
    element(app, "button", retry).click().run()

    assert not app.exception
    assert model.call_count == 2
    assert record(app, stage)["ai_verdict"] == "include"
    assert not _labels(app, "↻ Retry failed")
    assert run_button(app, stage).disabled


@pytest.mark.parametrize("stage", STAGES)
def test_outdated_rerun_bills_once_archives_the_old_decision_and_awaits_review(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    install_run_store(monkeypatch)
    app = create(stage, "confirmed_override")
    store(app)["papers"] = store(app)["papers"][:1]
    old = deepcopy(record(app, stage))
    store(app)["config"][f"{stage}_criteria"] = NEWER
    _ready(app.run())
    rerun = _labels(app, "↻ Re-run outdated")
    assert len(rerun) == 1 and "(1 paper · 1 call)" in rerun[0]
    element(app, "button", rerun[0]).click().run()

    assert not app.exception
    assert model.call_count == 1
    assert not _labels(app, "↻ Re-run outdated")
    current = record(app, stage)
    paper = store(app)["papers"][0]
    chash = state.criteria_hash(NEWER)
    assert current["criteria_hash"] == chash and current["source_run_id"]
    # The fresh answer is a proposal: nothing is decided until the reviewer confirms it.
    assert current.get("decision") is None and current.get("human_verdict") is None
    assert not state.is_reviewed(paper, stage, chash)
    assert not state.is_stale(paper, stage, chash)
    archived = current["history"][-1]
    assert (archived["decision"], archived["human_verdict"], archived["ai_verdict"]) == (
        old["decision"], old["human_verdict"], old["ai_verdict"])


@pytest.mark.parametrize("stage", STAGES)
def test_refresh_adopts_an_answer_already_paid_for_without_another_call(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    install_run_store(monkeypatch)
    app = create(stage, "confirmed_override")
    store(app)["papers"] = store(app)["papers"][:1]
    store(app)["config"][f"{stage}_criteria"] = NEWER
    saved = _attempt(app, stage, _spec(stage, NEWER), metadata=_metadata(app, stage))
    _ready(app.run())
    rerun = _labels(app, "↻ Re-run outdated")
    assert len(rerun) == 1 and "(1 paper · 0 calls)" in rerun[0]
    element(app, "button", rerun[0]).click().run()

    assert not app.exception
    model.assert_not_called()
    current = record(app, stage)
    assert current["source_run_id"] == saved["id"] and current["ai_verdict"] == "exclude"
    assert len(current["history"]) == 1
    assert not _labels(app, "↻ Re-run outdated")


@pytest.mark.parametrize("stage", STAGES)
def test_pending_paper_shows_its_saved_answer_instead_of_calling_again(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    install_run_store(monkeypatch)
    app = create(stage)
    store(app)["papers"] = store(app)["papers"][:1]
    criteria = store(app)["config"][f"{stage}_criteria"]
    saved = _attempt(app, stage, _spec(stage, criteria), metadata=_metadata(app, stage))
    _ready(app.run())
    assert "0 calls" in run_button(app, stage).label and not run_button(app, stage).disabled
    run_button(app, stage).click().run()

    assert not app.exception
    model.assert_not_called()
    assert record(app, stage)["source_run_id"] == saved["id"]
    assert run_button(app, stage).disabled


@pytest.mark.parametrize("stage", STAGES)
def test_unknown_outcome_is_repeated_only_after_explicit_acceptance(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    install_run_store(monkeypatch)
    app = create(stage)
    store(app)["papers"] = store(app)["papers"][:1]
    criteria = store(app)["config"][f"{stage}_criteria"]
    _attempt(app, stage, _spec(stage, criteria), finish=None, metadata=_metadata(app, stage))
    _ready(app.run())
    assert run_button(app, stage).disabled
    assert not _labels(app, "↻ Retry failed")
    repeat = "↻ Repeat attempts with unknown outcome (1 paper · 1 call)"
    assert element(app, "button", repeat).disabled
    element(app, "checkbox", "Repeat them anyway; I accept a possible second charge.").check().run()
    element(app, "button", repeat).click().run()

    assert not app.exception
    assert model.call_count == 1
    assert record(app, stage)["ai_verdict"] == "include"
    assert not _labels(app, "↻ Repeat attempts")


def test_a_pair_that_succeeded_is_not_retried_because_a_later_attempt_failed():
    paper = {"uid": "p1", "title": "T", "abstract": "A", "stages": {}}
    spec = {"criteria": "Include.", "prompt_version": "v1"}
    base = {"paper_uid": "p1", "stage": "abstract", "provider": "OpenAI", "model": "gpt-4.1",
            "config_hash": runs.config_hash("abstract", spec),
            "source_hash": runs.source_hash("abstract", paper), "prompt_version": "v1"}
    records = [base | {"id": "1", "status": "succeeded"}, base | {"id": "2", "status": "call_failed"}]
    model = [{"provider": "OpenAI", "model": "gpt-4.1", "api_key": "k"}]
    pairs, succeeded = runs.outcomes([paper], "abstract", spec, None, records)
    assert pairs == {("p1", "OpenAI", "gpt-4.1"): runs.REPEAT} and succeeded == {"p1"}
    for category in (runs.NEW, runs.FAILED, runs.INVALID, runs.UNKNOWN):
        assert runs.plan([paper], "abstract", spec, None, model, records, include=(category,)) == []
    assert len(runs.plan([paper], "abstract", spec, None, model, records, repeat=True)) == 1
    later = [base | {"id": "1", "status": "call_failed"}, base | {"id": "2", "status": "running"}]
    assert runs.outcomes([paper], "abstract", spec, None, later)[0] == {
        ("p1", "OpenAI", "gpt-4.1"): runs.UNKNOWN}


def test_extraction_invalid_retry_cannot_be_billed_twice(ui, monkeypatch):
    create, _, model = ui
    install_run_store(monkeypatch)
    model.return_value = {"answers": {}}
    app = create().run()
    extraction_element(app, "text_input", "API key").input("mock-key").run()
    extraction_element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert extraction_record(app)["field_errors"]
    retry = "↻ Retry invalid responses (1 paper · 1 call, may be billed again)"
    model.return_value = {"answers": {"q1": ANSWER}}
    extraction_element(app, "button", retry).click().run()

    assert not app.exception
    assert model.call_count == 2
    assert not [button.label for button in app.button if button.label.startswith("↻ Retry invalid")]
    # Nobody had reviewed the partly invalid proposal, so the complete answer replaces it.
    current = extraction_record(app)
    assert current["ai_answers"]["q1"]["values"] == ["Forest"] and not current.get("field_errors")
    assert "review_state" not in current
    assert extraction_element(app, "button", "▶ Run AI extraction (0 papers · 0 calls)").disabled


def test_extraction_retry_never_replaces_reviewed_answers(ui, monkeypatch):
    create, _, model = ui
    install_run_store(monkeypatch)
    model.return_value = {"answers": {}}
    app = create().run()
    extraction_element(app, "text_input", "API key").input("mock-key").run()
    extraction_element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    extraction_element(app, "selectbox", "Your answer").select("Forest").run()
    extraction_element(app, "number_input", "Evidence PDF page").set_value(1).run()
    extraction_element(app, "text_area", "Supporting quote").input("A forest site.").run()
    extraction_element(app, "button", "Save draft").click().run()
    reviewed = deepcopy(extraction_record(app))
    assert reviewed["review_state"] == "draft" and reviewed["final_answers"]["q1"]["values"] == ["Forest"]
    model.return_value = {"answers": {"q1": ANSWER | {"values": ["Wetland"]}}}
    extraction_element(
        app, "button", "↻ Retry invalid responses (1 paper · 1 call, may be billed again)").click().run()

    assert not app.exception
    assert model.call_count == 2
    assert extraction_record(app) == reviewed


def test_planning_a_large_stage_stays_linear():
    spec = {"criteria": "Include empirical studies.", "prompt_version": "v1"}
    papers = [{"uid": f"p{i}", "title": f"Title {i}", "abstract": "Text " * 200, "stages": {}}
              for i in range(1000)]
    names = ["gpt-a", "gpt-b", "gpt-c"]
    chash = runs.config_hash("abstract", spec)
    records = [{"id": f"{p['uid']}-{name}", "paper_uid": p["uid"], "stage": "abstract",
                "provider": "OpenAI", "model": name, "config_hash": chash,
                "source_hash": runs.source_hash("abstract", p), "prompt_version": "v1",
                "status": "succeeded"} for p in papers for name in names]
    selected = [{"provider": "OpenAI", "model": name, "api_key": "k"} for name in names]
    monkeypatch_models = {"OpenAI": names}
    original = runs.PROVIDERS
    runs.PROVIDERS = monkeypatch_models
    try:
        started = time.perf_counter()
        assert runs.plan(papers, "abstract", spec, None, selected, records) == []
        assert len(runs.plan(papers, "abstract", spec, None, selected, records, repeat=True)) == 3000
        elapsed = time.perf_counter() - started
    finally:
        runs.PROVIDERS = original
    assert elapsed < 1.0, f"planning took {elapsed:.2f}s"


@pytest.mark.parametrize("stage", STAGES)
def test_closed_comparison_panel_loads_no_answers_for_the_whole_stage(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    ledger = install_run_store(monkeypatch)
    app = create(stage)
    _ready(app.run())
    run_button(app, stage).click().run()
    ledger.load.reset_mock()
    ledger.index.reset_mock()
    app.run()

    assert not app.exception
    # Only the open paper's own attempts are read; counters reuse the cached identities.
    assert ledger.load.call_args_list
    assert all(call.args[3] for call in ledger.load.call_args_list)
    ledger.index.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_main_run_completes_a_batch_that_stopped_between_two_models(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    rows = install_run_store(monkeypatch).rows
    app = create(stage)
    store(app)["papers"] = store(app)["papers"][:1]
    _ready(app.run())
    run_button(app, stage).click().run()
    assert model.call_count == 1 and record(app, stage)["source_run_id"] == rows[0]["id"]
    assert run_button(app, stage).disabled

    # The second model of the batch never ran; the paper is still undecided.
    element(app, "checkbox", "Compare multiple models").check().run()
    element(app, "selectbox", "Provider 2").select("OpenAI").run()
    element(app, "selectbox", "Model 2").select(PROVIDERS["OpenAI"][1]).run()
    assert "(1 paper · 1 call)" in run_button(app, stage).label
    run_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 2
    assert [(row["model"], row["status"]) for row in rows] == [
        (PROVIDERS["OpenAI"][0], "succeeded"), (PROVIDERS["OpenAI"][1], "succeeded")]
    assert record(app, stage)["source_run_id"] == rows[0]["id"]
    assert run_button(app, stage).disabled


@pytest.mark.parametrize("stage", STAGES)
def test_main_run_leaves_decided_and_unlinked_answers_to_the_comparison_panel(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    install_run_store(monkeypatch)
    # The first paper shows an answer recorded before attempts were kept.
    app = create(stage, "ai")
    store(app)["papers"] = store(app)["papers"][:1]
    _ready(app.run())
    assert run_button(app, stage).disabled and "0 calls" in run_button(app, stage).label
    model.assert_not_called()

    paper = {"uid": "p", "stages": {stage: {"ai_verdict": "include", "source_run_id": "r"}}}
    assert runs.awaiting_models([paper], stage) == [paper]
    decided = {"uid": "d", "stages": {stage: {"ai_verdict": "include", "source_run_id": "r",
                                              "decision": state.DECISION_AGREE}}}
    assert runs.awaiting_models([decided], stage) == []


def test_extraction_main_run_serves_unreviewed_papers_with_linked_answers_only():
    stage = state.STAGE_EXTRACTION
    empty = {"uid": "a", "stages": {}}
    linked = {"uid": "b", "stages": {stage: {"ai_answers": {"q": {}}, "source_run_id": "r"}}}
    unlinked = {"uid": "c", "stages": {stage: {"ai_answers": {"q": {}}}}}
    drafted = {"uid": "d", "stages": {stage: {"ai_answers": {"q": {}}, "source_run_id": "r",
                                              "review_state": "draft"}}}
    assert runs.awaiting_models([empty, linked, unlinked, drafted], stage) == [empty, linked]


@pytest.mark.parametrize("stage", STAGES)
def test_page_says_which_calls_are_left_to_the_comparison_panel(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    install_run_store(monkeypatch)
    app = create(stage, "ai")
    store(app)["papers"] = store(app)["papers"][:1]
    _ready(app.run())

    assert not app.exception
    assert any("1 paper · 1 call are not started by this button" in caption.value
               for caption in app.caption)
    model.assert_not_called()
