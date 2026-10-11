"""Offline screening UI regressions for durable decisions and paid-result recovery."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData

from core import auth, db, fulltext_storage
from features.screening import judge, prompts
from features.workflow import state
from run_fixtures import install_run_store


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "offline-screening-project"
PDF = b"%PDF-1.7\noffline screening fixture"
DIGEST = fulltext_storage.sha256(PDF)
STAGES = [state.STAGE_ABSTRACT, state.STAGE_FULLTEXT]


def element(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def store(app):
    return app.session_state["project_store"]


def record(app, stage, index=0):
    return state.stage_state(store(app)["papers"][index], stage)


def advancing_uids(app, stage):
    data = store(app)
    next_stage = state.STAGE_FULLTEXT if stage == state.STAGE_ABSTRACT else state.STAGE_EXTRACTION
    hashes = {key: state.criteria_hash(data["config"][f"{key}_criteria"]) for key in STAGES}
    return [paper["uid"] for _, paper in state.eligible_papers(
        data["papers"], data["mode"], next_stage, hashes,
    )]


def run_button(app, stage):
    prefix = "▶ Run AI screening" if stage == state.STAGE_ABSTRACT else "▶ Run AI on ready PDFs"
    return next(button for button in app.button if button.label.startswith(prefix))


@pytest.fixture
def screening_ui(monkeypatch):
    st.cache_data.clear()
    install_run_store(monkeypatch)
    monkeypatch.setattr(auth, "sidebar_user", lambda: None)
    monkeypatch.setattr(auth, "current_user", lambda: "screening@example.invalid")
    monkeypatch.setattr(st, "pdf", lambda *args, **kwargs: st.caption("Offline PDF viewer"))
    monkeypatch.setattr(st, "page_link", lambda *args, **kwargs: None)
    monkeypatch.setattr(fulltext_storage, "ensure_configured", lambda: None)
    monkeypatch.setattr(fulltext_storage, "backend_label", lambda: "offline fixture")
    load_pdf = Mock(return_value=PDF)
    monkeypatch.setattr(fulltext_storage, "load_pdf", load_pdf)
    save = Mock()
    monkeypatch.setattr(db, "save_project", save)
    metadata = {f"paper{i}": {
        "filename": f"study{i}.pdf", "storage_key": f"offline/paper{i}.pdf", "sha256": DIGEST,
        "file_size": len(PDF), "page_count": 2, "status": "ok",
    } for i in range(2)}
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: deepcopy(metadata))
    model = Mock(return_value={"verdict": "include", "reason": "The study meets the criteria."})
    monkeypatch.setattr(judge, "judge_abstract", model)
    monkeypatch.setattr(judge, "judge_fulltext", model)

    def create(stage, scenario="pending"):
        mode = state.MODE_PRISMA if stage == state.STAGE_ABSTRACT else state.MODE_DIRECT
        project = state.new_project_data(mode)
        project["config"].update(abstract_criteria="Include relevant abstracts.",
                                 fulltext_criteria="Include relevant studies.")
        papers = [state.new_paper("", f"Study {i}", "An offline abstract.") for i in range(2)]
        for index, paper in enumerate(papers):
            paper["uid"] = f"paper{index}"
        if scenario != "pending":
            version = (prompts.ABSTRACT_PROMPT_VERSION if stage == state.STAGE_ABSTRACT
                       else prompts.FULLTEXT_PROMPT_VERSION)
            state.set_ai_result(papers[0], stage, "include", "Eligible study.", "OpenAI", "test-model",
                                state.criteria_hash(project["config"][f"{stage}_criteria"]), version)
        if scenario == "confirmed_override":
            state.set_human_verdict(papers[0], stage, "include")
        elif scenario == "unconfirmed_override":
            state.record_disagree(papers[0], stage)
        app = AppTest.from_file(str(ROOT / "views" / f"{stage}_screening.py"), default_timeout=10)
        app.session_state["active_project_id"] = PROJECT_ID
        app.session_state["active_project_name"] = "Offline screening"
        app.session_state["project_store"] = {
            "mode": mode, "config": project["config"], "papers": papers,
            "original_df": pd.DataFrame(), "cursors": {},
            state._VERSION_KEY: 1,
        }
        return app

    return create, save, model, load_pdf


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("action", ["agree", "disagree", "human", "override"])
def test_failed_decision_save_preserves_state_eligibility_and_cursor(screening_ui, stage, action):
    create, save, model, _ = screening_ui
    scenario = {"agree": "ai", "disagree": "confirmed_override",
                "human": "pending", "override": "unconfirmed_override"}[action]
    app = create(stage, scenario).run()
    assert not app.exception
    if action in {"human", "override"}:
        app.radio[0].set_value("Include").run()
    before = deepcopy(store(app)["papers"])
    eligible_before = advancing_uids(app, stage)
    cursor_before = deepcopy(store(app)["cursors"])
    save.side_effect = db.DatabaseError("Offline save failure")
    label = {"agree": "✓ Agree", "disagree": "✗ Disagree",
             "human": "Confirm verdict", "override": "Confirm your verdict"}[action]
    element(app, "button", label).click().run()
    assert not app.exception
    assert store(app)["papers"] == before
    assert advancing_uids(app, stage) == eligible_before
    assert store(app)["cursors"] == cursor_before
    assert any("Changes were not saved" in error.value for error in app.error)
    model.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_failed_prompt_autosave_restores_criteria_and_current_results(screening_ui, stage):
    create, save, model, _ = screening_ui
    app = create(stage, "confirmed_override").run()
    before = deepcopy(store(app)["config"])
    eligible_before = advancing_uids(app, stage)
    save.side_effect = db.DatabaseError("Offline save failure")
    label = "Abstract prompt" if stage == state.STAGE_ABSTRACT else "Full-text prompt"
    element(app, "text_area", label).input("Changed criteria that must not persist.").run()
    assert not app.exception
    assert store(app)["config"] == before
    assert advancing_uids(app, stage) == eligible_before
    assert any("Changes were not saved" in error.value for error in app.error)
    model.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_failed_initial_checkpoint_prevents_all_model_calls(screening_ui, stage):
    create, save, model, _ = screening_ui
    app = create(stage).run()
    element(app, "text_input", "API key").input("offline-openai-key").run()
    before = deepcopy(store(app)["papers"])
    save.side_effect = db.DatabaseError("Offline initial save failure")
    run_button(app, stage).click().run()
    assert not app.exception
    assert store(app)["papers"] == before
    assert not store(app).get("unsaved_results")
    assert any("Stopping before the run" in error.value for error in app.error)
    model.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_failed_paid_checkpoint_blocks_work_and_retry_only_saves(screening_ui, stage):
    create, save, model, _ = screening_ui
    app = create(stage).run()
    element(app, "text_input", "API key").input("offline-openai-key").run()
    save.side_effect = [None, db.DatabaseError("Offline checkpoint failure")]
    run_button(app, stage).click().run()
    assert not app.exception
    assert model.call_count == 1
    assert record(app, stage)["ai_verdict"] == "include"
    assert not record(app, stage, 1).get("ai_verdict")
    assert store(app).get("unsaved_results") is True
    assert [button.label for button in app.button if not button.disabled] == ["Retry saving results"]
    assert app.get("download_button")
    assert not app.radio
    assert not app.text_input

    save.side_effect = db.DatabaseError("Offline retry failure")
    element(app, "button", "Retry saving results").click().run()
    assert not app.exception
    assert store(app).get("unsaved_results") is True
    assert model.call_count == 1
    assert [button.label for button in app.button if not button.disabled] == ["Retry saving results"]

    save.side_effect = None
    element(app, "button", "Retry saving results").click().run()
    assert not app.exception
    assert not store(app).get("unsaved_results")
    assert model.call_count == 1
    assert record(app, stage)["ai_verdict"] == "include"
    assert not record(app, stage, 1).get("ai_verdict")
    assert element(app, "button", "✓ Agree")


@pytest.mark.parametrize("conflict", [False, True])
def test_unsaved_work_can_be_backed_up_and_explicitly_discarded(screening_ui, monkeypatch,
                                                              conflict):
    create, save, model, _ = screening_ui
    app = create(state.STAGE_ABSTRACT)
    store(app)[state._CONFLICT_KEY] = "Another window saved." if conflict else None
    store(app)["unsaved_results"] = True
    server_data = state.new_project_data(state.MODE_PRISMA)
    server_data["config"]["abstract_criteria"] = "Current saved criteria"
    load = Mock(return_value=(server_data, 7, (["Title"], [{"Title": "Saved study"}])))
    monkeypatch.setattr(db, "load_project_bundle", load)
    app.run()
    assert not app.exception
    assert app.get("download_button")
    assert element(app, "button", "Retry saving results").disabled is conflict
    assert element(app, "button", "Discard local changes and reload").disabled
    assert element(app, "button", "Close project without saving").disabled
    save.assert_not_called()
    element(app, "checkbox", "I have backed up my work and want to discard this session's changes.").check().run()
    element(app, "button", "Discard local changes and reload").click().run()
    assert not app.exception
    assert not store(app).get("unsaved_results")
    assert store(app)["config"]["abstract_criteria"] == "Current saved criteria"
    assert store(app)[state._VERSION_KEY] == 7
    assert store(app)["original_df"].to_dict("records") == [{"Title": "Saved study"}]
    load.assert_called_once()
    save.assert_not_called()
    model.assert_not_called()


def test_deleted_project_recovery_can_close_without_saving(screening_ui, monkeypatch):
    create, save, model, _ = screening_ui
    app = create(state.STAGE_ABSTRACT)
    store(app)["unsaved_results"] = True
    monkeypatch.setattr(db, "load_project_bundle", lambda *args: ({}, 0, ([], [])))
    app.run()
    element(app, "checkbox", "I have backed up my work and want to discard this session's changes.").check().run()
    element(app, "button", "Discard local changes and reload").click().run()
    assert not app.exception
    assert any("no longer exists" in error.value for error in app.error)
    assert store(app).get("unsaved_results")
    element(app, "button", "Close project without saving").click().run()
    assert not app.exception
    assert app.session_state["active_project_id"] is None
    assert "project_store" not in app.session_state
    save.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_rerun_after_model_response_recovers_without_another_model_call(screening_ui, stage):
    create, save, model, _ = screening_ui
    app = create(stage).run()
    element(app, "text_input", "API key").input("offline-openai-key").run()

    def respond_then_rerun(*args):
        context = get_script_run_ctx()
        assert context is not None and context.script_requests is not None
        context.script_requests.request_rerun(RerunData())
        return {"verdict": "include", "reason": "Relevant study."}

    model.side_effect = respond_then_rerun
    run_button(app, stage).click().run()
    assert not app.exception
    assert store(app).get("unsaved_results")
    assert record(app, stage)["ai_verdict"] == "include"
    assert model.call_count == 1
    assert save.call_count == 1
    element(app, "button", "Retry saving results").click().run()
    assert not app.exception
    assert not store(app).get("unsaved_results")
    assert record(app, stage)["ai_verdict"] == "include"
    assert model.call_count == 1
    assert save.call_count == 2


@pytest.mark.parametrize("stage", STAGES)
def test_human_verdict_selection_belongs_to_paper_uid_not_list_position(screening_ui, stage):
    create, _, model, _ = screening_ui
    app = create(stage).run()
    app.radio[0].set_value("Include").run()
    assert app.radio[0].value == "Include"
    store(app)["papers"].reverse()
    app.run()
    assert not app.exception
    assert store(app)["papers"][0]["uid"] == "paper1"
    assert app.radio[0].value is None
    assert element(app, "button", "Confirm verdict").disabled
    assert not advancing_uids(app, stage)
    model.assert_not_called()


@pytest.mark.parametrize("scenario", ["pending", "ai"])
def test_corrupt_pdf_disables_review_and_never_reaches_provider(screening_ui, scenario):
    create, _, model, load_pdf = screening_ui
    load_pdf.return_value = b"%PDF-1.7\ndifferent bytes"
    app = create(state.STAGE_FULLTEXT, scenario).run()
    assert not app.exception
    assert any("differs from its metadata" in error.value for error in app.error)
    if scenario == "ai":
        assert element(app, "button", "✓ Agree").disabled
        assert element(app, "button", "✗ Disagree").disabled
    else:
        app.radio[0].set_value("Include").run()
        assert element(app, "button", "Confirm verdict").disabled
    element(app, "text_input", "API key").input("offline-openai-key").run()
    run_button(app, state.STAGE_FULLTEXT).click().run()
    assert not app.exception
    model.assert_not_called()
    assert not advancing_uids(app, state.STAGE_FULLTEXT)
    failed_paper = record(app, state.STAGE_FULLTEXT, 1 if scenario == "ai" else 0)
    assert "differs from its metadata" in failed_paper["ai_error"]


@pytest.mark.parametrize("stage", STAGES)
def test_provider_switch_does_not_reuse_another_providers_key(screening_ui, stage):
    create, _, model, _ = screening_ui
    app = create(stage).run()
    element(app, "text_input", "API key").input("offline-openai-key").run()
    element(app, "selectbox", "Provider").select("Anthropic").run()
    assert not app.exception
    assert element(app, "text_input", "API key").value == ""
    assert run_button(app, stage).disabled
    element(app, "text_input", "API key").input("offline-anthropic-key").run()
    run_button(app, stage).click().run()
    assert not app.exception
    assert model.call_count == 2
    assert all(call.args[0] == "Anthropic" and call.args[2] == "offline-anthropic-key"
               for call in model.call_args_list)
    assert "offline-openai-key" not in repr(store(app))
    assert "offline-anthropic-key" not in repr(store(app))


@pytest.mark.parametrize("stage", STAGES)
def test_jump_picker_moves_the_cursor_and_follows_prev_next(screening_ui, stage):
    create, _, _, _ = screening_ui
    app = create(stage).run()
    jump = element(app, "selectbox", "Jump to paper")
    assert jump.options == ["1. Study 0 — Pending", "2. Study 1 — Pending"]

    jump.select_index(1).run()
    assert not app.exception
    assert store(app)["cursors"][stage] == 1
    assert "Study 1" in " ".join(heading.value for heading in app.markdown)

    # Prev must not be dragged back by a picker still holding the old position.
    next(button for button in app.button if button.label.startswith("‹ Prev")).click().run()
    assert not app.exception
    assert store(app)["cursors"][stage] == 0
    assert element(app, "selectbox", "Jump to paper").value == 0
