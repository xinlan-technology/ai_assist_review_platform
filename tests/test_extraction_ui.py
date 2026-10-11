"""Offline Streamlit interaction tests with in-memory projects and mocked PDFs."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import auth, db, fulltext_storage, ui as core_ui
from features.extraction import judge, schema, state as extraction
from features.workflow import state
from run_fixtures import install_run_store


ROOT = Path(__file__).resolve().parents[1]
PDF = b"%PDF-1.7\nlocal mock PDF"
DIGEST = fulltext_storage.sha256(PDF)
QUESTION = {"id": "q1", "text": "Ecosystem?", "type": "single_choice", "options": ["Forest", "Wetland"]}
ANSWER = {"values": ["Forest"], "other_text": "", "page": 1, "quote": "A forest site.", "issue": ""}


def element(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


@pytest.fixture
def ui(monkeypatch):
    st.cache_data.clear()
    install_run_store(monkeypatch)
    monkeypatch.setattr(auth, "sidebar_user", lambda: None)
    monkeypatch.setattr(auth, "current_user", lambda: "test@example.com")
    monkeypatch.setattr(auth, "require_login", lambda: "test@example.com")
    monkeypatch.setattr(st, "pdf", lambda *args, **kwargs: st.caption("PDF viewer"))
    save = Mock(return_value=True)
    model = Mock(return_value={"answers": {"q1": ANSWER}})
    monkeypatch.setattr(state, "save_active", save)
    monkeypatch.setattr(judge, "extract_pdf", model)
    monkeypatch.setattr(fulltext_storage, "load_pdf", lambda key: PDF)
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        "paper1": {"storage_key": "mock/paper1.pdf", "sha256": DIGEST, "status": "ok",
                   "filename": "study.pdf", "file_size": len(PDF), "page_count": 2},
    })

    def create(questions=None, ai=False, empty=False, filename="views/extraction.py"):
        project = state.new_project_data(state.MODE_DIRECT)
        spec = schema.build_spec("Extract from the study only.", questions or [QUESTION])
        if not empty:
            project["config"].update(extraction_instructions=spec["instructions"], extraction_questions=spec["questions"])
        paper = state.new_paper("", "Test study", "")
        paper["uid"] = "paper1"
        state.set_human_verdict(paper, state.STAGE_FULLTEXT, state.VERDICT_INCLUDE,
                                review_hash=state.criteria_hash(project["config"]["fulltext_criteria"]))
        if ai:
            extraction.set_ai_result(paper, spec, DIGEST, {"answers": {"q1": ANSWER}}, "OpenAI", "model", 2)
        project["papers"] = [paper]
        app = AppTest.from_file(str(ROOT / filename), default_timeout=10)
        app.session_state["active_project_id"] = "test-project"
        app.session_state["active_project_name"] = "Test project"
        app.session_state["project_store"] = {
            "mode": project["mode"], "config": project["config"], "papers": project["papers"],
            "original_df": pd.DataFrame(), "cursors": {},
        }
        return app

    return create, save, model


def record(app):
    return extraction.get(app.session_state["project_store"]["papers"][0])


def test_builder_saves_canonical_questions_with_reserved_options(ui):
    create, _, _ = ui
    app = create(empty=True).run()
    assert not app.exception
    element(app, "button", "+ Add question").click().run()
    element(app, "text_input", "Question").input("Study setting?")
    element(app, "text_area", "Options (one per line)").input("Forest\nWetland\n\nForest")
    app.run()
    element(app, "button", "Save extraction setup").click().run()
    assert not app.exception
    questions = app.session_state["project_store"]["config"]["extraction_questions"]
    assert questions[0]["text"] == "Study setting?"
    assert questions[0]["options"] == ["Forest", "Wetland", "Other", "Not reported"]
    assert questions[0]["id"]


def test_ai_answer_requires_review_then_confirms_without_second_call(ui):
    create, _, model = ui
    app = create().run()
    element(app, "text_input", "API key").input("mock-key").run()
    element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert not app.exception
    assert model.call_count == 1
    assert element(app, "button", "Confirm extraction").disabled
    element(app, "checkbox", "Reviewed this question").check().run()
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    assert record(app)["review_state"] == "confirmed"
    assert record(app)["decisions"] == {"q1": "accepted"}
    assert element(app, "button", "▶ Run AI extraction (0 papers · 0 calls)").disabled
    assert model.call_count == 1


def test_human_other_requires_explanation_and_evidence(ui):
    create, _, model = ui
    app = create().run()
    element(app, "selectbox", "Your answer").select("Other").run()
    element(app, "checkbox", "Reviewed this question").check().run()
    assert element(app, "button", "Confirm extraction").disabled
    element(app, "text_input", "Explain Other").input("Grassland")
    element(app, "number_input", "Evidence PDF page").set_value(2)
    element(app, "text_area", "Supporting quote").input("The site was grassland.")
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    assert record(app)["final_answers"]["q1"]["other_text"] == "Grassland"
    assert record(app)["decisions"] == {"q1": "human"}
    model.assert_not_called()


def test_setup_change_requires_acknowledgment_and_archive_unlocks_manual_review(ui):
    create, _, _ = ui
    app = create(ai=True).run()
    element(app, "text_input", "Question").input("Main ecosystem?").run()
    assert element(app, "button", "Save extraction setup").disabled
    element(app, "checkbox", "I understand that changing the setup makes existing extraction results outdated.").check().run()
    element(app, "button", "Save extraction setup").click().run()
    assert not app.exception
    assert any("outdated" in item.value.lower() for item in app.warning)
    assert not any(item.label == "Confirm extraction" for item in app.button)
    element(app, "button", "Archive outdated without AI (1)").click().run()
    assert not app.exception
    assert record(app)["history"][-1]["spec"]["questions"][0]["text"] == "Ecosystem?"
    assert element(app, "selectbox", "Your answer").value is None


def test_unsaved_ai_result_blocks_work_and_retry_only_saves(ui):
    create, save, model = ui
    app = create().run()
    element(app, "text_input", "API key").input("mock-key").run()
    save.side_effect = [True, False]
    element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert not app.exception
    assert app.session_state["project_store"].get("unsaved_results") is True
    assert [button.label for button in app.button if not button.disabled] == ["Retry saving results"]
    assert len(app.get("download_button")) == 1
    save.side_effect = None
    element(app, "button", "Retry saving results").click().run()
    assert not app.exception
    assert record(app)["ai_answers"]["q1"] == ANSWER
    assert model.call_count == 1


def test_failed_review_save_rolls_back_confirmation(ui):
    create, save, _ = ui
    app = create(ai=True).run()
    element(app, "checkbox", "Reviewed this question").check().run()
    # The ticked review box is itself kept as draft progress.
    before = deepcopy(record(app))
    assert before["review_state"] == "draft" and before["checked_questions"] == ["q1"]
    save.return_value = False
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    assert record(app) == before
    assert any("not saved" in error.value for error in app.error)


@pytest.mark.parametrize("ai", [False, True])
def test_archiving_draft_explicitly_reenables_ai_without_automatic_call(ui, ai):
    create, _, model = ui
    app = create(ai=ai).run()
    element(app, "button", "Save draft").click().run()
    draft = deepcopy(record(app))
    element(app, "text_input", "API key").input("mock-key").run()
    assert element(app, "button", "▶ Run AI extraction (0 papers · 0 calls)").disabled
    element(app, "button", "Archive draft and start over").click().run()
    assert not app.exception
    assert set(record(app)) - {"form_nonce"} == {"history"}
    archived = record(app)["history"][-1]
    assert archived["final_answers"] == draft["final_answers"]
    assert archived["review_state"] == "draft"
    if ai:
        assert archived["ai_answers"] == draft["ai_answers"]
    assert not element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").disabled
    model.assert_not_called()
    element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert not app.exception
    assert model.call_count == 1
    assert record(app)["ai_answers"]["q1"] == ANSWER


def test_failed_draft_archive_save_restores_current_draft(ui):
    create, save, model = ui
    app = create(ai=True).run()
    element(app, "button", "Save draft").click().run()
    draft = deepcopy(record(app))
    save.return_value = False
    element(app, "button", "Archive draft and start over").click().run()
    assert not app.exception
    assert record(app) == draft
    assert any("not saved" in error.value for error in app.error)
    model.assert_not_called()


def test_app_navigation_is_blocked_until_unsaved_results_persist(ui):
    create, save, _ = ui
    app = create(filename="app.py")
    app.session_state["project_store"]["unsaved_results"] = True
    save.return_value = False
    app.run()
    assert not app.exception
    assert [button.label for button in app.button if not button.disabled] == ["Retry saving results"]
    assert len(app.get("download_button")) == 1
    element(app, "button", "Retry saving results").click().run()
    assert app.session_state["project_store"].get("unsaved_results") is True


def test_reordering_preserves_question_ids_and_current_edits(ui):
    create, _, _ = ui
    questions = [QUESTION, {"id": "q2", "text": "Region?", "type": "open_text"}]
    app = create(questions=questions).run()
    app.text_input(key="extraction:test-project:question:q1:text").input("Main ecosystem?")
    app.button(key="extraction:test-project:question:q2:up").click().run()
    element(app, "button", "Save extraction setup").click().run()
    assert not app.exception
    saved = app.session_state["project_store"]["config"]["extraction_questions"]
    assert [q["id"] for q in saved] == ["q2", "q1"]
    assert saved[1]["text"] == "Main ecosystem?"


def test_multiple_choice_exclusive_exit_and_open_text_not_reported(ui):
    create, _, model = ui
    questions = [dict(QUESTION, type="multiple_choice"), {"id": "q2", "text": "Region?", "type": "open_text"}]
    app = create(questions=questions).run()
    element(app, "multiselect", "Your answers").set_value(["Forest", "Not reported"])
    element(app, "checkbox", "Not reported").check()
    for checkbox in app.checkbox:
        if checkbox.label == "Reviewed this question":
            checkbox.check()
    app.run()
    assert element(app, "button", "Confirm extraction").disabled
    element(app, "multiselect", "Your answers").set_value(["Forest"])
    app.number_input[0].set_value(1)
    next(t for t in app.text_area if t.label == "Supporting quote").input("A forest site.")
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    assert record(app)["final_answers"]["q2"]["values"] == ["Not reported"]
    assert record(app)["final_answers"]["q2"]["page"] is None
    model.assert_not_called()


def test_invalid_model_answer_is_visible_and_never_auto_retried(ui):
    create, _, model = ui
    model.return_value = {"answers": {}}
    app = create().run()
    element(app, "text_input", "API key").input("mock-key").run()
    element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert not app.exception
    assert record(app)["field_errors"]
    assert element(app, "button", "▶ Run AI extraction (0 papers · 0 calls)").disabled
    assert element(app, "selectbox", "Your answer").value is None
    assert not element(app, "button", "↻ Retry invalid responses (1 paper · 1 call, may be billed again)").disabled
    assert model.call_count == 1


def test_mismatched_pdf_disables_human_confirmation(ui, monkeypatch):
    create, _, model = ui
    monkeypatch.setattr(fulltext_storage, "load_pdf", lambda key: b"different document")
    app = create(ai=True).run()
    assert not app.exception
    assert any("differs" in e.value for e in app.error)
    assert element(app, "button", "Confirm extraction").disabled
    assert element(app, "button", "Save draft").disabled
    model.assert_not_called()


def test_provider_limit_blocks_ai_but_keeps_manual_review_available(ui, monkeypatch):
    create, _, model = ui
    monkeypatch.setattr(schema, "provider_issue", lambda provider, spec: "Reduce options before using this provider.")
    app = create().run()
    element(app, "text_input", "API key").input("mock-key").run()
    assert not app.exception
    assert element(app, "button", "▶ Run AI extraction (0 papers · 0 calls)").disabled
    assert not element(app, "button", "Save draft").disabled
    assert not element(app, "selectbox", "Your answer").disabled
    assert any("Reduce options" in warning.value for warning in app.warning)
    model.assert_not_called()


def test_unsaved_answers_survive_a_rerun(ui):
    create, _, _ = ui
    app = create(ai=True).run()
    element(app, "text_area", "Supporting quote").input("my own evidence").run()
    element(app, "text_area", "Extraction instructions").input("a changed instruction").run()
    assert not app.exception
    saved = record(app)
    assert saved["review_state"] == "draft"
    assert saved["final_answers"]["q1"]["quote"] == "my own evidence"


def test_autosave_validation_failure_restores_data_without_blocking_recovery(ui, monkeypatch):
    create, save, model = ui
    app = create(ai=True).run()
    before = deepcopy(record(app))

    def invalid(paper, *args, **kwargs):
        extraction.get(paper)["review_state"] = "draft"
        raise ValueError("Invalid draft")

    monkeypatch.setattr(extraction, "save_review", invalid)
    element(app, "text_area", "Supporting quote").input("Edited evidence.").run()
    assert not app.exception
    assert record(app) == before
    assert not app.session_state["project_store"].get("unsaved_results")
    save.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("action", ["next", "run", "setup"])
def test_failed_autosave_blocks_actions_and_retries_without_model_calls(ui, monkeypatch, action):
    create, save, model = ui
    app = create(ai=True)
    store = app.session_state["project_store"]
    second = state.new_paper("", "Second study", "")
    second["uid"] = "paper2"
    state.set_human_verdict(second, state.STAGE_FULLTEXT, state.VERDICT_INCLUDE,
                            review_hash=state.criteria_hash(store["config"]["fulltext_criteria"]))
    store["papers"].append(second)
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        uid: {"storage_key": f"mock/{uid}.pdf", "sha256": DIGEST, "status": "ok",
              "filename": "study.pdf", "file_size": len(PDF), "page_count": 2}
        for uid in ("paper1", "paper2")})
    app.run()
    if action == "run":
        element(app, "text_input", "API key").input("mock-key").run()
    elif action == "setup":
        element(app, "text_area", "Extraction instructions").input("Changed instructions.").run()
        element(app, "checkbox", "I understand that changing the setup makes existing extraction results outdated.").check().run()
    before_config = deepcopy(store["config"])
    save.return_value = False
    calls_before = save.call_count
    element(app, "text_area", "Supporting quote").set_value("My unsaved evidence.")
    label = {"next": "Next ›", "run": "▶ Run AI extraction (1 paper · 1 call)", "setup": "Save extraction setup"}[action]
    element(app, "button", label).click().run()

    assert not app.exception
    assert app.session_state["project_store"].get("unsaved_results") is True
    assert record(app)["final_answers"]["q1"]["quote"] == "My unsaved evidence."
    assert record(app)["review_state"] == "draft"
    assert [button.label for button in app.button if not button.disabled] == ["Retry saving results"]
    assert len(app.get("download_button")) == 1
    assert store["cursors"].get(state.STAGE_EXTRACTION, 0) == 0
    assert store["config"] == before_config
    assert save.call_count == calls_before + 1
    model.assert_not_called()

    app.run()
    assert save.call_count == calls_before + 1
    assert [button.label for button in app.button if not button.disabled] == ["Retry saving results"]
    element(app, "button", "Retry saving results").click().run()
    assert app.session_state["project_store"].get("unsaved_results") is True
    assert save.call_count == calls_before + 2
    save.return_value = True
    element(app, "button", "Retry saving results").click().run()

    assert not app.exception
    assert not app.session_state["project_store"].get("unsaved_results")
    assert record(app)["final_answers"]["q1"]["quote"] == "My unsaved evidence."
    assert store["cursors"].get(state.STAGE_EXTRACTION, 0) == 0
    assert store["config"] == before_config
    assert save.call_count == calls_before + 3
    model.assert_not_called()


def test_navigating_to_another_paper_keeps_typed_answers(ui, monkeypatch):
    create, _, _ = ui
    app = create(ai=True)
    store = app.session_state["project_store"]
    second = state.new_paper("", "Second study", "")
    second["uid"] = "paper2"
    state.set_human_verdict(second, state.STAGE_FULLTEXT, state.VERDICT_INCLUDE,
                            review_hash=state.criteria_hash(store["config"]["fulltext_criteria"]))
    store["papers"].append(second)
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        uid: {"storage_key": f"mock/{uid}.pdf", "sha256": DIGEST, "status": "ok",
              "filename": "study.pdf", "file_size": len(PDF), "page_count": 2}
        for uid in ("paper1", "paper2")})
    app.run()
    element(app, "text_area", "Supporting quote").input("hand written evidence").run()
    element(app, "button", "Next ›").click().run()
    assert not app.exception
    assert extraction.get(store["papers"][0])["final_answers"]["q1"]["quote"] == "hand written evidence"


def test_ai_run_is_not_mistaken_for_a_human_edit(ui):
    """An empty form drawn before an AI run is not the reviewer's draft."""
    create, _, _ = ui
    app = create().run()
    element(app, "text_input", "API key").input("mock-key").run()
    element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert not app.exception
    saved = record(app)
    assert saved["ai_answers"]["q1"]["values"] == ["Forest"]
    assert "review_state" not in saved


def test_untouched_empty_form_is_never_stored(ui):
    create, _, _ = ui
    app = create().run()
    element(app, "text_area", "Extraction instructions").input("a changed instruction").run()
    assert not app.exception
    assert record(app) == {}


@pytest.mark.parametrize("kind", ["multiple_choice", "open_text"])
def test_clearing_ai_not_reported_autosaves_empty_draft_across_navigation(ui, monkeypatch, kind):
    create, save, model = ui
    question = dict(QUESTION, type=kind)
    app = create(questions=[question])
    store = app.session_state["project_store"]
    paper = store["papers"][0]
    spec = schema.build_spec("Extract from the study only.", [question])
    extraction.set_ai_result(paper, spec, DIGEST, {"answers": {"q1": {
        "values": ["Not reported"], "other_text": "", "page": None, "quote": "", "issue": "",
    }}}, "OpenAI", "model", 2)
    second = state.new_paper("", "Second study", "")
    second["uid"] = "paper2"
    state.set_human_verdict(second, state.STAGE_FULLTEXT, state.VERDICT_INCLUDE,
                            review_hash=state.criteria_hash(store["config"]["fulltext_criteria"]))
    store["papers"].append(second)
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        uid: {"storage_key": f"mock/{uid}.pdf", "sha256": DIGEST, "status": "ok",
              "filename": "study.pdf", "file_size": len(PDF), "page_count": 2}
        for uid in ("paper1", "paper2")})
    app.run()
    save.assert_not_called()

    if kind == "multiple_choice":
        element(app, "multiselect", "Your answers").set_value([]).run()
    else:
        element(app, "checkbox", "Not reported").uncheck().run()

    assert not app.exception
    assert record(app)["review_state"] == "draft"
    assert record(app)["final_answers"]["q1"]["values"] == []
    assert save.call_count == 1
    element(app, "button", "Next ›").click().run()
    element(app, "button", "‹ Prev").click().run()

    assert not app.exception
    if kind == "multiple_choice":
        assert element(app, "multiselect", "Your answers").value == []
    else:
        assert not element(app, "checkbox", "Not reported").value
        assert element(app, "text_area", "Your answer").value == ""
    assert record(app)["final_answers"]["q1"]["values"] == []
    assert extraction.get(second) == {}
    assert save.call_count == 1
    model.assert_not_called()


def test_editing_and_confirming_in_one_interaction_still_confirms(ui):
    """Autosave must preserve widget identity through the confirmation click."""
    create, _, _ = ui
    app = create(ai=True).run()
    element(app, "checkbox", "Reviewed this question").check().run()
    element(app, "text_area", "Supporting quote").set_value("my own evidence")
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    saved = record(app)
    assert saved["review_state"] == "confirmed"
    assert saved["final_answers"]["q1"]["quote"] == "my own evidence"


def test_ai_not_reported_with_evidence_is_not_rewritten_as_a_human_draft(ui):
    create, _, _ = ui
    app = create().run()
    paper = app.session_state["project_store"]["papers"][0]
    spec = schema.build_spec("Extract from the study only.", [QUESTION])
    extraction.set_ai_result(paper, spec, DIGEST, {"answers": {"q1": {
        "values": ["Not reported"], "other_text": "", "page": 7,
        "quote": "No funding statement appears.", "issue": "",
    }}}, "OpenAI", "model", 2)
    app.run()

    element(app, "text_area", "Extraction instructions").input("an unrelated edit").run()

    assert not app.exception
    assert "review_state" not in record(app)


def test_a_read_only_form_never_autosaves(ui, monkeypatch):
    """A PDF hash mismatch blocks recording answers."""
    create, _, _ = ui
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        "paper1": {"storage_key": "mock/paper1.pdf", "sha256": "0" * 64, "status": "ok",
                   "filename": "study.pdf", "file_size": len(PDF), "page_count": 2},
    })
    app = create(ai=True).run()

    element(app, "text_area", "Extraction instructions").input("an unrelated edit").run()

    assert not app.exception
    assert "review_state" not in record(app)


def test_untrusted_extraction_content_is_displayed_literally(ui, monkeypatch):
    create, _, _ = ui

    def payload(field):
        return f'![{field}](https://example.invalid/{field}) <img src="https://example.invalid/pixel">'

    fields = {name: payload(name) for name in (
        "project", "title", "question", "guidance", "filename", "answer", "other", "quote", "error", "issue",
    )}
    questions = [
        dict(QUESTION, text=fields["question"], guidance=fields["guidance"]),
        {"id": "q2", "text": "Reported outcome?", "type": "open_text"},
    ]
    app = create(questions=questions)
    app.session_state["active_project_name"] = fields["project"]
    app.session_state["extraction:test-project:autosaved"] = fields["title"]
    paper = app.session_state["project_store"]["papers"][0]
    paper["title"] = fields["title"]
    spec = schema.build_spec("Extract from the study only.", questions)
    extraction.set_ai_result(paper, spec, DIGEST, {"answers": {
        "q1": dict(ANSWER, values=["Other"], other_text=fields["other"], quote=fields["quote"]),
        "q2": dict(ANSWER, values=[fields["answer"]]),
    }}, "OpenAI", "model", 2)
    extraction.get(paper).update(ai_error=fields["error"], field_errors={"q1": fields["issue"]})
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: {
        "paper1": {"storage_key": "mock/paper1.pdf", "sha256": DIGEST, "status": "ok",
                   "filename": fields["filename"], "file_size": len(PDF), "page_count": 2},
    })

    app.run()

    assert not app.exception
    rich_text = [item.value for kind in ("markdown", "caption", "error", "warning", "info")
                 for item in getattr(app, kind)] + [item.label for item in app.expander]
    for name in ("project", "title", "question", "guidance", "filename", "other", "error", "issue"):
        assert any(core_ui.escape_markdown(fields[name]) in text for text in rich_text)
    assert all(value not in text for value in fields.values() for text in rich_text)
    plain_text = [item.value for item in app.text]
    assert fields["answer"] in plain_text
    assert fields["quote"] in plain_text
    assert element(app, "text_input", "Question").value == fields["question"]
    assert record(app)["ai_answers"]["q2"]["values"] == [fields["answer"]]


@pytest.mark.parametrize("source", ["database", "storage"])
def test_extraction_read_errors_cannot_render_remote_images(ui, monkeypatch, source):
    create, _, _ = ui
    message = "![request](https://example.invalid/error)"

    def fail(*args):
        error = db.DatabaseError if source == "database" else fulltext_storage.FulltextStorageError
        raise error(message)

    if source == "database":
        monkeypatch.setattr(db, "load_fulltexts", fail)
    else:
        monkeypatch.setattr(fulltext_storage, "load_pdf", fail)

    app = create().run()

    assert not app.exception
    assert any(item.value == core_ui.escape_markdown(message) for item in app.error)
