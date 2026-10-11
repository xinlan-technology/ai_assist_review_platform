"""Offline comparisons preserve attempts without silently changing decisions."""
from copy import deepcopy

import pytest

from core import db, fulltext_storage, ui
from core.llm import InvalidModelResponse
from features.workflow import state
from run_fixtures import install_run_store
from test_screening_ui import (
    PROJECT_ID, STAGES, advancing_uids, element, record, screening_ui, store,
)


@pytest.fixture
def comparison_ui(screening_ui, monkeypatch):
    create, save, model, _ = screening_ui
    persistence = install_run_store(monkeypatch)

    def render(stage, scenario="pending", *, second_provider="OpenAI"):
        app = create(stage, scenario)
        store(app)["papers"] = store(app)["papers"][:1]
        app.run()
        element(app, "text_input", "API key").input("offline-openai-key").run()
        element(app, "checkbox", "Compare multiple models").check().run()
        element(app, "selectbox", "Provider 2").select(second_provider).run()
        if second_provider == "OpenAI":
            element(app, "selectbox", "Model 2").select("gpt-4.1").run()
        element(app, "toggle", "Model comparison and run history").set_value(True).run()
        assert not app.exception
        return app

    return render, save, model, persistence


def comparison_button(app, stage):
    return app.button(key=f"runs:{stage}:{PROJECT_ID}:run")


@pytest.mark.parametrize("stage", STAGES)
def test_two_models_save_separate_answers_and_only_confirmation_advances(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    model.side_effect = lambda provider, name, *args: {
        "verdict": "exclude" if name == "gpt-4.1" else "include",
        "reason": f"Independent assessment from {name}.",
    }
    app = render(stage)
    assert comparison_button(app, stage).label == "Run model comparison (2 calls)"
    comparison_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 2
    assert len(persistence.rows) == 2
    assert len({row["id"] for row in persistence.rows}) == 2
    assert {row["result"]["ai_verdict"] for row in persistence.rows} == {"include", "exclude"}
    assert all(row["status"] == "succeeded" for row in persistence.rows)
    assert record(app, stage)["ai_verdict"] == "include"
    assert not advancing_uids(app, stage)
    assert any("Models disagree" in warning.value for warning in app.warning)
    assert comparison_button(app, stage).disabled
    assert {"Download all AI runs (JSON)", "Download all AI runs (CSV)"} <= {
        download.label for download in app.get("download_button")
    }

    before = deepcopy(record(app, stage))
    element(app, "selectbox", "Reference AI result").select_index(1).run()
    assert record(app, stage) == before
    assert not advancing_uids(app, stage)
    element(app, "button", "Confirm final verdict").click().run()

    assert not app.exception
    assert record(app, stage)["source_run_id"] == persistence.rows[1]["id"]
    assert record(app, stage)["ai_verdict"] == "exclude"
    assert state.final_verdict(store(app)["papers"][0], stage) == "exclude"
    assert state.is_reviewed(store(app)["papers"][0], stage)
    assert model.call_count == 2


@pytest.mark.parametrize("stage", STAGES)
def test_explicit_repeat_appends_history_and_preserves_human_verdict(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    model.return_value = {"verdict": "exclude", "reason": "An additional opinion."}
    app = render(stage, "confirmed_override")
    original = deepcopy(record(app, stage))
    eligible = advancing_uids(app, stage)
    comparison_button(app, stage).click().run()
    assert not app.exception
    assert record(app, stage) == original
    assert advancing_uids(app, stage) == eligible
    first_attempts = deepcopy(persistence.rows)
    app.run()
    assert model.call_count == 2

    element(app, "checkbox", "Repeat selected models").check().run()
    assert not comparison_button(app, stage).disabled
    comparison_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 4
    assert len(persistence.rows) == 4
    assert persistence.rows[:2] == first_attempts
    assert len({row["id"] for row in persistence.rows}) == 4
    assert record(app, stage) == original
    assert advancing_uids(app, stage) == eligible
    assert "offline-openai-key" not in repr(persistence.rows)
    assert "offline-openai-key" not in repr(store(app))


@pytest.mark.parametrize("stage", STAGES)
def test_comparison_requires_each_provider_key_and_keeps_keys_out_of_history(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    app = render(stage, second_provider="Anthropic")
    assert element(app, "text_input", "API key · Anthropic").value == ""
    assert comparison_button(app, stage).disabled
    model.assert_not_called()

    element(app, "text_input", "API key · Anthropic").input("offline-anthropic-key").run()
    comparison_button(app, stage).click().run()

    assert not app.exception
    assert [(call.args[0], call.args[2]) for call in model.call_args_list] == [
        ("OpenAI", "offline-openai-key"), ("Anthropic", "offline-anthropic-key"),
    ]
    assert len(persistence.rows) == 2
    for key in ("offline-openai-key", "offline-anthropic-key"):
        assert key not in repr(persistence.rows)
        assert key not in repr(store(app))


@pytest.mark.parametrize("stage", STAGES)
def test_comparison_start_failure_never_calls_a_provider(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    app = render(stage)
    persistence.start.side_effect = db.DatabaseError("Offline run-start failure")
    comparison_button(app, stage).click().run()
    assert not app.exception
    assert any(ui.escape_markdown("Offline run-start failure") in error.value for error in app.error)
    assert not persistence.rows
    model.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_failed_run_history_checkpoint_retries_saving_without_paid_repeat(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    app = render(stage)
    finish = persistence.finish.side_effect
    persistence.finish.side_effect = db.DatabaseError("Offline run checkpoint failure")
    comparison_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 1
    assert len(persistence.rows) == 1
    assert persistence.rows[0]["status"] == "running"
    assert store(app)["pending_ai_completion"]["result"]["ai_verdict"] == "include"
    app.run()
    assert model.call_count == 1
    element(app, "button", "Save pending AI answer").click().run()
    assert store(app)["pending_ai_completion"]
    assert model.call_count == 1

    persistence.finish.side_effect = finish
    element(app, "button", "Save pending AI answer").click().run()
    assert not app.exception
    assert "pending_ai_completion" not in store(app)
    assert persistence.rows[0]["status"] == "succeeded"
    assert model.call_count == 1
    assert len(persistence.rows) == 1


@pytest.mark.parametrize("stage", STAGES)
def test_history_read_failure_is_literal_and_does_not_start_paid_work(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    app = render(stage)
    message = "![remote image](https://example.invalid/history-error)"
    persistence.load.side_effect = db.DatabaseError(message)
    app.run()
    assert not app.exception
    assert any(error.value == ui.escape_markdown(message) for error in app.error)
    assert message not in [error.value for error in app.error]
    persistence.start.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize("stage", STAGES)
def test_three_models_count_calls_and_preserve_each_provider_identity(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    app = render(stage)
    element(app, "selectbox", "Number of models").select(3).run()
    # A duplicate is not another model and must not trigger duplicate charges.
    element(app, "selectbox", "Provider 3").select("OpenAI").run()
    element(app, "selectbox", "Model 3").select("gpt-4.1-mini").run()
    assert comparison_button(app, stage).disabled
    element(app, "selectbox", "Provider 3").select("Google").run()
    assert comparison_button(app, stage).disabled
    element(app, "text_input", "API key · Google").input("offline-google-key").run()
    assert comparison_button(app, stage).label == "Run model comparison (3 calls)"
    comparison_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 3
    assert [(row["provider"], row["model"]) for row in persistence.rows] == [
        (call.args[0], call.args[1]) for call in model.call_args_list
    ]
    assert len({row["batch_id"] for row in persistence.rows}) == 1
    assert model.call_args_list[-1].args[2] == "offline-google-key"
    assert "offline-google-key" not in repr(persistence.rows)


@pytest.mark.parametrize("stage", STAGES)
def test_invalid_answer_stays_in_history_without_automatic_retry(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    model.side_effect = [
        InvalidModelResponse("Invalid answer with offline-openai-key"),
        {"verdict": "include", "reason": "A valid second opinion."},
    ]
    app = render(stage)
    comparison_button(app, stage).click().run()
    assert not app.exception
    assert [row["status"] for row in persistence.rows] == ["invalid_response", "succeeded"]
    assert "offline-openai-key" not in repr(persistence.rows)
    assert "[redacted]" in persistence.rows[0]["result"]["ai_error"]
    assert comparison_button(app, stage).disabled
    app.run()
    assert model.call_count == 2
    assert len(persistence.rows) == 2


@pytest.mark.parametrize("stage", STAGES)
def test_prompt_changes_make_old_runs_unselectable_without_deleting_history(comparison_ui, stage):
    render, _, model, persistence = comparison_ui
    app = render(stage)
    comparison_button(app, stage).click().run()
    before = deepcopy(persistence.rows)
    prompt = "Abstract prompt" if stage == state.STAGE_ABSTRACT else "Full-text prompt"
    element(app, "text_area", prompt).input("Updated screening requirements.").run()

    assert not app.exception
    assert persistence.rows == before
    assert model.call_count == 2
    assert not any(control.label == "Reference AI result" for control in app.selectbox)
    assert not any(button.label == "Confirm final verdict" for button in app.button)
    assert comparison_button(app, stage).label == "Run model comparison (2 calls)"
    assert not comparison_button(app, stage).disabled


@pytest.mark.parametrize("stage", STAGES)
def test_failed_project_checkpoint_blocks_comparison_before_run_start(comparison_ui, stage):
    render, save, model, persistence = comparison_ui
    app = render(stage)
    save.side_effect = db.DatabaseError("Offline project checkpoint failure")
    comparison_button(app, stage).click().run()
    assert not app.exception
    assert not persistence.rows
    persistence.start.assert_not_called()
    model.assert_not_called()


def test_replaced_pdf_cannot_reuse_or_adopt_old_run_answers(comparison_ui, monkeypatch):
    render, _, model, persistence = comparison_ui
    stage = state.STAGE_FULLTEXT
    app = render(stage)
    comparison_button(app, stage).click().run()
    before = deepcopy(persistence.rows)
    metadata = db.load_fulltexts("screening@example.invalid", PROJECT_ID)
    replacement = b"%PDF-1.7\nreplacement evidence"
    metadata["paper0"]["sha256"] = fulltext_storage.sha256(replacement)
    metadata["paper0"]["file_size"] = len(replacement)
    monkeypatch.setattr(db, "load_fulltexts", lambda *args: deepcopy(metadata))
    monkeypatch.setattr(fulltext_storage, "load_pdf", lambda *args: replacement)
    app.run()

    assert not app.exception
    assert model.call_count == 2
    assert persistence.rows == before
    assert not any(control.label == "Reference AI result" for control in app.selectbox)
    assert not any(button.label == "Confirm final verdict" for button in app.button)
    assert comparison_button(app, stage).label == "Run model comparison (2 calls)"


@pytest.mark.parametrize("stage", STAGES)
def test_comparison_respects_selected_papers(screening_ui, monkeypatch, stage):
    create, _, model, _ = screening_ui
    persistence = install_run_store(monkeypatch)
    app = create(stage).run()
    element(app, "text_input", "API key").input("offline-openai-key").run()
    element(app, "checkbox", "Compare multiple models").check().run()
    element(app, "selectbox", "Provider 2").select("OpenAI").run()
    element(app, "selectbox", "Model 2").select("gpt-4.1").run()
    element(app, "toggle", "Model comparison and run history").set_value(True).run()
    assert comparison_button(app, stage).label == "Run model comparison (4 calls)"
    element(app, "checkbox", "All papers in this stage").uncheck().run()
    assert comparison_button(app, stage).label == "Run model comparison (0 calls)"
    element(app, "multiselect", "Papers to compare").set_value(["paper1"]).run()
    assert comparison_button(app, stage).label == "Run model comparison (2 calls)"
    comparison_button(app, stage).click().run()

    assert not app.exception
    assert model.call_count == 2
    assert len(persistence.rows) == 2
    assert {row["paper_uid"] for row in persistence.rows} == {"paper1"}
    assert not record(app, stage).get("ai_verdict")
    assert record(app, stage, 1)["ai_verdict"] == "include"
