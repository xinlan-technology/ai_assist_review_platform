"""Independent extraction runs, question-level sources, and explicit review saves."""
from copy import deepcopy

import pytest

from core import db
from core.llm import PROVIDERS
from features.extraction import schema, state as extraction
from features.workflow import state
from run_fixtures import install_run_store
from test_extraction_ui import ANSWER, DIGEST, QUESTION, element, record, ui


QUESTIONS = [QUESTION, {"id": "q2", "text": "Region?", "type": "open_text"}]
SPEC = schema.build_spec("Extract from the study only.", QUESTIONS)
ANSWERS_A = {"q1": ANSWER, "q2": dict(ANSWER, values=["Brazil"], quote="A site in Brazil.")}
ANSWERS_B = {
    "q1": dict(ANSWER, values=["Wetland"], quote="A wetland site."),
    "q2": dict(ANSWER, values=["Canada"], quote="A site in Canada."),
}


def candidate(answers, identifier="run-a", spec=SPEC, digest=DIGEST):
    paper = {}
    extraction.set_ai_result(paper, spec, digest, {"answers": answers}, "OpenAI", "model", 2)
    return {"id": identifier, "result": deepcopy(extraction.get(paper))}


def test_candidate_selection_is_read_only_and_returns_independent_values():
    run = candidate(ANSWERS_A)
    before = deepcopy(run)
    answer = extraction.candidate_answer(run, SPEC, DIGEST, "q1", 2)
    answer["values"].clear()
    assert run == before


@pytest.mark.parametrize("change", ["spec", "source", "prompt", "invalid"])
def test_candidate_rejects_incompatible_or_invalid_question(change):
    run = candidate(ANSWERS_A)
    if change == "spec":
        run["result"]["spec_hash"] = "old-setup"
    elif change == "source":
        run["result"]["source_sha256"] = "old-pdf"
    elif change == "prompt":
        run["result"]["prompt_version"] = "old-prompt"
    else:
        run["result"]["ai_answers"]["q1"]["page"] = 3
    with pytest.raises(ValueError):
        extraction.candidate_answer(run, SPEC, DIGEST, "q1", 2)


def test_mixed_question_confirmation_keeps_source_ids_and_previous_review():
    first, second = candidate(ANSWERS_A), candidate(ANSWERS_B, "run-b")
    mixed = {
        "q1": extraction.candidate_answer(first, SPEC, DIGEST, "q1", 2),
        "q2": extraction.candidate_answer(second, SPEC, DIGEST, "q2", 2),
    }
    paper = {}
    extraction.save_review(paper, SPEC, DIGEST, ANSWERS_A, 2, confirm=True)
    extraction.save_review(paper, SPEC, DIGEST, mixed, 2, confirm=True,
                           source_run_ids={"q1": "run-a", "q2": "run-b"}, source_answers=mixed)
    current = extraction.get(paper)
    assert current["source_run_ids"] == {"q1": "run-a", "q2": "run-b"}
    assert current["source_answers"] == mixed
    assert current["decisions"] == {"q1": "accepted", "q2": "accepted"}
    assert current["history"][-1]["final_answers"] == ANSWERS_A
    mixed["q2"]["values"].clear()
    assert current["final_answers"]["q2"]["values"] == ["Canada"]


def test_invalid_source_mapping_does_not_change_confirmed_review():
    paper = {}
    extraction.save_review(paper, SPEC, DIGEST, ANSWERS_A, 2, confirm=True)
    before = deepcopy(paper)
    with pytest.raises(ValueError, match="source runs"):
        extraction.save_review(paper, SPEC, DIGEST, ANSWERS_B, 2, confirm=True,
                               source_run_ids={"unknown": "run-b"})
    assert paper == before


@pytest.mark.parametrize("change", ["run_id", "secret", "question"])
def test_source_metadata_rejects_mismatches_and_non_audit_fields_atomically(change):
    paper = {}
    extraction.save_review(paper, SPEC, DIGEST, ANSWERS_A, 2, confirm=True)
    before = deepcopy(paper)
    metadata = {"q1": {"id": "run-a", "provider": "OpenAI", "model": "model-a", "completed_at": None}}
    if change == "run_id":
        metadata["q1"]["id"] = "wrong-run"
    elif change == "secret":
        metadata["q1"]["api_key"] = "never-store-this"
    else:
        metadata["unknown"] = metadata.pop("q1")
    with pytest.raises(ValueError, match="Source metadata"):
        extraction.save_review(paper, SPEC, DIGEST, ANSWERS_A, 2, confirm=True,
                               source_run_ids={"q1": "run-a"}, source_metadata=metadata)
    assert paper == before


def test_default_candidate_metadata_survives_review_edits_without_explicit_provenance():
    paper = {}
    extraction.set_ai_result(paper, SPEC, DIGEST, {"answers": ANSWERS_A}, "OpenAI", "model-a", 2)
    extraction.get(paper)["source_run_id"] = "run-a"
    extraction.save_review(paper, SPEC, DIGEST, ANSWERS_A, 2, confirm=True)
    before = deepcopy(extraction.get(paper)["source_metadata"])
    assert before["q1"]["id"] == before["q2"]["id"] == "run-a"
    assert before["q1"]["provider"] == "OpenAI"
    assert before["q1"]["model"] == "model-a"
    extraction.save_review(paper, SPEC, DIGEST, ANSWERS_B, 2, confirm=False)
    assert extraction.get(paper)["source_metadata"] == before
    assert extraction.get(paper)["source_run_ids"] == {"q1": "run-a", "q2": "run-a"}


@pytest.fixture
def comparison(ui, monkeypatch):
    create, save, model = ui
    ledger = install_run_store(monkeypatch)
    models = [{"provider": "OpenAI", "model": name, "api_key": "mock-key"}
              for name in list(PROVIDERS["OpenAI"])[:2]]
    from core import ui as core_ui
    monkeypatch.setattr(core_ui, "additional_models", lambda *args: deepcopy(models))
    model.side_effect = lambda provider, name, *args: {
        "answers": deepcopy(ANSWERS_A if name == models[0]["model"] else ANSWERS_B),
    }
    return create, save, model, ledger, models


def run_both(comparison):
    create, _, _, ledger, _ = comparison
    app = create(questions=QUESTIONS).run()
    element(app, "text_input", "API key").input("mock-key").run()
    element(app, "button", "▶ Run AI extraction (1 paper · 2 calls)").click().run()
    assert not app.exception
    assert len(ledger.rows) == 2
    return app


def test_default_extraction_run_compares_two_conflicting_models_without_review_write(comparison):
    app = run_both(comparison)
    _, _, model, ledger, _ = comparison
    assert model.call_count == 2
    before = deepcopy(record(app))
    source_selectors = [item for item in app.selectbox if item.label == "Use answer from"]
    assert len(source_selectors) == 2
    assert all(len(item.options) == 3 for item in source_selectors)
    source_selectors[0].set_value(ledger.rows[1]["id"]).run()
    assert not app.exception
    assert element(app, "selectbox", "Your answer").value == "Wetland"
    assert record(app) == before
    assert "review_state" not in record(app)
    tables = [item.value for item in app.dataframe if "Answer" in item.value.columns]
    assert any(set(table["Answer"]) == {"Forest", "Wetland"} for table in tables)


def test_supported_secondary_model_runs_when_primary_setup_is_blocked(comparison, monkeypatch):
    create, _, model, ledger, models = comparison
    models[1].update(provider="Anthropic", model=PROVIDERS["Anthropic"][0])
    monkeypatch.setattr(schema, "provider_issue", lambda provider, spec:
                        "This setup exceeds the provider limit." if provider == "OpenAI" else None)
    app = create(questions=QUESTIONS).run()
    element(app, "text_input", "API key").input("mock-key").run()
    element(app, "button", "▶ Run AI extraction (1 paper · 1 call)").click().run()
    assert not app.exception
    assert model.call_count == 1
    assert len(ledger.rows) == 1
    assert ledger.rows[0]["provider"] == "Anthropic"
    assert ledger.rows[0]["status"] == "succeeded"


def test_per_question_sources_confirm_mixed_answers_and_retain_on_new_run(comparison):
    app = run_both(comparison)
    _, _, model, ledger, _ = comparison
    source_selectors = [item for item in app.selectbox if item.label == "Use answer from"]
    source_selectors[1].set_value(ledger.rows[1]["id"]).run()
    for checkbox in app.checkbox:
        if checkbox.label == "Reviewed this question":
            checkbox.check()
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    current = record(app)
    assert current["review_state"] == "confirmed"
    assert current["final_answers"]["q1"]["values"] == ["Forest"]
    assert current["final_answers"]["q2"]["values"] == ["Canada"]
    assert current["source_run_ids"] == {"q1": ledger.rows[0]["id"], "q2": ledger.rows[1]["id"]}
    for qid, run in zip(("q1", "q2"), ledger.rows):
        assert current["source_metadata"][qid] == {
            field: run[field] for field in ("id", "provider", "model", "completed_at")
        }
    assert current["decisions"] == {"q1": "accepted", "q2": "accepted"}
    before = deepcopy(current)
    element(app, "toggle", "Model comparison and run history").set_value(True).run()
    element(app, "checkbox", "Repeat selected models").check().run()
    element(app, "button", "Run model comparison (2 calls)").click().run()
    assert not app.exception
    assert len(ledger.rows) == 4
    assert record(app) == before
    source_selectors = [item for item in app.selectbox if item.label == "Use answer from"]
    source_selectors[0].set_value(ledger.rows[1]["id"]).run()
    assert not app.exception
    assert record(app) == before
    assert model.call_count == 4


def test_candidate_edits_are_kept_as_a_draft_with_their_source(comparison):
    app = run_both(comparison)
    _, _, _, ledger, _ = comparison
    element(app, "selectbox", "Use answer from").set_value(ledger.rows[1]["id"]).run()
    # Choosing a source only fills the form; nothing is stored until it is edited or left.
    assert "review_state" not in record(app)
    element(app, "selectbox", "Your answer").select("Other").run()
    element(app, "text_input", "Explain Other").input("Grassland").run()
    assert record(app)["review_state"] == "draft"
    assert record(app)["source_run_ids"]["q1"] == ledger.rows[1]["id"]
    assert record(app)["final_answers"]["q1"]["other_text"] == "Grassland"
    element(app, "button", "Save draft").click().run()
    assert not app.exception
    assert record(app)["review_state"] == "draft"
    assert record(app)["source_run_ids"]["q1"] == ledger.rows[1]["id"]
    assert record(app)["final_answers"]["q1"]["values"] == ["Other"]
    assert record(app)["decisions"]["q1"] == "edited"


def test_current_answer_edits_retain_mixed_model_metadata_after_confirmation(comparison):
    app = run_both(comparison)
    _, _, _, ledger, _ = comparison
    selectors = [item for item in app.selectbox if item.label == "Use answer from"]
    selectors[1].set_value(ledger.rows[1]["id"]).run()
    for checkbox in app.checkbox:
        if checkbox.label == "Reviewed this question":
            checkbox.check()
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    assert not app.exception
    before = deepcopy(record(app))
    selectors = [item for item in app.selectbox if item.label == "Use answer from"]
    selectors[1].set_value("").run()
    element(app, "text_area", "Your answer").input("Mexico").run()
    assert record(app) == before
    element(app, "button", "Save draft").click().run()
    assert not app.exception
    assert record(app)["source_metadata"] == before["source_metadata"]
    assert record(app)["source_run_ids"] == before["source_run_ids"]
    assert record(app)["source_answers"] == before["source_answers"]
    assert record(app)["final_answers"]["q2"]["values"] == ["Mexico"]
    assert record(app)["decisions"]["q2"] == "edited"


def test_older_confirmation_recovers_initial_source_id_on_current_answer_edit(comparison):
    app = run_both(comparison)
    _, _, _, ledger, _ = comparison
    for checkbox in app.checkbox:
        if checkbox.label == "Reviewed this question":
            checkbox.check()
    app.run()
    element(app, "button", "Confirm extraction").click().run()
    record(app).pop("source_run_ids")
    record(app).pop("source_metadata")
    app.run()
    element(app, "selectbox", "Your answer").select("Wetland").run()
    element(app, "button", "Save draft").click().run()
    assert not app.exception
    initial = ledger.rows[0]
    assert record(app)["source_run_ids"] == {"q1": initial["id"], "q2": initial["id"]}
    assert record(app)["source_metadata"]["q1"]["model"] == initial["model"]
    assert record(app)["decisions"]["q1"] == "edited"


def test_incompatible_runs_are_history_only(comparison):
    create, _, _, ledger, _ = comparison
    app = create(questions=QUESTIONS)
    attempt = ledger.start(
        "test@example.com", "test-project", paper_uid="paper1", stage=state.STAGE_EXTRACTION,
        batch_id="old-batch", provider="OpenAI", model="old-model",
        config_hash=schema.spec_hash(SPEC), source_hash="old-pdf", prompt_version=SPEC["prompt_version"],
        prompt_snapshot=SPEC,
    )
    ledger.finish("test@example.com", "test-project", attempt["id"], status="succeeded",
                  result=candidate(ANSWERS_A, digest="old-pdf")["result"], duration_seconds=0.1)
    app.run()
    assert not app.exception
    assert not [item for item in app.selectbox if item.label == "Use answer from"]
    assert record(app) == {}
    assert any("old-model" in str(item.value) for item in app.dataframe)


def test_failed_run_completion_blocks_ui_and_retries_only_persistence(comparison):
    create, _, model, ledger, _ = comparison
    app = create(questions=QUESTIONS).run()
    element(app, "text_input", "API key").input("mock-key").run()
    finish = ledger.finish.side_effect
    ledger.finish.side_effect = db.DatabaseError("Temporary run persistence failure.")
    element(app, "button", "▶ Run AI extraction (1 paper · 2 calls)").click().run()
    assert not app.exception
    assert model.call_count == 1
    assert ledger.rows[0]["status"] == "running"
    assert app.session_state["project_store"]["pending_ai_completion"]["result"]["ai_answers"] == ANSWERS_A
    assert [button.label for button in app.button if not button.disabled] == ["Save pending AI answer"]
    app.run()
    assert model.call_count == 1
    assert [button.label for button in app.button if not button.disabled] == ["Save pending AI answer"]
    ledger.finish.side_effect = finish
    element(app, "button", "Save pending AI answer").click().run()
    assert not app.exception
    assert "pending_ai_completion" not in app.session_state["project_store"]
    assert ledger.rows[0]["status"] == "succeeded"
    assert model.call_count == 1
