"""Shared multi-model runner integration with real SQLite and mocked providers."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.runtime.state import session_state_proxy
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.state.session_state import SessionState

from core import db
from core.llm import InvalidModelResponse, MAX_INLINE_PDF_BYTES
from features.extraction import schema, state as extraction
from features.screening import prompts
from features.workflow import run_controls, runs, state


SPEC = {"criteria": "Include field studies.", "prompt_version": prompts.ABSTRACT_PROMPT_VERSION}
MODELS = [
    {"provider": "OpenAI", "model": "gpt-4.1-mini", "api_key": "offline-openai-secret"},
    {"provider": "Anthropic", "model": "claude-haiku-4-5", "api_key": "offline-anthropic-secret"},
]
PDF = b"%PDF-1.7\nmock-document"
EXTRACTION_SPEC = schema.build_spec("Find the setting.", [{
    "id": "setting", "text": "Setting?", "type": "open_text", "options": [], "guidance": "",
}])
ANSWER = {"setting": {"values": ["Forest"], "other_text": "", "page": 1,
                      "quote": "The forest study site.", "issue": ""}}


@pytest.fixture
def project(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'integration.sqlite'}")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    control, errors = {}, []
    raw = SessionState()

    def on_yield():
        interruption = control.pop("interrupt", None)
        if interruption is not None:
            raise interruption

    safe = SafeSessionState(raw, on_yield)
    monkeypatch.setattr(session_state_proxy, "get_session_state", lambda: safe)
    interface = SimpleNamespace(session_state=session_state_proxy.SessionStateProxy(),
                                error=errors.append, info=Mock())
    monkeypatch.setattr(state, "st", interface)
    monkeypatch.setattr(runs, "st", interface)
    monkeypatch.setattr(state.auth, "current_user", lambda: "reviewer@example.com")
    data = state.new_project_data(state.MODE_PRISMA)
    data["config"]["abstract_criteria"] = SPEC["criteria"]
    data["papers"] = [state.new_paper("", f"Study {index}", "Field study abstract") for index in range(2)]
    pid = db.create_project("reviewer@example.com", "Review", data)
    state.set_active(pid, "Review")
    state.load_into_session(data, version=1, source=([], []))
    model = Mock(return_value={"verdict": "include", "reason": "A field study."})
    monkeypatch.setattr(runs.judge, "judge_abstract", model)
    monkeypatch.setattr(runs.judge, "judge_fulltext", model)
    monkeypatch.setattr(runs.fulltext_storage, "load_pdf", lambda _: PDF)
    metadata = {paper["uid"]: {
        "status": "ok", "storage_key": f"offline/{paper['uid']}.pdf",
        "sha256": runs.fulltext_storage.sha256(PDF), "filename": "study.pdf",
        "file_size": len(PDF), "page_count": 2,
    } for paper in state.papers()}
    yield SimpleNamespace(pid=pid, user="reviewer@example.com", control=control,
                          errors=errors, model=model, metadata=metadata)
    engine.dispose()


def _execute(project, *, models=None, repeat=False, papers=None):
    return runs.execute(state.papers()[:1] if papers is None else papers,
                        state.STAGE_ABSTRACT, SPEC, None,
                        MODELS[:1] if models is None else models, repeat=repeat)


@pytest.fixture
def extraction_model(monkeypatch):
    model = Mock(return_value={"answers": ANSWER})
    monkeypatch.setattr(runs.extraction_judge, "extract_pdf", model)
    return model


def _execute_extraction(project, *, papers=None, repeat=False, progress=None):
    return runs.execute(state.papers()[:1] if papers is None else papers,
                        "extraction", EXTRACTION_SPEC, project.metadata, MODELS[:1],
                        repeat=repeat, progress=progress)


def test_stale_initial_save_prevents_paid_calls_and_run_creation(project):
    db.save_project(project.user, project.pid, db.load_project(project.user, project.pid), expected_version=1)
    assert not _execute(project)
    project.model.assert_not_called()
    assert db.load_ai_runs(project.user, project.pid) == []


def test_real_runner_persists_each_model_and_only_seeds_the_initial_review(project):
    project.model.side_effect = [
        {"verdict": "include", "reason": "Field data."},
        {"verdict": "exclude", "reason": "Insufficient evidence."},
    ]
    assert _execute(project, models=MODELS)
    records = db.load_ai_runs(project.user, project.pid)
    assert len(records) == 2
    assert {row["status"] for row in records} == {"succeeded"}
    assert len({row["batch_id"] for row in records}) == 1
    assert [row["result"]["ai_verdict"] for row in records] == ["include", "exclude"]
    saved = db.load_project(project.user, project.pid)["papers"][0]["stages"]["abstract"]
    assert saved["ai_verdict"] == "include"
    assert saved["source_run_id"] == records[0]["id"]
    assert saved.get("decision") is None
    assert project.model.call_count == 2


def test_failed_completion_save_retains_paid_answer_and_recovery_never_calls_model(project, monkeypatch):
    finish = db.finish_ai_run
    monkeypatch.setattr(db, "finish_ai_run", Mock(side_effect=db.DatabaseError("Temporary database failure")))
    assert not _execute(project)
    pending = deepcopy(state._store()["pending_ai_completion"])
    assert pending["result"]["ai_verdict"] == "include"
    assert db.load_ai_runs(project.user, project.pid)[0]["status"] == "running"
    assert not _execute(project, repeat=True)
    assert project.model.call_count == 1
    monkeypatch.setattr(db, "finish_ai_run", finish)

    def rerun():
        raise RerunException(RerunData())

    monkeypatch.setattr(run_controls, "st", SimpleNamespace(
        error=project.errors.append, button=lambda *a, **k: True, rerun=rerun,
        session_state=state.st.session_state,
        stop=lambda: (_ for _ in ()).throw(StopException()),
    ))
    with pytest.raises(RerunException):
        run_controls.recover_pending()
    assert "pending_ai_completion" not in state._store()
    assert db.load_ai_runs(project.user, project.pid)[0]["result"] == pending["result"]
    assert _execute(project)
    assert project.model.call_count == 1


def test_failed_review_checkpoint_does_not_lose_or_repay_for_completed_run(project, monkeypatch):
    save = db.save_project
    checkpoints = []

    def fail_second(*args, **kwargs):
        checkpoints.append(True)
        if len(checkpoints) == 2:
            raise db.DatabaseError("Checkpoint unavailable")
        return save(*args, **kwargs)

    monkeypatch.setattr(db, "save_project", fail_second)
    assert not _execute(project)
    records = db.load_ai_runs(project.user, project.pid)
    assert records[0]["status"] == "succeeded"
    assert state.stage_state(state.papers()[0], "abstract")["source_run_id"] == records[0]["id"]
    assert state.has_unsaved_results()
    saved = db.load_project(project.user, project.pid)["papers"][0]
    assert not state.stage_state(saved, "abstract").get("ai_verdict")
    assert state.save_result()
    assert _execute(project)
    assert project.model.call_count == 1
    assert len(db.load_ai_runs(project.user, project.pid)) == 1


@pytest.mark.parametrize("interruption", [StopException(), RerunException(RerunData())])
def test_interrupted_provider_leaves_unknown_outcome_and_requires_explicit_repeat(project, interruption):
    project.model.side_effect = interruption
    with pytest.raises(type(interruption)):
        _execute(project)
    first = db.load_ai_runs(project.user, project.pid)[0]
    assert first["status"] == "running" and first["result"] is None
    project.model.side_effect = None
    assert _execute(project)
    assert project.model.call_count == 1
    assert _execute(project, repeat=True)
    assert project.model.call_count == 2
    records = db.load_ai_runs(project.user, project.pid)
    assert [row["status"] for row in records] == ["running", "succeeded"]
    assert records[0] == first


def test_interruption_after_completion_commit_can_recover_exact_write(project, monkeypatch):
    finish = db.finish_ai_run

    def finish_then_interrupt(*args, **kwargs):
        finish(*args, **kwargs)
        raise StopException()

    monkeypatch.setattr(db, "finish_ai_run", finish_then_interrupt)
    with pytest.raises(StopException):
        _execute(project)
    pending = state._store()["pending_ai_completion"]
    first = db.load_ai_runs(project.user, project.pid)[0]
    assert first["status"] == "succeeded"
    assert finish(project.user, project.pid, **pending) == first
    assert project.model.call_count == 1


def test_repeat_runs_never_replace_existing_human_screening_decision(project):
    paper = state.papers()[0]
    state.set_human_verdict(paper, "abstract", "exclude", state.criteria_hash(SPEC["criteria"]))
    before = deepcopy(state.stage_state(paper, "abstract"))
    assert _execute(project, models=MODELS)
    assert _execute(project, models=MODELS, repeat=True)
    assert state.stage_state(paper, "abstract") == before
    saved = db.load_project(project.user, project.pid)["papers"][0]["stages"]["abstract"]
    assert saved == before
    assert len(db.load_ai_runs(project.user, project.pid)) == 4
    assert project.model.call_count == 4


@pytest.mark.parametrize("error,status", [
    (RuntimeError("Provider unavailable"), "call_failed"),
    (InvalidModelResponse("Invalid JSON"), "invalid_response"),
])
def test_failures_are_separate_terminal_attempts_and_require_explicit_repeat(project, error, status):
    project.model.side_effect = error
    assert _execute(project)
    assert db.load_ai_runs(project.user, project.pid)[0]["status"] == status
    assert _execute(project)
    assert project.model.call_count == 1
    assert _execute(project, repeat=True)
    assert project.model.call_count == 2
    assert len(db.load_ai_runs(project.user, project.pid)) == 2


def test_all_selected_secrets_are_redacted_from_saved_output_and_input_snapshot(project):
    project.model.return_value = {"verdict": "include", "reason": " ".join(m["api_key"] for m in MODELS)}
    assert _execute(project, models=MODELS)
    serialized = json.dumps(db.load_ai_runs(project.user, project.pid))
    for model in MODELS:
        assert model["api_key"] not in serialized
    assert "[redacted]" in serialized
    assert all("api_key" not in record["prompt_snapshot"] for record in db.load_ai_runs(project.user, project.pid))


def test_redaction_covers_nested_dictionary_keys_without_mutating_the_input():
    key = "offline-secret"
    original = {key: [{"prefix-" + key: key}], "plain": "readable diagnostic"}
    redacted = runs._redact(original, [key])
    assert redacted == {"[redacted]": [{"prefix-[redacted]": "[redacted]"}],
                        "plain": "readable diagnostic"}
    assert key in original


def test_overlapping_selected_keys_are_redacted_longest_first_in_output_and_errors(project):
    short, long = "offline-key", "offline-key-extension"
    assert runs._redact({long: long}, [short, long, short]) == {"[redacted]": "[redacted]"}
    models = deepcopy(MODELS)
    for model, key in zip(models, (short, long)):
        model["api_key"] = key
    project.model.side_effect = RuntimeError(f"Provider rejected {long}")
    assert _execute(project, models=models)
    records = db.load_ai_runs(project.user, project.pid)
    assert all(row["result"]["ai_error"] == "Provider rejected [redacted]" for row in records)


def _malformed_secret_answer():
    return {"answers": {"setting": {
        "api_key": MODELS[0]["api_key"],
        MODELS[1]["api_key"]: [{"authorization": MODELS[0]["api_key"]}],
        "diagnostic": "Unsupported model fields",
    }}}


def test_malformed_credential_fields_remain_readable_and_persist_without_secrets(
        project, extraction_model):
    extraction_model.return_value = _malformed_secret_answer()
    assert runs.execute(state.papers()[:1], "extraction", EXTRACTION_SPEC,
                        project.metadata, MODELS)
    records = db.load_ai_runs(project.user, project.pid)
    saved = db.load_project(project.user, project.pid)
    assert len(records) == 2
    assert all(row["status"] == "invalid_response" for row in records)
    for row in records:
        diagnostic = row["result"]["invalid_answers"]["setting"]
        assert isinstance(diagnostic, str) and len(diagnostic) <= 12000
        assert json.loads(diagnostic) == {
            "api_key": "[redacted]", "[redacted]": [{"authorization": "[redacted]"}],
            "diagnostic": "Unsupported model fields",
        }
    for model in MODELS:
        assert model["api_key"] not in json.dumps(records)
        assert model["api_key"] not in json.dumps(saved)
    assert "pending_ai_completion" not in state._store()


def test_malformed_answer_recovery_backup_is_redacted_and_saves_without_rebilling(
        project, extraction_model, monkeypatch):
    extraction_model.return_value = _malformed_secret_answer()
    finish = db.finish_ai_run
    monkeypatch.setattr(db, "finish_ai_run", Mock(side_effect=db.DatabaseError("Temporary failure")))
    assert not runs.execute(state.papers()[:1], "extraction", EXTRACTION_SPEC,
                            project.metadata, MODELS)
    pending = deepcopy(state._store()["pending_ai_completion"])
    assert pending["status"] == "invalid_response"
    download = Mock()
    recovery_ui = SimpleNamespace(
        error=project.errors.append, button=lambda *a, **k: False,
        checkbox=lambda *a, **k: False, download_button=download,
        session_state=state.st.session_state,
        stop=lambda: (_ for _ in ()).throw(StopException()),
        rerun=lambda: (_ for _ in ()).throw(RerunException(RerunData())),
    )
    monkeypatch.setattr(run_controls, "st", recovery_ui)
    with pytest.raises(StopException):
        run_controls.recover_pending()
    backup = download.call_args.args[1]
    assert json.loads(backup) == pending
    for model in MODELS:
        assert model["api_key"] not in backup
        assert model["api_key"] not in json.dumps(pending)
    monkeypatch.setattr(db, "finish_ai_run", finish)
    recovery_ui.button = lambda *a, **k: True
    with pytest.raises(RerunException):
        run_controls.recover_pending()
    assert "pending_ai_completion" not in state._store()
    records = db.load_ai_runs(project.user, project.pid)
    assert records[0]["status"] == "invalid_response"
    assert records[0]["result"] == pending["result"]
    saved = db.load_project(project.user, project.pid)
    for model in MODELS:
        assert model["api_key"] not in json.dumps(records)
        assert model["api_key"] not in json.dumps(saved)
    extraction_model.assert_called_once()


def test_malformed_answer_diagnostic_is_bounded(project, extraction_model):
    extraction_model.return_value = {"answers": {"setting": {"unexpected": "x" * 20000}}}
    assert _execute_extraction(project)
    row = db.load_ai_runs(project.user, project.pid)[0]
    assert row["status"] == "invalid_response"
    assert len(row["result"]["invalid_answers"]["setting"]) == 12000


def test_malformed_answer_is_redacted_before_truncation(project, extraction_model):
    extraction_model.return_value = {
        "answers": {"setting": {"junk": "x" * 11980 + MODELS[0]["api_key"]}},
    }
    assert _execute_extraction(project)
    diagnostic = db.load_ai_runs(project.user, project.pid)[0]["result"]["invalid_answers"]["setting"]
    assert len(diagnostic) <= 12000
    assert "offline" not in diagnostic
    assert "[redacted]" in diagnostic


def test_concurrent_save_during_paid_call_keeps_attempt_without_overwriting_other_review(project):
    original_model = project.model.return_value

    def change_project(*args):
        other, version = db.load_project_versioned(project.user, project.pid)
        other["config"]["abstract_criteria"] = "Other reviewer's updated criteria"
        db.save_project(project.user, project.pid, other, expected_version=version)
        return original_model

    project.model.side_effect = change_project
    assert not _execute(project, models=MODELS)
    assert project.model.call_count == 1
    assert db.load_ai_runs(project.user, project.pid)[0]["status"] == "succeeded"
    saved = db.load_project(project.user, project.pid)
    assert saved["config"]["abstract_criteria"] == "Other reviewer's updated criteria"
    assert not saved["papers"][0]["stages"].get("abstract", {}).get("ai_verdict")


def test_project_deletion_during_paid_call_does_not_recreate_or_leak_project(project):
    def delete_project(*args):
        db.delete_project(project.user, project.pid)
        return {"verdict": "include", "reason": "Paid response"}

    project.model.side_effect = delete_project
    assert not _execute(project)
    assert project.model.call_count == 1
    assert db.load_project(project.user, project.pid) == {}
    assert db.load_ai_runs(project.user, project.pid) == []
    assert state._store()["pending_ai_completion"]["result"]["ai_reason"] == "Paid response"


def test_plan_applies_pdf_limits_per_paper_and_model(project):
    first, second = state.papers()
    metadata = deepcopy(project.metadata)
    metadata[first["uid"]]["page_count"] = 101
    metadata[second["uid"]]["file_size"] = MAX_INLINE_PDF_BYTES + 1
    spec = {"criteria": SPEC["criteria"], "prompt_version": prompts.FULLTEXT_PROMPT_VERSION}
    tasks = runs.plan(state.papers(), "fulltext", spec, metadata, MODELS, [])
    assert [(paper["uid"], model["provider"]) for paper, model in tasks] == [(first["uid"], "OpenAI")]
    metadata[first["uid"]]["status"] = "failed"
    assert runs.plan(state.papers(), "fulltext", spec, metadata, MODELS, []) == []


def test_plan_rejects_invalid_selections_and_tracks_exact_input_compatibility(project):
    paper = state.papers()[0]
    for models in ([], MODELS * 2, [MODELS[0], MODELS[0]], [{"provider": "Unknown", "model": "unknown"}]):
        with pytest.raises(ValueError):
            runs.plan([paper], "abstract", SPEC, None, models, [])
    assert _execute(project)
    records = db.load_ai_runs(project.user, project.pid)
    assert runs.plan([paper], "abstract", SPEC, None, MODELS[:1], records) == []
    changed = deepcopy(paper)
    changed["abstract"] = "Replaced abstract"
    assert len(runs.plan([changed], "abstract", SPEC, None, MODELS[:1], records)) == 1
    assert len(runs.plan([paper], "abstract", SPEC | {"criteria": "New criterion"}, None, MODELS[:1], records)) == 1
    assert len(runs.plan([paper], "abstract", SPEC | {"prompt_version": "next-version"}, None, MODELS[:1], records)) == 1


def test_extraction_run_preserves_final_human_answers_and_partial_invalid_results(project, monkeypatch):
    paper = state.papers()[0]
    source = project.metadata[paper["uid"]]["sha256"]
    extraction.save_review(paper, EXTRACTION_SPEC, source, ANSWER, page_count=2, confirm=True)
    original = deepcopy(extraction.get(paper))
    model = Mock(return_value={"answers": {"setting": {"values": []}}})
    monkeypatch.setattr(runs.extraction_judge, "extract_pdf", model)
    assert runs.execute([paper], "extraction", EXTRACTION_SPEC, project.metadata, MODELS[:1])
    assert extraction.get(paper) == original
    attempt = db.load_ai_runs(project.user, project.pid)[0]
    assert attempt["status"] == "invalid_response"
    assert attempt["result"]["field_errors"]
    assert "final_answers" not in attempt["result"]
    assert attempt["prompt_snapshot"] == EXTRACTION_SPEC
    assert model.call_count == 1


def test_known_legacy_results_import_without_becoming_new_human_decisions(project):
    # Results recorded before attempts existed are imported by the project's first save.
    state._store()[state._RUNS_MIGRATED_KEY] = False
    paper = state.papers()[0]
    state.set_ai_result(paper, "abstract", "exclude", "Older AI rationale", "OpenAI", "gpt-4.1-mini",
                        state.criteria_hash(SPEC["criteria"]), SPEC["prompt_version"])
    state.record_agree(paper, "abstract")
    original = deepcopy(state.stage_state(paper, "abstract"))
    assert _execute(project)
    records = db.load_ai_runs(project.user, project.pid)
    assert len(records) == 2
    assert sum(row["prompt_snapshot"].get("legacy", False) for row in records) == 1
    assert state.stage_state(paper, "abstract") == original
    assert state.current_final_verdict(paper, "abstract", state.criteria_hash(SPEC["criteria"])) == "exclude"
    assert project.model.call_count == 1


def test_extraction_failed_initial_save_prevents_all_calls(project, extraction_model):
    db.save_project(project.user, project.pid, db.load_project(project.user, project.pid), expected_version=1)
    assert not _execute_extraction(project)
    extraction_model.assert_not_called()
    assert db.load_ai_runs(project.user, project.pid) == []


def test_extraction_failed_checkpoint_stops_batch_and_resume_preserves_paid_answer(
        project, extraction_model, monkeypatch):
    papers = state.papers()
    save = db.save_project
    calls = []

    def fail_checkpoint(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise db.DatabaseError("Checkpoint unavailable")
        return save(*args, **kwargs)

    monkeypatch.setattr(db, "save_project", fail_checkpoint)
    assert not _execute_extraction(project, papers=papers)
    assert extraction_model.call_count == 1
    assert extraction.get(papers[0])["ai_answers"] == ANSWER
    assert db.load_ai_runs(project.user, project.pid)[0]["result"]["ai_answers"] == ANSWER
    remaining = [paper for paper in papers if extraction.pending(
        paper, EXTRACTION_SPEC, project.metadata[paper["uid"]]["sha256"])]
    assert [paper["uid"] for paper in remaining] == [papers[1]["uid"]]
    assert state.save_result()
    progress = Mock()
    assert _execute_extraction(project, papers=papers, progress=progress)
    assert extraction_model.call_count == len(papers)
    assert progress.call_args.args == (1, 1)
    assert len(db.load_ai_runs(project.user, project.pid)) == len(papers)


def test_extraction_marks_the_paid_answer_unsaved_before_its_checkpoint_starts(project, extraction_model, monkeypatch):
    events = []
    save = db.save_project

    def load(key):
        assert db.load_ai_runs(project.user, project.pid)[0]["status"] == "running"
        events.append("load")
        return PDF

    def extract(*args):
        assert events[-1] == "load"
        events.append("model")
        return {"answers": ANSWER}

    def save_project(*args, **kwargs):
        # The answer is attached to the paper and marked before the first
        # interruptible step of its checkpoint.
        if "model" in events:
            assert state.has_unsaved_results()
            assert extraction.get(state.papers()[0])["ai_answers"]
        events.append("save")
        return save(*args, **kwargs)

    monkeypatch.setattr(runs.fulltext_storage, "load_pdf", load)
    monkeypatch.setattr(db, "save_project", save_project)
    extraction_model.side_effect = extract
    assert _execute_extraction(project)
    assert events == ["save", "load", "model", "save"]
    assert not state.has_unsaved_results()


@pytest.mark.parametrize("error,status,pending", [
    (RuntimeError("Timed out"), "call_failed", True),
    (InvalidModelResponse("Invalid JSON"), "invalid_response", False),
])
def test_extraction_errors_persist_and_only_explicit_repeat_retries(
        project, extraction_model, error, status, pending):
    paper = state.papers()[0]
    source = project.metadata[paper["uid"]]["sha256"]
    extraction_model.side_effect = error
    assert _execute_extraction(project)
    assert extraction.get(paper)["ai_error_kind"] == status
    assert extraction.pending(paper, EXTRACTION_SPEC, source) is pending
    assert db.load_ai_runs(project.user, project.pid)[0]["status"] == status
    assert _execute_extraction(project)
    assert extraction_model.call_count == 1
    assert _execute_extraction(project, repeat=True)
    assert extraction_model.call_count == 2


def test_extraction_pdf_hash_mismatch_never_calls_provider(project, extraction_model):
    paper = state.papers()[0]
    project.metadata[paper["uid"]]["sha256"] = "different"
    assert _execute_extraction(project)
    extraction_model.assert_not_called()
    assert "Reattach" in extraction.get(paper)["ai_error"]
    assert db.load_ai_runs(project.user, project.pid)[0]["status"] == "call_failed"


def test_extraction_partial_invalid_fields_are_never_automatically_retried(project, extraction_model):
    paper = state.papers()[0]
    extraction_model.return_value = {"answers": {"setting": {"values": []}}}
    assert _execute_extraction(project)
    source = project.metadata[paper["uid"]]["sha256"]
    assert extraction.status(paper, EXTRACTION_SPEC, source) == "Invalid"
    assert not extraction.pending(paper, EXTRACTION_SPEC, source)
    assert _execute_extraction(project)
    assert extraction_model.call_count == 1
    assert db.load_ai_runs(project.user, project.pid)[0]["status"] == "invalid_response"
