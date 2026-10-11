"""Routine saves and attempts do not read the project document back."""
import pytest
from sqlalchemy import create_engine, event

from core import db
from features.workflow import state

USER = "reviewer@example.com"


@pytest.fixture
def engine(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'bookkeeping.sqlite'}")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    yield engine
    engine.dispose()


@pytest.fixture
def statements(engine):
    seen = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        seen.append(" ".join(statement.split()))

    event.listen(engine, "before_cursor_execute", capture)
    yield seen
    event.remove(engine, "before_cursor_execute", capture)


def _reads_document(statements):
    return [s for s in statements if s.upper().startswith("SELECT") and "projects.data" in s]


def _start(pid, uid="paper-1", **options):
    return db.start_ai_run(
        USER, pid, paper_uid=uid, stage="abstract", batch_id="batch", provider="Provider",
        model="model", config_hash="criteria", source_hash="source", prompt_version="v1",
        prompt_snapshot={"criteria": "Include field studies."}, **options)


def _finish(pid, run_id, **options):
    return db.finish_ai_run(USER, pid, run_id, status="succeeded",
                            result={"ai_verdict": "include", "ai_reason": "Matches."},
                            duration_seconds=1.0, **options)


def test_versioned_attempt_and_migrated_save_never_select_the_document(engine, statements):
    pid = db.create_project(USER, "Review", {"papers": [{"uid": "paper-1", "stages": {}}]})
    statements.clear()
    run = _start(pid, expected_version=1)
    assert _finish(pid, run["id"], verify_paper=False)["status"] == "succeeded"
    version = db.save_project(USER, pid, {"papers": [{"uid": "paper-1", "stages": {}}]},
                              expected_version=1, import_legacy_runs=False)
    assert version == 2
    assert _reads_document(statements) == []
    assert not [s for s in statements if s.upper().startswith("SELECT") and "FROM ai_runs" in s
                and "ai_runs.id =" not in s]


def test_unversioned_attempt_still_verifies_the_paper(engine):
    pid = db.create_project(USER, "Review", {"papers": [{"uid": "paper-1", "stages": {}}]})
    with pytest.raises(db.DatabaseError, match="Paper no longer exists"):
        _start(pid, uid="missing")
    with pytest.raises(db.ProjectConflictError):
        _start(pid, expected_version=7)


def test_named_removed_papers_lose_their_attempts_without_a_document_read(engine, statements):
    pid = db.create_project(USER, "Review", {"papers": [{"uid": "paper-1"}, {"uid": "paper-2"}]})
    gone, kept = _start(pid), _start(pid, uid="paper-2")
    statements.clear()
    db.save_project(USER, pid, {"papers": [{"uid": "paper-2"}]}, expected_version=1,
                    import_legacy_runs=False, removed_paper_uids=["paper-1"])
    assert _reads_document(statements) == []
    assert [run["id"] for run in db.load_ai_runs(USER, pid)] == [kept["id"]]
    with pytest.raises(db.DatabaseError, match="not accessible"):
        _finish(pid, gone["id"], verify_paper=False)


def test_run_index_omits_answers_and_prompt_snapshots(engine):
    pid = db.create_project(USER, "Review", {"papers": [{"uid": "paper-1"}]})
    run = _finish(pid, _start(pid)["id"])
    assert db.load_ai_run_index(USER, pid, "abstract") == [
        {name: run[name] for name in db._RUN_INDEX_COLUMNS}]
    assert db.load_ai_run_index(USER, pid, "fulltext") == []
    assert db.load_ai_run_index("other@example.com", pid) == []


def test_project_version_lookup_is_scoped_to_the_owner(engine):
    pid = db.create_project(USER, "Review", {"papers": []})
    assert db.project_version(USER, pid) == 1
    db.save_project(USER, pid, {"papers": []}, expected_version=1)
    assert db.project_version(USER, pid) == 2
    assert db.project_version("other@example.com", pid) is None
    assert db.project_version(USER, "missing") is None


@pytest.fixture
def session(engine, monkeypatch):
    from types import SimpleNamespace
    fake = SimpleNamespace(session_state={}, error=lambda *args, **kwargs: None)
    monkeypatch.setattr(state, "st", fake)
    monkeypatch.setattr(state.auth, "current_user", lambda: USER)
    return fake


def _legacy_project():
    paper = state.new_paper("", "Study", "Abstract")
    paper["uid"] = "paper-1"
    paper["stages"]["abstract"] = {"ai_verdict": "include", "ai_reason": "Earlier answer.",
                                   "provider": "OpenAI", "model": "gpt-4.1", "criteria_hash": "h"}
    data = state.new_project_data(state.MODE_PRISMA)
    data.pop(state._RUNS_MIGRATED_KEY)
    data["papers"] = [paper]
    return data


def test_first_save_imports_earlier_answers_once_and_later_saves_skip_the_scan(session, statements):
    pid = db.create_project(USER, "Legacy", _legacy_project())
    state.set_active(pid, "Legacy")
    state.reload_project(pid)
    assert state._RUNS_MIGRATED_KEY not in state.snapshot()

    statements.clear()
    assert state.save_active()
    assert len(db.load_ai_runs(USER, pid)) == 1
    assert db.load_project(USER, pid)[state._RUNS_MIGRATED_KEY] is True
    assert state.snapshot()[state._RUNS_MIGRATED_KEY] is True

    statements.clear()
    assert state.save_active()
    assert _reads_document(statements) == []
    assert not [s for s in statements if "FROM ai_runs" in s]
    assert len(db.load_ai_runs(USER, pid)) == 1

    # A reopened project keeps the marker, so the scan never comes back.
    state.reload_project(pid)
    statements.clear()
    assert state.save_active()
    assert _reads_document(statements) == []


def test_removing_a_paper_through_the_session_deletes_its_attempts(session):
    from features.workflow import documents
    data = state.new_project_data(state.MODE_DIRECT)
    papers = [state.new_paper("", f"Study {i}", "") for i in range(2)]
    data["papers"] = papers
    pid = db.create_project(USER, "Direct", data)
    state.set_active(pid, "Direct")
    state.reload_project(pid)
    assert state.save_active()
    removed, kept = (_start(pid, uid=paper["uid"]) for paper in papers)

    documents.remove(state.prepare_save(), papers[0]["uid"])

    assert [run["id"] for run in db.load_ai_runs(USER, pid)] == [kept["id"]]
    assert removed["id"] not in {run["id"] for run in db.load_ai_run_index(USER, pid)}


def test_screening_history_keeps_every_archived_decision():
    paper = state.new_paper("", "Study", "Abstract")
    for round_number in range(14):
        state.set_ai_result(paper, state.STAGE_ABSTRACT, "include", f"Round {round_number}",
                            "OpenAI", "gpt-4.1", f"hash-{round_number}", "v1")
        state.record_agree(paper, state.STAGE_ABSTRACT)
        state.archive_ai_result(paper, state.STAGE_ABSTRACT)
    history = state.stage_state(paper, state.STAGE_ABSTRACT)["history"]
    assert [entry["ai_reason"] for entry in history] == [f"Round {n}" for n in range(14)]
    assert all(entry["decision"] == state.DECISION_AGREE for entry in history)


def test_first_generation_project_imports_its_ai_history_on_the_first_save(session, statements):
    legacy = {"topic": "Earlier topic", "threshold": 80, "rubric": [],
              "papers": [{"title": "Study", "abstract": "Abstract", "doi": "", "ai_score": 90,
                          "ai_reason": "Relevant.", "decision": "agree"}]}
    assert state._RUNS_MIGRATED_KEY not in state.migrate(dict(legacy))
    pid = db.create_project(USER, "First generation", legacy)
    state.set_active(pid, "First generation")
    state.reload_project(pid)
    assert not state._store()[state._RUNS_MIGRATED_KEY]

    assert state.save_active()

    imported = db.load_ai_runs(USER, pid)
    assert len(imported) == 1 and imported[0]["result"]["ai_verdict"] == "include"
    assert db.load_project(USER, pid)[state._RUNS_MIGRATED_KEY] is True
