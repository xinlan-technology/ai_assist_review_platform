"""Durable AI attempts remain scoped, immutable, and independent of review state."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib

import pytest
from sqlalchemy import create_engine, event, select, update
from sqlalchemy.exc import SQLAlchemyError

from core import db


USER = "reviewer@example.com"
OTHER = "other@example.com"


@pytest.fixture
def engine(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.sqlite'}")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    yield engine
    engine.dispose()


def _project(*uids, user=USER, **extra):
    return db.create_project(user, "Review", {
        "papers": [{"uid": uid, "stages": {}} for uid in uids], **extra,
    })


def _start(pid, user=USER, **overrides):
    arguments = {
        "paper_uid": "paper-1", "stage": "abstract", "batch_id": "batch-1",
        "provider": "Provider", "model": "model", "config_hash": "criteria-hash",
        "source_hash": "abstract-hash", "prompt_version": "abstract-v1",
        "prompt_snapshot": {"criteria": "Include field studies."},
    }
    arguments.update(overrides)
    return db.start_ai_run(user, pid, **arguments)


def _finish(pid, run_id, user=USER, **overrides):
    arguments = {
        "status": "succeeded", "result": {"ai_verdict": "include", "ai_reason": "Matches."},
        "duration_seconds": 1.25,
    }
    arguments.update(overrides)
    return db.finish_ai_run(user, pid, run_id, **arguments)


def test_start_and_finish_preserve_project_and_version(engine):
    pid = _project("paper-1")
    original = db.load_project_versioned(USER, pid)
    run = _start(pid, expected_version=1)
    assert run["status"] == "running"
    assert run["completed_at"] is None and run["result"] is None
    assert run["project_id"] == pid and run["user_email"] == USER
    completed = _finish(pid, run["id"])
    assert completed["status"] == "succeeded"
    assert completed["completed_at"] >= completed["started_at"]
    assert completed["duration_seconds"] == 1.25
    assert db.load_ai_runs(USER, pid) == [completed]
    assert db.load_project_versioned(USER, pid) == original


def test_start_rejects_foreign_missing_project_and_missing_paper(engine):
    pid = _project("paper-1")
    for user, target in ((OTHER, pid), (USER, "missing")):
        with pytest.raises(db.DatabaseError, match="not accessible"):
            _start(target, user)
    with pytest.raises(db.DatabaseError, match="Paper no longer exists"):
        _start(pid, paper_uid="missing")
    assert db.load_ai_runs(USER, pid) == []


def test_start_checks_version_without_modifying_project(engine):
    pid = _project("paper-1")
    data = db.load_project(USER, pid)
    db.save_project(USER, pid, data, expected_version=1)
    with pytest.raises(db.ProjectConflictError, match="another tab"):
        _start(pid, expected_version=1)
    assert db.load_ai_runs(USER, pid) == []
    assert _start(pid, expected_version=2)["status"] == "running"
    assert db.load_project_versioned(USER, pid)[1] == 2


def test_stable_start_id_is_exactly_idempotent(engine):
    pid = _project("paper-1")
    first = _start(pid, run_id="stable-id")
    assert _start(pid, run_id="stable-id") == first
    with pytest.raises(db.DatabaseError, match="different attempt"):
        _start(pid, run_id="stable-id", model="another-model")
    done = _finish(pid, first["id"])
    assert _start(pid, run_id="stable-id") == done
    assert len(db.load_ai_runs(USER, pid)) == 1


def test_concurrent_start_and_finish_retries_keep_one_immutable_record(engine):
    pid = _project("paper-1")
    with ThreadPoolExecutor(max_workers=4) as pool:
        starts = list(pool.map(lambda _: _start(pid, run_id="concurrent-id"), range(8)))
        finishes = list(pool.map(lambda _: _finish(pid, "concurrent-id"), range(8)))
    assert all(record == starts[0] for record in starts)
    assert all(record == finishes[0] for record in finishes)
    assert db.load_ai_runs(USER, pid) == [finishes[0]]


def test_stable_run_id_cannot_read_or_replace_other_project(engine):
    pid = _project("paper-1")
    other_pid = _project("paper-1", user=OTHER)
    first = _start(pid, run_id="shared-id")
    with pytest.raises(db.DatabaseError, match="different attempt"):
        _start(other_pid, OTHER, run_id="shared-id")
    assert db.load_ai_runs(OTHER, pid) == []
    assert db.load_ai_runs(USER, other_pid) == []
    assert db.load_ai_runs(OTHER, other_pid) == []
    assert db.load_ai_runs(USER, pid) == [first]


@pytest.mark.parametrize("status", ["succeeded", "invalid_response", "call_failed"])
def test_finish_terminal_status_and_exact_retry(engine, status):
    pid = _project("paper-1")
    run = _start(pid)
    first = _finish(pid, run["id"], status=status)
    assert _finish(pid, run["id"], status=status) == first
    for change in ({"result": {"ai_verdict": "exclude"}}, {"duration_seconds": 2.0},
                   {"status": "call_failed" if status != "call_failed" else "succeeded"}):
        with pytest.raises(db.DatabaseError, match="cannot be changed"):
            _finish(pid, run["id"], **({"status": status} | change))
    assert db.load_ai_runs(USER, pid) == [first]


def test_finish_requires_owned_project_run_and_existing_paper(engine):
    pid = _project("paper-1")
    another_pid = _project("paper-1")
    run = _start(pid)
    for user, target, run_id in ((OTHER, pid, run["id"]), (USER, "missing", run["id"]),
                                 (USER, another_pid, run["id"]), (USER, pid, "missing")):
        with pytest.raises(db.DatabaseError, match="not accessible"):
            _finish(target, run_id, user)
    with engine.begin() as conn:
        conn.execute(update(db.projects).where(db.projects.c.id == pid).values(data={"papers": []}))
    with pytest.raises(db.DatabaseError, match="Paper no longer exists"):
        _finish(pid, run["id"])
    assert db.load_ai_runs(USER, pid)[0]["status"] == "running"


def test_runs_are_not_capped_and_filters_preserve_stable_order(engine, monkeypatch):
    pid = _project("paper-1", "paper-2")
    monkeypatch.setattr(db, "_now", lambda: "2026-01-01T00:00:00+00:00")
    for index in reversed(range(24)):
        _start(pid, run_id=f"run-{index:02}", batch_id=f"batch-{index // 3}")
    _start(pid, paper_uid="paper-2", stage="extraction")
    _start(pid, stage="fulltext")
    runs = db.load_ai_runs(USER, pid, stage="abstract", paper_uid="paper-1")
    assert [run["id"] for run in runs] == [f"run-{index:02}" for index in range(24)]
    assert len(db.load_ai_runs(USER, pid)) == 26
    assert len(db.load_ai_runs(USER, pid, paper_uid="paper-2")) == 1


def test_review_saves_retain_runs_and_removed_papers_delete_only_their_runs(engine):
    pid = _project("paper-1", "paper-2")
    first = _start(pid)
    second = _start(pid, paper_uid="paper-2")
    data, version = db.load_project_versioned(USER, pid)
    data["papers"][0]["stages"] = {"abstract": {"decision": "human", "human_verdict": "exclude"}}
    version = db.save_project(USER, pid, data, expected_version=version)
    assert db.load_ai_runs(USER, pid) == [first, second]
    data["papers"] = data["papers"][1:]
    db.save_project(USER, pid, data, expected_version=version)
    assert db.load_ai_runs(USER, pid) == [second]
    with pytest.raises(db.DatabaseError, match="not accessible"):
        _finish(pid, first["id"])


def test_run_cleanup_only_binds_removed_uids_in_bounded_chunks(engine):
    pid = _project(*(f"paper-{index}" for index in range(1205)))
    _start(pid)
    retained = _start(pid, paper_uid="paper-1204")
    data = db.load_project(USER, pid)
    deletions = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("DELETE FROM AI_RUNS"):
            deletions.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        db.save_project(USER, pid, data, expected_version=1)
        assert deletions == []
        data["papers"] = data["papers"][-5:]
        db.save_project(USER, pid, data, expected_version=2)
        assert len(deletions) == 3
        assert all("NOT IN" not in query and len(parameters) <= 502
                   for query, parameters in deletions)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert db.load_ai_runs(USER, pid) == [retained]


def test_csv_replacement_and_failed_save_cleanup_are_atomic(engine):
    pid = _project("paper-1")
    run = _start(pid)
    with pytest.raises(db.ProjectConflictError):
        db.save_project(USER, pid, {"papers": []}, expected_version=0, source=([], []))
    assert db.load_ai_runs(USER, pid) == [run]
    db.save_project(USER, pid, {"papers": [{"uid": "replacement"}]},
                    expected_version=1, source=(["Title"], [{"Title": "Replacement"}]))
    assert db.load_ai_runs(USER, pid) == []
    with pytest.raises(db.DatabaseError, match="not accessible"):
        _finish(pid, run["id"])


def test_project_deletion_cascades_without_cross_user_effects(engine):
    pid = _project("paper-1")
    other_pid = _project("paper-1", user=OTHER)
    first = _start(pid)
    second = _start(other_pid, OTHER)
    db.delete_project(OTHER, pid)
    assert db.load_ai_runs(USER, pid) == [first]
    db.delete_project(USER, pid)
    with engine.connect() as conn:
        stored_ids = list(conn.execute(select(db.ai_runs.c.id)).scalars())
    assert stored_ids == [second["id"]]
    assert db.load_ai_runs(OTHER, other_pid) == [second]
    with pytest.raises(db.DatabaseError, match="not accessible"):
        _finish(pid, first["id"])


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_duration_cannot_complete_run(engine, duration):
    pid = _project("paper-1")
    run = _start(pid)
    with pytest.raises(db.DatabaseError, match="duration"):
        _finish(pid, run["id"], duration_seconds=duration)
    assert db.load_ai_runs(USER, pid)[0]["status"] == "running"


def test_snapshots_are_json_copies_and_credentials_are_rejected(engine):
    pid = _project("paper-1")
    prompt = {"spec": {"instructions": "Extract settings.", "questions": []}}
    run = _start(pid, prompt_snapshot=prompt)
    prompt["spec"]["instructions"] = "Changed."
    assert db.load_ai_runs(USER, pid)[0]["prompt_snapshot"] == run["prompt_snapshot"]
    with pytest.raises(db.DatabaseError, match="credentials"):
        _start(pid, prompt_snapshot={"config": {"api_key": "must-not-be-stored"}})
    with pytest.raises(TypeError):
        _start(pid, api_key="must-not-be-stored")
    with pytest.raises(db.DatabaseError, match="JSON serializable"):
        _start(pid, prompt_snapshot={"object": object()})


def test_question_ids_are_not_mistaken_for_credential_metadata(engine):
    pid = _project("paper-1")
    run = _start(pid, stage="extraction")
    result = {"ai_answers": {"api_key": {"values": ["Not reported"]}},
              "field_errors": {"authorization": "Missing answer"}}
    assert _finish(pid, run["id"], result=result)["result"] == result


def test_legacy_snapshots_survive_history_trimming_and_ignore_human_revisions(engine):
    criteria = "Include field studies."
    chash = hashlib.sha256(criteria.encode()).hexdigest()[:16]
    history = [
        {"ai_verdict": "include", "ai_reason": f"Reason {index}", "criteria_hash": chash,
         "completed_at": f"2025-01-{index + 1:02}T00:00:00+00:00"}
        for index in range(12)
    ]
    paper = {"uid": "paper-1", "stages": {"abstract": {
        **deepcopy(history[-1]), "history": deepcopy(history), "decision": "agree",
    }}}
    data = {"config": {"abstract_criteria": criteria}, "papers": [paper]}
    pid = db.create_project(USER, "Legacy", data)
    assert db.load_ai_runs(USER, pid) == []
    incoming = deepcopy(data)
    incoming["papers"][0]["stages"]["abstract"]["history"] = []
    db.save_project(USER, pid, incoming, expected_version=1)
    imported = db.load_ai_runs(USER, pid)
    assert len(imported) == 12
    assert all(run["prompt_snapshot"] == {"legacy": True, "criteria": criteria} for run in imported)
    assert all(run["source_hash"] == "" for run in imported)
    current = incoming["papers"][0]["stages"]["abstract"]
    current.update(decision="disagree", human_verdict="exclude", archived_at="human-change")
    current["history"] = [{**current, "history": []}]
    db.save_project(USER, pid, incoming, expected_version=2)
    assert db.load_ai_runs(USER, pid) == imported


def test_legacy_import_checks_all_snapshot_ids_with_one_scoped_query_per_save(engine):
    history = [{"ai_verdict": "include", "ai_reason": f"Reason {index}"} for index in range(12)]
    data = {"papers": [{"uid": "paper-1", "stages": {"abstract": {
        **history[-1], "history": history,
    }}}]}
    pid = db.create_project(USER, "Legacy", data)
    queries = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT") and "FROM ai_runs" in statement:
            queries.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        for version in (1, 2):
            queries.clear()
            db.save_project(USER, pid, data, expected_version=version)
            assert len(queries) == 1
            query, parameters = queries[0]
            assert "ai_runs.project_id =" in query and "ai_runs.user_email =" in query
            assert parameters == (pid, USER)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(db.load_ai_runs(USER, pid)) == 12


def test_legacy_import_keeps_known_provenance_and_excludes_human_only_records(engine):
    spec = {"instructions": "Find settings", "questions": [], "prompt_version": "extraction-v1"}
    data = {"config": {"fulltext_criteria": "Changed criteria"}, "papers": [{
        "uid": "paper-1", "stages": {
            "abstract": {"decision": "human", "human_verdict": "include"},
            "fulltext": {"ai_error": "Bad format", "ai_error_kind": "invalid_response",
                         "criteria_hash": "old-hash"},
            "extraction": {"spec": spec, "spec_hash": "spec-hash", "source_sha256": "pdf-hash",
                           "ai_answers": {}, "field_errors": {"setting": "Missing answer"},
                           "final_answers": {"setting": "Human answer"}},
        },
    }]}
    pid = db.create_project(USER, "Legacy", data)
    db.save_project(USER, pid, data, expected_version=1)
    records = {run["stage"]: run for run in db.load_ai_runs(USER, pid)}
    assert set(records) == {"fulltext", "extraction"}
    assert records["fulltext"]["prompt_snapshot"] == {"legacy": True}
    assert records["fulltext"]["source_hash"] == ""
    assert records["extraction"]["prompt_snapshot"] == {"legacy": True, "spec": spec}
    assert records["extraction"]["source_hash"] == "pdf-hash"
    assert all(run["status"] == "invalid_response" for run in records.values())
    assert "final_answers" not in records["extraction"]["result"]


def test_migrating_a_snapshot_linked_to_a_run_does_not_duplicate_it(engine):
    pid = _project("paper-1")
    run = _start(pid)
    completed = _finish(pid, run["id"])
    data = db.load_project(USER, pid)
    data["papers"][0]["stages"]["abstract"] = completed["result"] | {"source_run_id": run["id"]}
    db.save_project(USER, pid, data, expected_version=1)
    assert db.load_ai_runs(USER, pid) == [completed]


def test_legacy_import_and_removed_paper_cleanup_share_transaction(engine):
    data = {"papers": [{"uid": "paper-1", "stages": {"abstract": {"ai_verdict": "include"}}}]}
    pid = db.create_project(USER, "Legacy", data)
    with pytest.raises(db.ProjectConflictError):
        db.save_project(USER, pid, {"papers": []}, expected_version=0)
    assert db.load_ai_runs(USER, pid) == []
    db.save_project(USER, pid, {"papers": []}, expected_version=1)
    assert db.load_ai_runs(USER, pid) == []


def test_save_failure_rolls_back_legacy_import_and_existing_run_cleanup(engine, monkeypatch):
    data = {"papers": [{"uid": "paper-1", "stages": {"abstract": {"ai_verdict": "include"}}}]}
    pid = db.create_project(USER, "Legacy", data)
    run = _start(pid)

    def fail(*args, **kwargs):
        raise SQLAlchemyError("Simulated metadata failure")

    monkeypatch.setattr(db, "_write_fulltext", fail)
    with pytest.raises(db.DatabaseError):
        db.save_project(USER, pid, {"papers": []}, expected_version=1, fulltext={})
    assert db.load_ai_runs(USER, pid) == [run]
    assert db.load_project_versioned(USER, pid) == (data, 1)
