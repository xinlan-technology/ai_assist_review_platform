"""Offline initialization, ownership, and locked PDF-deletion regressions."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event, Lock
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError

from core import db


INITIALIZE = db._get_engine
OWNER = "owner@example.invalid"
OTHER = "other@example.invalid"


@pytest.fixture
def initializer(monkeypatch):
    monkeypatch.setattr(db, "_get_engine", INITIALIZE)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_engine_lock", Lock())
    monkeypatch.setattr(db, "_conn_string", lambda: "postgresql://offline.invalid/test")
    engine = MagicMock()
    monkeypatch.setattr(db, "create_engine", Mock(return_value=engine))
    for name in ("_upgrade_projects_schema", "_upgrade_fulltexts_schema", "_protect_app_tables"):
        monkeypatch.setattr(db, name, Mock())
    monkeypatch.setattr(db._metadata, "create_all", Mock())
    return engine


def test_initialization_uses_one_transaction_and_publishes_after_commit(initializer):
    connection = initializer.begin.return_value.__enter__.return_value
    events = []

    def checkpoint(label):
        def record(*args):
            assert db._engine is None
            events.append(label)
        return record

    steps = [db._metadata.create_all, db._upgrade_projects_schema,
             db._upgrade_fulltexts_schema, db._protect_app_tables]
    for index, step in enumerate(steps):
        step.side_effect = checkpoint(index)
    initializer.begin.return_value.__exit__.side_effect = checkpoint("commit")

    assert db._get_engine() is initializer
    assert db._engine is initializer
    assert events == [0, 1, 2, 3, "commit"]
    initializer.begin.assert_called_once_with()
    for step in steps:
        step.assert_called_once_with(connection)
    initializer.dispose.assert_not_called()


@pytest.mark.parametrize("error", [SQLAlchemyError("offline"), RuntimeError("interrupted"), KeyboardInterrupt()])
def test_failed_initialization_never_publishes_and_disposes(initializer, error):
    db._protect_app_tables.side_effect = error
    expected = db.DatabaseError if isinstance(error, SQLAlchemyError) else type(error)

    with pytest.raises(expected):
        db._get_engine()

    assert db._engine is None
    initializer.dispose.assert_called_once_with()
    assert initializer.begin.return_value.__exit__.call_args.args[0] is type(error)


def test_commit_failure_never_publishes_the_engine(initializer):
    initializer.begin.return_value.__exit__.side_effect = SQLAlchemyError("Commit failed")
    with pytest.raises(db.DatabaseError):
        db._get_engine()
    assert db._engine is None
    initializer.dispose.assert_called_once_with()


def test_real_sqlite_initialization_remains_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "_get_engine", INITIALIZE)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_engine_lock", Lock())
    monkeypatch.setattr(db, "_conn_string", lambda: f"sqlite:///{tmp_path / 'initialization.sqlite'}")
    engine = db._get_engine()
    try:
        assert db._get_engine() is engine
        assert set(inspect(engine).get_table_names()) == {
            "projects", "fulltexts", "project_sources", "ai_runs",
        }
    finally:
        engine.dispose()


def test_concurrent_initialization_cannot_return_an_unprotected_engine(initializer):
    entered, release, second_started = Event(), Event(), Event()

    def create_tables(connection):
        entered.set()
        assert release.wait(5)

    def second_session():
        second_started.set()
        return db._get_engine()

    db._metadata.create_all.side_effect = create_tables
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(db._get_engine)
        try:
            assert entered.wait(5)
            second = pool.submit(second_session)
            assert second_started.wait(5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
            assert db._engine is None
            db._protect_app_tables.assert_not_called()
        finally:
            release.set()
        assert first.result(timeout=5) is initializer
        assert second.result(timeout=5) is initializer
    db.create_engine.assert_called_once()
    db._protect_app_tables.assert_called_once()


@pytest.fixture
def engine(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'security.sqlite'}")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    yield engine
    engine.dispose()


def test_legacy_mutators_require_owned_parent_and_cannot_poison_unique_keys(engine):
    pid = db.create_project(OWNER, "Review", {"papers": [{"uid": "paper"}]})
    for user, target in ((OTHER, pid), (OWNER, "missing")):
        with pytest.raises(db.DatabaseError, match="not accessible"):
            db.save_project_source(user, target, ["title"], [{"title": "Foreign"}])
        with pytest.raises(db.DatabaseError, match="not accessible"):
            db.upsert_fulltext(user, target, "paper", "foreign.pdf", "digest", "ok")
    db.save_project_source(OWNER, pid, ["title"], [{"title": "Owned"}])
    db.upsert_fulltext(OWNER, pid, "paper", "owned.pdf", "digest", "ok")
    assert db.load_project_source(OWNER, pid) == (["title"], [{"title": "Owned"}])
    assert db.load_fulltexts(OWNER, pid)["paper"]["filename"] == "owned.pdf"
    assert db.load_project_source(OTHER, pid) == ([], [])
    assert db.load_fulltexts(OTHER, pid) == {}


def test_delete_returns_owned_deduplicated_live_and_pending_pdf_keys(engine):
    pid = db.create_project(OWNER, "Review", {
        "papers": [{"uid": "paper"}],
        "pending_pdf_deletions": [{"storage_key": "old.pdf"}, {"storage_key": "current.pdf"}],
    })
    db.upsert_fulltext(OWNER, pid, "paper", "current.pdf", "digest", "ok", storage_key="current.pdf")
    other = db.create_project(OTHER, "Other", {"papers": [{"uid": "paper"}]})
    db.upsert_fulltext(OTHER, other, "paper", "foreign.pdf", "digest", "ok", storage_key="foreign.pdf")
    assert db.delete_project(OTHER, pid) == []
    assert db.delete_project(OWNER, pid) == ["current.pdf", "old.pdf"]
    assert db.delete_project(OWNER, pid) == []
    assert db.load_fulltexts(OWNER, pid) == {}
    assert db.load_fulltexts(OTHER, other)["paper"]["storage_key"] == "foreign.pdf"


def test_delete_waits_for_concurrent_save_and_collects_its_new_pdf(engine, monkeypatch):
    data = {"papers": [{"uid": "paper"}]}
    pid = db.create_project(OWNER, "Review", data)
    db.upsert_fulltext(OWNER, pid, "paper", "old.pdf", "old", "ok", storage_key="old.pdf")
    locked, release, deleting = Event(), Event(), Event()
    write_fulltext = db._write_fulltext

    def paused_write(*args, **kwargs):
        locked.set()
        assert release.wait(5)
        write_fulltext(*args, **kwargs)

    def delete():
        deleting.set()
        return db.delete_project(OWNER, pid)

    monkeypatch.setattr(db, "_write_fulltext", paused_write)
    replacement = dict(data, pending_pdf_deletions=[{"storage_key": "old.pdf"}])
    with ThreadPoolExecutor(max_workers=2) as pool:
        saving = pool.submit(db.save_project, OWNER, pid, replacement, expected_version=1, fulltext={
            "paper_uid": "paper", "filename": "new.pdf", "sha256": "new", "status": "ok",
            "storage_key": "new.pdf",
        })
        try:
            assert locked.wait(5)
            removing = pool.submit(delete)
            assert deleting.wait(5)
            with pytest.raises(TimeoutError):
                removing.result(timeout=0.05)
        finally:
            release.set()
        assert saving.result(timeout=5) == 2
        assert removing.result(timeout=5) == ["new.pdf", "old.pdf"]
    assert db.load_project(OWNER, pid) == {}
    assert db.load_fulltexts(OWNER, pid) == {}
