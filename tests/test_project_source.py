"""The imported spreadsheet is stored once, beside the project, not inside it."""
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine

from core import db
from features.workflow import state


@pytest.fixture
def engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    return engine


@pytest.fixture
def session(monkeypatch):
    fake = SimpleNamespace(session_state={}, error=lambda *args, **kwargs: None)
    monkeypatch.setattr(state, "st", fake)
    fake.session_state["project_store"] = {
        "mode": state.MODE_PRISMA,
        "config": {"abstract_criteria": "c"},
        "papers": [state.new_paper("10.1/a", "A study", "Abstract")],
        "original_df": pd.DataFrame([{"Title": "A study", "Abstract": "Abstract"}]),
        "cursors": {},
    }
    return fake


def test_round_trip_and_project_delete_cleans_up(engine):
    pid = db.create_project("r@example.com", "Study", {})
    db.save_project_source("r@example.com", pid, ["Title"], [{"Title": "A study"}])
    assert db.load_project_source("r@example.com", pid) == (["Title"], [{"Title": "A study"}])
    assert db.load_project_source("other@example.com", pid) == ([], [])

    db.save_project_source("r@example.com", pid, ["Title"], [{"Title": "Replaced"}])
    assert db.load_project_source("r@example.com", pid)[1] == [{"Title": "Replaced"}]

    db.delete_project("r@example.com", pid)
    assert db.load_project_source("r@example.com", pid) == ([], [])


def test_snapshot_no_longer_carries_the_spreadsheet(session):
    snapshot = state.snapshot()
    assert "original_records" not in snapshot and "original_columns" not in snapshot
    assert snapshot["papers"] and snapshot["mode"] == state.MODE_PRISMA


def test_pending_source_is_offered_once_and_written_with_the_document(session, monkeypatch):
    """Atomic imports prevent pairing papers with another import's rows."""
    state._store()[state._SOURCE_KEY] = True
    assert state._pending_source() == (["Title", "Abstract"],
                                       [{"Title": "A study", "Abstract": "Abstract"}])

    calls = []

    def fake_save(user_email, pid, data, expected_version=None, source=None, **options):
        calls.append({"data_keys": sorted(data), "source": source})
        return 7

    monkeypatch.setattr(db, "save_project", fake_save)
    monkeypatch.setattr(state.auth, "current_user", lambda: "r@example.com")
    session.session_state["active_project_id"] = "p1"

    assert state.save_active() is True
    assert calls[0]["source"][1] == [{"Title": "A study", "Abstract": "Abstract"}]
    assert "original_records" not in calls[0]["data_keys"]
    assert state._store()[state._SOURCE_KEY] is False

    assert state.save_active() is True
    assert calls[1]["source"] is None


def test_a_failed_save_keeps_the_spreadsheet_owed(session, monkeypatch):
    def boom(*args, **kwargs):
        raise db.DatabaseError("database down")

    monkeypatch.setattr(db, "save_project", boom)
    monkeypatch.setattr(state.auth, "current_user", lambda: "r@example.com")
    session.session_state["active_project_id"] = "p1"
    state._store()[state._SOURCE_KEY] = True

    assert state.save_active() is False
    assert state._store()[state._SOURCE_KEY] is True


def test_a_legacy_document_moves_its_spreadsheet_to_the_side_table(session):
    state.load_into_session({
        "schema_version": state.SCHEMA_VERSION,
        "mode": state.MODE_PRISMA,
        "config": {},
        "papers": [],
        "original_columns": ["Title"],
        "original_records": [{"Title": "Legacy row"}],
        "cursors": {},
    }, version=4)
    assert state._store()[state._SOURCE_KEY] is True
    assert state._store()["original_df"].to_dict("records") == [{"Title": "Legacy row"}]
    assert state.project_version() == 4


def test_opening_a_project_loads_its_own_spreadsheet(engine, session, monkeypatch):
    """Cover both project switching and the first open of a session."""
    monkeypatch.setattr(state.auth, "current_user", lambda: "r@example.com")
    user = "r@example.com"
    first = db.create_project(user, "First", {"schema_version": state.SCHEMA_VERSION,
                                              "mode": state.MODE_PRISMA, "config": {},
                                              "papers": [], "cursors": {}})
    second = db.create_project(user, "Second", {"schema_version": state.SCHEMA_VERSION,
                                                "mode": state.MODE_PRISMA, "config": {},
                                                "papers": [], "cursors": {}})
    db.save_project_source(user, first, ["Title"], [{"Title": "First sheet"}])
    db.save_project_source(user, second, ["Title"], [{"Title": "Second sheet"}])

    state.set_active(first, "First")
    data, version = db.load_project_versioned(user, second)
    state.set_active(second, "Second")
    state.load_into_session(data, version, project_id=second)
    assert state._store()["original_df"].to_dict("records") == [{"Title": "Second sheet"}]

    state.set_active(None, None)
    data, version = db.load_project_versioned(user, first)
    state.set_active(first, "First")
    state.load_into_session(data, version, project_id=first)
    assert state._store()["original_df"].to_dict("records") == [{"Title": "First sheet"}]


def test_a_version_3_document_still_opens_and_is_upgraded(session):
    """Older documents with inline spreadsheets remain readable."""
    state.load_into_session({
        "schema_version": 3, "mode": state.MODE_PRISMA, "config": {}, "papers": [],
        "original_columns": ["Title"], "original_records": [{"Title": "Legacy row"}],
        "cursors": {},
    }, version=2)
    assert state._store()["original_df"].to_dict("records") == [{"Title": "Legacy row"}]
    assert state._store()[state._SOURCE_KEY] is True
    assert state.snapshot()["schema_version"] == state.SCHEMA_VERSION
