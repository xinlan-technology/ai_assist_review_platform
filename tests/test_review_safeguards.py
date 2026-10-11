"""Safeguards that keep recorded review work intact and honestly reported."""
from types import SimpleNamespace

import pandas as pd
import pytest

from core import csv_io
from features.extraction import state as extraction
from features.workflow import state

A, F = state.STAGE_ABSTRACT, state.STAGE_FULLTEXT


def test_unmatched_columns_are_not_guessed():
    assert csv_io.guess_column(["Title", "Abstract"], csv_io.TITLE_CANDIDATES) == "Title"
    assert csv_io.guess_column(["Field A", "Field B"], csv_io.TITLE_CANDIDATES) is None
    assert csv_io.guess_column([], csv_io.TITLE_CANDIDATES) is None


def test_clicking_disagree_again_keeps_the_confirmed_override():
    paper = state.new_paper("", "Study", "Abstract")
    state.set_ai_result(paper, A, state.VERDICT_INCLUDE, "r", "p", "m", "hash", "v")
    state.record_disagree(paper, A)
    state.set_human_verdict(paper, A, state.VERDICT_EXCLUDE, review_hash="hash")

    state.record_disagree(paper, A)

    assert state.current_final_verdict(paper, A, "hash") == state.VERDICT_EXCLUDE


def test_archive_never_drops_the_last_confirmed_extraction():
    spec = {"instructions": "", "prompt_version": "extraction-v1", "questions": [
        {"id": "q1", "text": "Q", "type": "open_text", "options": [], "guidance": ""}]}
    answers = {"q1": {"values": ["Forest"], "other_text": "", "page": 1,
                      "quote": "evidence", "issue": ""}}
    paper = state.new_paper("", "Study", "")
    extraction.save_review(paper, spec, "digest", answers, page_count=2, confirm=True)
    extraction.archive(paper)
    for _ in range(6):
        extraction.set_error(paper, spec, "digest", "network down", "call_failed")
        extraction.archive(paper)

    history = extraction.get(paper)["history"]
    assert len(history) == 7
    assert history[0].get("review_state") == "confirmed"


@pytest.fixture
def session(monkeypatch):
    fake = SimpleNamespace(session_state={}, error=lambda *a, **k: None)
    monkeypatch.setattr(state, "st", fake)
    return fake


def test_export_marks_withheld_results_instead_of_blanking_them(session):
    paper = state.new_paper("10.1/a", "A study", "Abstract")
    state.set_ai_result(paper, A, state.VERDICT_INCLUDE, "fits", "p", "m",
                        state.criteria_hash("first criteria"), "v")
    state.record_agree(paper, A)
    state.set_ai_result(paper, F, state.VERDICT_INCLUDE, "full text fits", "p", "m",
                        state.criteria_hash("fulltext criteria"), "v")
    state.record_agree(paper, F)

    session.session_state["project_store"] = {
        "mode": state.MODE_PRISMA,
        "config": {"abstract_criteria": "second criteria",
                   "fulltext_criteria": "fulltext criteria"},
        "papers": [paper],
        "original_df": pd.DataFrame([{"Title": "A study"}]),
        "cursors": {},
    }

    row = state.results_dataframe().loc[0]
    assert row["Abstract: Final"] == state.LABEL_STALE
    assert row["Full-text: AI verdict"] == "Include"
    assert row["Full-text: AI reason"] == "full text fits"
    assert row["Full-text: Final"] == state.LABEL_WITHHELD


def test_jump_picker_does_not_undo_an_auto_advance(monkeypatch):
    from core import ui

    fake = SimpleNamespace(session_state={}, selectbox=lambda *a, **k: None)
    monkeypatch.setattr(ui, "st", fake)
    labels = ["1. A", "2. B", "3. C"]

    assert ui.jump_to_paper("jump", 0, labels) == 0
    assert ui.jump_to_paper("jump", 1, labels) == 1
    assert fake.session_state["jump"] == 1

    fake.session_state["jump"] = 2
    assert ui.jump_to_paper("jump", 1, labels) == 2

    fake.session_state["jump:moved"] = 1
    assert ui.jump_to_paper("jump", 2, labels) == 1


def test_a_row_without_a_version_can_still_be_saved(monkeypatch):
    """Schema upgrades leave legacy rows with a nullable version column."""
    from sqlalchemy import create_engine, text

    from core import db

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text(
            """
            CREATE TABLE projects (
                id VARCHAR PRIMARY KEY, user_email VARCHAR NOT NULL, name VARCHAR NOT NULL,
                data JSON NOT NULL, created_at VARCHAR NOT NULL, updated_at VARCHAR NOT NULL
            )
            """
        ))
    db._upgrade_projects_schema(engine)
    db._metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO projects (id, user_email, name, data, created_at, updated_at) "
            "VALUES ('p1', 'r@example.com', 'Legacy', '{}', 'then', 'then')"
        ))
        assert conn.execute(text("SELECT version FROM projects")).scalar() is None
    monkeypatch.setattr(db, "_get_engine", lambda: engine)

    assert db.load_project_versioned("r@example.com", "p1")[1] == 1
    with pytest.raises(db.ProjectConflictError):
        db.save_project("r@example.com", "p1", {"papers": ["wrong version"]}, expected_version=3)
    assert db.save_project("r@example.com", "p1", {"papers": ["kept"]}, expected_version=1) == 2
    with pytest.raises(db.ProjectConflictError):
        db.save_project("r@example.com", "p1", {"papers": ["stale"]}, expected_version=1)
    assert db.load_project("r@example.com", "p1") == {"papers": ["kept"]}


def test_decision_history_export_lists_every_archived_screening_outcome(monkeypatch):
    paper = state.new_paper("10.1000/abc", "Study", "Abstract")
    for round_number, verdict in enumerate(["include", "exclude", "include"]):
        state.set_ai_result(paper, state.STAGE_ABSTRACT, verdict, f"Reason {round_number}",
                            "OpenAI", "gpt-4.1", f"hash-{round_number}", "v1")
        state.stage_state(paper, state.STAGE_ABSTRACT)["source_run_id"] = f"run-{round_number}"
        state.record_agree(paper, state.STAGE_ABSTRACT)
        state.archive_ai_result(paper, state.STAGE_ABSTRACT)
    untouched = state.new_paper("", "Never archived", "")
    monkeypatch.setattr(state, "papers", lambda: [paper, untouched])

    history = state.decision_history_dataframe()

    assert list(history["AI reason or error"]) == ["Reason 0", "Reason 1", "Reason 2"]
    assert list(history["AI verdict"]) == ["Include", "Exclude", "Include"]
    assert set(history["Decision"]) == {state.DECISION_AGREE}
    assert list(history["Source run ID"]) == ["run-0", "run-1", "run-2"]
    assert set(history["Paper"]) == {1} and set(history["Stage"]) == {state.STAGE_TITLES[state.STAGE_ABSTRACT]}
