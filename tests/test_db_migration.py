"""Database compatibility tests; no Streamlit server or network required."""
from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
import pytest

from core import db


def test_legacy_fulltexts_table_gets_pdf_storage_columns():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE fulltexts (
                    id VARCHAR PRIMARY KEY,
                    project_id VARCHAR NOT NULL,
                    user_email VARCHAR NOT NULL,
                    paper_uid VARCHAR NOT NULL,
                    filename VARCHAR,
                    sha256 VARCHAR,
                    status VARCHAR NOT NULL,
                    text TEXT,
                    created_at VARCHAR NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO fulltexts
                    (id, project_id, user_email, paper_uid, status, text, created_at)
                VALUES
                    ('1', 'p', 'u@example.com', 'paper', 'ok', 'legacy', 'now')
                """
            )
        )

    db._upgrade_fulltexts_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("fulltexts")}
    assert set(db._FULLTEXT_ADDED_COLUMNS) <= columns
    with engine.connect() as conn:
        row = conn.execute(text("SELECT text FROM fulltexts WHERE id = '1'")).scalar_one()
    assert row == "legacy"


def test_save_project_requires_an_owned_existing_row(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    pid = db.create_project("reviewer@example.com", "Study", {"version": 1})
    db.save_project("reviewer@example.com", pid, {"version": 2})
    assert db.load_project("reviewer@example.com", pid) == {"version": 2}
    for user, target in (("other@example.com", pid), ("reviewer@example.com", "missing")):
        with pytest.raises(db.DatabaseError, match="not accessible"):
            db.save_project(user, target, {"version": 3})
    assert db.load_project("reviewer@example.com", pid) == {"version": 2}
