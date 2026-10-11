"""Concurrent sessions: a stale writer is refused instead of overwriting."""
from sqlalchemy import create_engine, event, inspect, text
import pytest

from core import db


@pytest.fixture
def engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    return engine


def test_stale_writer_is_refused_and_the_saved_work_survives(engine):
    pid = db.create_project("reviewer@example.com", "Study", {"papers": []})
    _, version = db.load_project_versioned("reviewer@example.com", pid)

    new_version = db.save_project("reviewer@example.com", pid, {"papers": ["ai results"]},
                                  expected_version=version)
    assert new_version == version + 1

    # A second session still holds the original version.
    with pytest.raises(db.ProjectConflictError, match="another tab"):
        db.save_project("reviewer@example.com", pid, {"papers": []}, expected_version=version)
    assert db.load_project("reviewer@example.com", pid) == {"papers": ["ai results"]}


def test_force_save_overwrites_deliberately(engine):
    pid = db.create_project("reviewer@example.com", "Study", {"papers": []})
    db.save_project("reviewer@example.com", pid, {"papers": ["other tab"]}, expected_version=1)
    db.save_project("reviewer@example.com", pid, {"papers": ["mine"]}, expected_version=None)
    assert db.load_project("reviewer@example.com", pid) == {"papers": ["mine"]}


def test_missing_or_foreign_project_still_reports_access_not_conflict(engine):
    pid = db.create_project("reviewer@example.com", "Study", {})
    for user, target in (("other@example.com", pid), ("reviewer@example.com", "missing")):
        with pytest.raises(db.DatabaseError, match="not accessible"):
            db.save_project(user, target, {}, expected_version=1)


def test_legacy_projects_table_gains_a_version_column(monkeypatch):
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
        conn.execute(text(
            "INSERT INTO projects VALUES ('p1', 'u@example.com', 'Old', '{}', 'then', 'then')"
        ))
    db._upgrade_projects_schema(engine)
    assert "version" in {c["name"] for c in inspect(engine).get_columns("projects")}
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    assert db.load_project_versioned("u@example.com", "p1")[1] == 1


def _attachment(digest="old"):
    return {
        "paper_uid": "paper-1", "filename": "study.pdf", "sha256": digest,
        "status": "ok", "storage_key": f"pdf/{digest}", "file_size": 123,
        "page_count": 2, "error": None, "text": None,
    }


def test_bundle_loads_matching_document_version_and_source_in_one_select(engine):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {"papers": ["old"]})
    source = (["Title"], [{"Title": "New study"}])
    db.save_project(user, pid, {"papers": ["new"]}, expected_version=1, source=source)
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert db.load_project_bundle(user, pid) == ({"papers": ["new"]}, 2, source)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert len(statements) == 1
    assert "LEFT OUTER JOIN" in statements[0].upper()
    assert db.load_project_bundle("other@example.com", pid) == ({}, 0, ([], []))
    assert db.load_project_bundle(user, "missing") == ({}, 0, ([], []))


def test_bundle_handles_a_project_without_source_and_ignores_foreign_source(engine):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {"papers": []})
    assert db.load_project_bundle(user, pid) == ({"papers": []}, 1, ([], []))
    with engine.begin() as conn:
        conn.execute(db.project_sources.insert().values(
            project_id=pid, user_email="other@example.com", columns=["Private"],
            records=[{"Private": "not this user's source"}], created_at="then",
        ))
    assert db.load_project_bundle(user, pid) == ({"papers": []}, 1, ([], []))


def test_pdf_metadata_and_archived_results_commit_together(engine):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {"papers": ["old verdict"]})
    db.save_project(user, pid, {"papers": ["old verdict"]},
                    expected_version=1, fulltext=_attachment())
    version = db.save_project(user, pid, {"papers": ["pending"]},
                              expected_version=2, fulltext=_attachment("new"))
    assert version == 3
    assert db.load_project_versioned(user, pid) == ({"papers": ["pending"]}, 3)
    assert db.load_fulltexts(user, pid)["paper-1"]["sha256"] == "new"
    assert db.load_fulltexts(user, pid)["paper-1"]["storage_key"] == "pdf/new"


@pytest.mark.parametrize("new_source", [None, (["Title"], [{"Title": "Wrong study"}])])
def test_conflict_cannot_change_either_pdf_metadata_or_csv(engine, new_source):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {"papers": []})
    source = (["Title"], [{"Title": "Current study"}])
    db.save_project(user, pid, {"papers": ["current"]}, expected_version=1,
                    source=source, fulltext=_attachment())
    with pytest.raises(db.ProjectConflictError):
        db.save_project(user, pid, {"papers": ["stale"]}, expected_version=1,
                        source=new_source, fulltext=_attachment("stale"))
    assert db.load_project_bundle(user, pid) == ({"papers": ["current"]}, 2, source)
    assert db.load_fulltexts(user, pid)["paper-1"]["sha256"] == "old"


def test_metadata_failure_rolls_back_document_version_source_and_previous_pdf(engine):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {"papers": []})
    source = (["Title"], [{"Title": "Current study"}])
    db.save_project(user, pid, {"papers": ["current"]}, expected_version=1,
                    source=source, fulltext=_attachment())
    invalid = {**_attachment("new"), "status": None}
    with pytest.raises(db.DatabaseError):
        db.save_project(user, pid, {"papers": ["pending"]}, expected_version=2,
                        source=(["Title"], [{"Title": "Replacement"}]), fulltext=invalid)
    assert db.load_project_bundle(user, pid) == ({"papers": ["current"]}, 2, source)
    assert db.load_fulltexts(user, pid)["paper-1"]["sha256"] == "old"


def test_foreign_writer_cannot_add_a_source_or_pdf(engine):
    pid = db.create_project("reviewer@example.com", "Study", {})
    with pytest.raises(db.DatabaseError, match="not accessible"):
        db.save_project("other@example.com", pid, {}, expected_version=1,
                        source=(["Title"], [{"Title": "Foreign"}]), fulltext=_attachment())
    assert db.load_project_source("other@example.com", pid) == ([], [])
    assert db.load_fulltexts("other@example.com", pid) == {}


@pytest.mark.parametrize("removal", [{"remove_fulltexts": True}, {"remove_fulltext_uids": ["paper-1"]}])
def test_pdf_removal_is_atomic_with_project_version_and_source(engine, removal):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {"papers": ["paper-1"]})
    db.save_project(user, pid, {"papers": ["paper-1"]}, expected_version=1,
                    fulltext=_attachment())
    with pytest.raises(db.ProjectConflictError):
        db.save_project(user, pid, {"papers": []}, expected_version=1, **removal)
    assert db.fulltext_key_in_use(user, "pdf/old")
    source = (["Title"], [{"Title": "Replacement"}])
    db.save_project(user, pid, {"papers": []}, expected_version=2, source=source, **removal)
    assert db.load_fulltexts(user, pid) == {}
    assert not db.fulltext_key_in_use(user, "pdf/old")
    assert db.load_project_bundle(user, pid) == ({"papers": []}, 3, source)


def test_guarded_metadata_delete_does_not_remove_a_replacement(engine):
    user = "reviewer@example.com"
    pid = db.create_project(user, "Study", {})
    db.save_project(user, pid, {}, expected_version=1, fulltext=_attachment("new"))
    db.delete_fulltext(user, pid, "paper-1", expected_storage_key="pdf/old")
    assert db.fulltext_key_in_use(user, "pdf/new")
    assert not db.fulltext_key_in_use("other@example.com", "pdf/new")
    db.delete_fulltext(user, pid, "paper-1", expected_storage_key="pdf/new")
    assert not db.fulltext_key_in_use(user, "pdf/new")
