"""PDF lifecycle regressions using SQLite and in-memory storage."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from core import db, fulltext_storage
from features.workflow import documents, state


@pytest.fixture
def project(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    monkeypatch.setattr(state, "st", SimpleNamespace(session_state={}, error=lambda message: None))
    user = "reviewer@example.test"
    monkeypatch.setattr(state.auth, "current_user", lambda: user)
    data = state.new_project_data(state.MODE_DIRECT)
    data.pop("original_columns", None)
    data.pop("original_records", None)
    data["pending_pdf_deletions"] = []
    paper = state.new_paper("", "Study", "")
    paper["stages"]["fulltext"] = {
        "ai_verdict": "include", "decision": "agree", "document_hash": "old",
    }
    data["papers"] = [paper]
    pid = db.create_project(user, "Review", data)
    entry = {"paper_uid": paper["uid"], "filename": "old.pdf", "storage_key": "old.pdf",
             "sha256": "old", "status": "ok"}
    db.save_project(user, pid, data, expected_version=1, fulltext=entry,
                    source=(["Title"], [{"Title": "Study"}]))
    state.set_active(pid, "Review")
    state.reload_project(pid)
    files = {"old.pdf": b"old"}
    monkeypatch.setattr(fulltext_storage, "save_pdf", lambda key, content: files.__setitem__(key, content))
    monkeypatch.setattr(fulltext_storage, "delete_pdf", lambda key: files.pop(key, None))
    yield SimpleNamespace(user=user, pid=pid, paper=state.papers()[0], files=files)
    engine.dispose()


def attach(project, *, save=None, digest="new", paper=None, new_paper=False, retry=None):
    return documents.attach(save or state.prepare_save(), paper or project.paper, b"new",
                            "study.pdf", 2, digest, retry if retry is not None else [],
                            new_paper=new_paper)


def test_attachment_commits_detached_state_before_cleanup(project, monkeypatch):
    before = deepcopy(state.snapshot())
    save_pdf = fulltext_storage.save_pdf

    def upload(key, content):
        assert state.snapshot() == before
        assert db.load_project(project.user, project.pid) == before
        assert db.load_fulltexts(project.user, project.pid)[project.paper["uid"]]["storage_key"] == "old.pdf"
        save_pdf(key, content)

    monkeypatch.setattr(fulltext_storage, "save_pdf", upload)
    entry, changed = attach(project)
    assert changed
    assert set(project.files) == {"old.pdf", entry["storage_key"]}
    stage = state.papers()[0]["stages"]["fulltext"]
    assert stage["decision"] is None
    assert stage["history"][-1]["decision"] == "agree"
    assert db.load_fulltexts(project.user, project.pid)[project.paper["uid"]]["storage_key"] == entry["storage_key"]
    assert documents.cleanup(state.prepare_save())
    assert set(project.files) == {entry["storage_key"]}


@pytest.mark.parametrize("boundary", ["upload", "commit"])
def test_precommit_stop_rolls_back_unique_blob_without_mutating_live_state(project, monkeypatch, boundary):
    before = deepcopy(state.snapshot())
    if boundary == "upload":
        def interrupted(key, content):
            project.files[key] = content
            raise StopException()
        monkeypatch.setattr(fulltext_storage, "save_pdf", interrupted)
    else:
        def interrupted(*args, **kwargs):
            raise StopException()
        monkeypatch.setattr(db, "save_project", interrupted)
    with pytest.raises(StopException):
        attach(project)
    assert state.snapshot() == before
    assert db.load_project(project.user, project.pid) == before
    assert project.files == {"old.pdf": b"old"}


def test_ambiguous_commit_does_not_delete_committed_blob(project, monkeypatch):
    save = db.save_project

    def interrupted(*args, **kwargs):
        save(*args, **kwargs)
        raise StopException()

    monkeypatch.setattr(db, "save_project", interrupted)
    with pytest.raises(StopException):
        attach(project)
    metadata = db.load_fulltexts(project.user, project.pid)[project.paper["uid"]]
    assert metadata["storage_key"] in project.files
    assert "old.pdf" in project.files
    state.reload_project(project.pid)
    assert state.project_version() == 3
    monkeypatch.setattr(db, "save_project", save)
    assert documents.cleanup(state.prepare_save())
    assert metadata["storage_key"] in project.files


def test_new_paper_is_not_published_on_failed_attach(project, monkeypatch):
    paper = state.new_paper("", "New paper", "")
    before = deepcopy(state.snapshot())
    monkeypatch.setattr(db, "save_project", lambda *a, **k: (_ for _ in ()).throw(db.DatabaseError("Offline")))
    with pytest.raises(db.DatabaseError):
        attach(project, paper=paper, new_paper=True)
    assert state.snapshot() == before
    assert db.load_project(project.user, project.pid) == before
    assert paper["uid"] not in db.load_fulltexts(project.user, project.pid)
    assert project.files == {"old.pdf": b"old"}


def test_same_content_repair_keeps_decisions(project):
    before = deepcopy(state.papers()[0]["stages"])
    entry, changed = attach(project, digest="old")
    assert not changed
    assert state.papers()[0]["stages"] == before
    assert entry["storage_key"] != "old.pdf"
    assert documents.cleanup(state.prepare_save())
    assert set(project.files) == {entry["storage_key"]}


def test_stale_attachment_cannot_overwrite_or_delete_current_blob(project):
    stale = state.prepare_save()
    entry, _ = attach(project)
    with pytest.raises(db.ProjectConflictError):
        attach(project, save=stale, digest="other")
    assert set(project.files) == {"old.pdf", entry["storage_key"]}
    assert db.load_fulltexts(project.user, project.pid)[project.paper["uid"]]["storage_key"] == entry["storage_key"]


def test_active_blob_in_legacy_cleanup_queue_is_preserved(project):
    state.pending_pdf_deletions().append({"paper_uid": project.paper["uid"], "storage_key": "old.pdf"})
    assert state.save_active()
    assert documents.cleanup(state.prepare_save())
    assert project.files == {"old.pdf": b"old"}
    assert db.load_fulltexts(project.user, project.pid)[project.paper["uid"]]["storage_key"] == "old.pdf"
    assert state.pending_pdf_deletions() == []


def test_cleanup_read_failure_retains_blob_and_retry_target(project, monkeypatch):
    def unavailable(*args):
        raise db.DatabaseError("Offline")
    monkeypatch.setattr(db, "fulltext_key_in_use", unavailable)
    assert documents.cleanup_keys(project.user, ["old.pdf"]) == ["old.pdf"]
    assert project.files == {"old.pdf": b"old"}


def test_cleanup_failure_survives_project_reload(project, monkeypatch):
    entry, _ = attach(project)
    delete_pdf = fulltext_storage.delete_pdf

    def unavailable(key):
        raise fulltext_storage.FulltextStorageError("Offline")

    monkeypatch.setattr(fulltext_storage, "delete_pdf", unavailable)
    assert not documents.cleanup(state.prepare_save())
    state.reload_project(project.pid)
    assert state.pending_pdf_deletions() == [{"paper_uid": None, "storage_key": "old.pdf"}]
    monkeypatch.setattr(fulltext_storage, "delete_pdf", delete_pdf)
    assert documents.cleanup(state.prepare_save())
    assert set(project.files) == {entry["storage_key"]}


def test_removal_unlinks_metadata_in_same_transaction(project):
    documents.remove(state.prepare_save(), project.paper["uid"])
    assert state.papers() == []
    assert db.load_project(project.user, project.pid)["papers"] == []
    assert db.load_fulltexts(project.user, project.pid) == {}
    assert "old.pdf" in project.files
    assert documents.cleanup(state.prepare_save())
    assert not project.files


def test_csv_replacement_cleanup_cannot_delete_a_later_attachment(project):
    paper = state.new_paper("", "Replacement", "")
    source = (["Title"], [{"Title": "Replacement"}])
    documents.replace_papers(state.prepare_save(), [paper], source)
    assert db.load_fulltexts(project.user, project.pid) == {}
    assert db.load_project_source(project.user, project.pid) == source
    assert state._store()["original_df"].to_dict("records") == source[1]
    stale_cleanup = state.prepare_save()
    entry, _ = attach(project, paper=paper)
    with pytest.raises(db.ProjectConflictError):
        documents.cleanup(stale_cleanup)
    assert entry["storage_key"] in project.files
    assert paper["uid"] in db.load_fulltexts(project.user, project.pid)
    assert documents.cleanup(state.prepare_save())
    assert set(project.files) == {entry["storage_key"]}


def test_failed_csv_replacement_preserves_source_and_metadata(project):
    stale = state.prepare_save()
    assert state.save_active()
    before = db.load_project_bundle(project.user, project.pid)
    with pytest.raises(db.ProjectConflictError):
        documents.replace_papers(stale, [], ([], []))
    assert db.load_project_bundle(project.user, project.pid) == before
    assert project.paper["uid"] in db.load_fulltexts(project.user, project.pid)
    assert project.files == {"old.pdf": b"old"}
