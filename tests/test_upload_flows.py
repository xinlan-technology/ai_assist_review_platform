"""Upload-page integration tests with SQLite and in-memory files."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pypdf import PdfWriter
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import auth, db, fulltext_storage
from features.screening import judge
from features.workflow import state


ROOT = Path(__file__).resolve().parents[1]
USER = "upload@example.test"


def upload(name, content):
    stream = BytesIO(content)
    stream.name = name
    return stream


def pdf(name):
    stream = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": name})
    writer.write(stream)
    return upload(name, stream.getvalue())


def button(app, label):
    return next(item for item in app.button if item.label == label)


@pytest.fixture
def upload_ui(monkeypatch):
    engine = create_engine("sqlite:///:memory:", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    monkeypatch.setattr(auth, "current_user", lambda: USER)
    monkeypatch.setattr(auth, "sidebar_user", lambda: None)
    monkeypatch.setattr(st, "pdf", lambda *args, **kwargs: st.caption("Offline PDF viewer"))
    monkeypatch.setattr(st, "page_link", lambda *args, **kwargs: None)
    monkeypatch.setattr(fulltext_storage, "ensure_configured", lambda: None)
    monkeypatch.setattr(fulltext_storage, "backend_label", lambda: "in-memory test storage")
    uploads, files = {}, {}
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploads.get(label))
    monkeypatch.setattr(fulltext_storage, "save_pdf", lambda key, data: files.update({key: data}))
    monkeypatch.setattr(fulltext_storage, "load_pdf", lambda key: files[key])
    monkeypatch.setattr(fulltext_storage, "delete_pdf", lambda key: files.pop(key, None))
    model = Mock(side_effect=AssertionError("Uploads must not call a model."))
    monkeypatch.setattr(judge, "judge_abstract", model)
    monkeypatch.setattr(judge, "judge_fulltext", model)
    st.cache_data.clear()

    def create(stage, mode, *, existing=False, attached=False):
        data = state.new_project_data(mode)
        data["config"].update(abstract_criteria="Include relevant studies.",
                               fulltext_criteria="Include relevant studies.")
        data["pending_pdf_deletions"] = []
        source, entry = ([], []), None
        if existing:
            paper = state.new_paper("", "Original study", "Original abstract.")
            state.set_human_verdict(paper, state.STAGE_ABSTRACT, "include",
                                    review_hash=state.criteria_hash(data["config"]["abstract_criteria"]))
            data["papers"] = [paper]
            source = (["Title", "Abstract"], [{"Title": paper["title"], "Abstract": paper["abstract"]}])
            if attached:
                content = pdf("old.pdf").getvalue()
                files["old.pdf"] = content
                entry = {
                    "paper_uid": paper["uid"], "filename": "old.pdf", "storage_key": "old.pdf",
                    "sha256": fulltext_storage.sha256(content), "file_size": len(content),
                    "page_count": 1, "status": "ok",
                }
        pid = db.create_project(USER, "Upload regression", data)
        db.save_project(USER, pid, data, expected_version=1, source=source, fulltext=entry)
        data, version, source = db.load_project_bundle(USER, pid)
        app = AppTest.from_file(str(ROOT / "views" / f"{stage}_screening.py"), default_timeout=15)
        app.session_state["active_project_id"] = pid
        app.session_state["active_project_name"] = "Upload regression"
        app.session_state["project_store"] = {
            "mode": mode, "config": deepcopy(data["config"]), "papers": deepcopy(data["papers"]),
            "original_df": pd.DataFrame(source[1], columns=source[0]), "cursors": {},
            "pending_pdf_deletions": [], state._VERSION_KEY: version, state._SOURCE_KEY: False,
        }
        return app, pid

    yield SimpleNamespace(create=create, uploads=uploads, files=files)
    model.assert_not_called()
    st.cache_data.clear()
    engine.dispose()


@pytest.mark.parametrize("save_fails", [False, True])
def test_direct_upload_button_saves_complete_papers_or_nothing(upload_ui, monkeypatch, save_fails):
    first, second = pdf("first.pdf"), pdf("second.pdf")
    upload_ui.uploads["PDF files"] = [first, second]
    app, pid = upload_ui.create("fulltext", state.MODE_DIRECT)
    app.run()
    assert not app.exception
    before = db.load_project_bundle(USER, pid)
    if save_fails:
        monkeypatch.setattr(db, "save_project", Mock(side_effect=db.DatabaseError("Offline save failure")))
    button(app, "Add PDFs to project").click().run()
    assert not app.exception
    data, version, _ = db.load_project_bundle(USER, pid)
    metadata = db.load_fulltexts(USER, pid)
    store = app.session_state["project_store"]
    if save_fails:
        assert (data, version, db.load_project_source(USER, pid)) == before
        assert store["papers"] == []
        assert metadata == {}
        assert upload_ui.files == {}
        assert any("could not be added" in error.value for error in app.error)
    else:
        assert [paper["title"] for paper in data["papers"]] == ["first", "second"]
        assert set(metadata) == {paper["uid"] for paper in data["papers"]}
        assert {meta["filename"] for meta in metadata.values()} == {"first.pdf", "second.pdf"}
        assert all(meta["page_count"] == 1 for meta in metadata.values())
        assert set(upload_ui.files.values()) == {first.getvalue(), second.getvalue()}
        assert store["papers"] == data["papers"]
    assert store[state._VERSION_KEY] == version


def test_prisma_attach_button_preserves_the_approved_paper(upload_ui):
    document = pdf("matched.pdf")
    upload_ui.uploads["Matching PDF"] = document
    app, pid = upload_ui.create("fulltext", state.MODE_PRISMA, existing=True)
    before = db.load_project(USER, pid)["papers"][0]
    app.run()
    assert not app.exception
    button(app, "Attach PDF").click().run()
    assert not app.exception
    data, version = db.load_project_versioned(USER, pid)
    assert len(data["papers"]) == 1
    paper = data["papers"][0]
    assert paper["uid"] == before["uid"]
    assert paper["stages"]["abstract"] == before["stages"]["abstract"]
    metadata = db.load_fulltexts(USER, pid)
    assert set(metadata) == {paper["uid"]}
    entry = metadata[paper["uid"]]
    assert entry["filename"] == "matched.pdf"
    assert upload_ui.files[entry["storage_key"]] == document.getvalue()
    assert app.session_state["project_store"][state._VERSION_KEY] == version
    assert app.session_state["project_store"]["papers"] == data["papers"]


@pytest.mark.parametrize("mode,expected", [
    (state.MODE_DIRECT, {"PDF files", "Replacement PDF"}),
    (state.MODE_PRISMA, {"Matching PDF"}),
])
def test_pdf_upload_widgets_enforce_the_storage_limit(upload_ui, monkeypatch, mode, expected):
    seen = set()

    def uploader(label, **options):
        assert options["max_upload_size"] == fulltext_storage.MAX_UPLOAD_PDF_BYTES // (1024 * 1024)
        seen.add(label)
        return None

    monkeypatch.setattr(st, "file_uploader", uploader)
    app, _ = upload_ui.create("fulltext", mode, existing=True, attached=True)
    app.run()
    assert not app.exception
    assert seen == expected


@pytest.mark.parametrize("save_fails", [False, True])
def test_csv_load_button_replaces_source_and_unlinks_pdfs_atomically(upload_ui, monkeypatch, save_fails):
    upload_ui.uploads["CSV file"] = upload(
        "replacement.csv", b"Title,Abstract,Note\nReplacement study,Replacement abstract.,Kept column\n",
    )
    app, pid = upload_ui.create("abstract", state.MODE_PRISMA, existing=True, attached=True)
    before = db.load_project_bundle(USER, pid)
    old_metadata = db.load_fulltexts(USER, pid)
    old_files = dict(upload_ui.files)
    app.run()
    assert not app.exception
    assert button(app, "Load papers into project").disabled
    next(item for item in app.checkbox if item.key == "confirm_replace_papers").check().run()
    if save_fails:
        monkeypatch.setattr(db, "save_project", Mock(side_effect=db.DatabaseError("Offline save failure")))
    button(app, "Load papers into project").click().run()
    assert not app.exception
    data, version, source = db.load_project_bundle(USER, pid)
    store = app.session_state["project_store"]
    if save_fails:
        assert (data, version, source) == before
        assert db.load_fulltexts(USER, pid) == old_metadata
        assert upload_ui.files == old_files
        assert any("Could not save the imported papers" in error.value for error in app.error)
    else:
        assert len(data["papers"]) == 1
        assert data["papers"][0]["title"] == "Replacement study"
        assert data["papers"][0]["uid"] not in old_metadata
        assert db.load_fulltexts(USER, pid) == {}
        assert upload_ui.files == {}
        assert data["pending_pdf_deletions"] == []
        assert source == (["Title", "Abstract", "Note"], [{
            "Title": "Replacement study", "Abstract": "Replacement abstract.", "Note": "Kept column",
        }])
    expected_papers = deepcopy(data["papers"])
    for paper in expected_papers:
        state.stage_state(paper, state.STAGE_ABSTRACT)
    assert store["papers"] == expected_papers
    assert store[state._VERSION_KEY] == version
    assert store["original_df"].to_dict("records") == source[1]


def test_direct_upload_names_the_files_left_out_after_a_database_error(upload_ui, monkeypatch):
    upload_ui.uploads["PDF files"] = [pdf("first.pdf"), pdf("second.pdf"), pdf("third.pdf")]
    app, pid = upload_ui.create("fulltext", state.MODE_DIRECT)
    app.run()
    save, calls = db.save_project, []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise db.DatabaseError("Offline save failure")
        return save(*args, **kwargs)

    monkeypatch.setattr(db, "save_project", fail_second)
    button(app, "Add PDFs to project").click().run()

    assert not app.exception
    data, _, _ = db.load_project_bundle(USER, pid)
    assert [paper["title"] for paper in data["papers"]] == ["first"]
    # The page redraws after the partial success; the report must survive that.
    report = " ".join(error.value for error in app.error).replace("\\", "")
    assert "second.pdf: Offline save failure" in report
    assert "Not processed after the database error: third.pdf" in report
    assert any("Added 1 PDF" in notice.value for notice in app.success)
