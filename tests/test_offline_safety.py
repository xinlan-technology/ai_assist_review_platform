"""Dependency-free regressions using real functions and in-memory I/O boundaries."""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def load_functions(filename, bindings, names=None):
    tree = ast.parse((ROOT / filename).read_text())
    allowed = (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    tree.body = [node for node in tree.body if isinstance(node, allowed)
                 and (names is None or getattr(node, "name", None) in names)]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    namespace = dict(copy=copy, deepcopy=copy.deepcopy, hashlib=hashlib, inspect=inspect,
                     json=json, math=math, re=re, uuid=uuid, uuid4=uuid.uuid4,
                     datetime=datetime, timezone=timezone)
    namespace.update(bindings)
    exec(compile(tree, filename, "exec"), namespace)
    return namespace


class Rerun(BaseException):
    pass


class Stop(BaseException):
    pass


class FakeUI:
    def __init__(self):
        self.session_state = {}
        self.clicked = None
        self.acknowledged = False
        self.downloads = []
        self.buttons = []
        self.error = Mock()
        self.warning = Mock()

    def button(self, label, **options):
        self.buttons.append(label)
        return label == self.clicked and not options.get("disabled", False)

    def checkbox(self, label):
        return self.acknowledged

    def download_button(self, label, data, **options):
        self.downloads.append(json.loads(data))

    def rerun(self):
        raise Rerun()

    def stop(self):
        raise Stop()


class Frame:
    """Only the DataFrame serialization boundary used by session persistence."""
    def __init__(self, records=None, columns=None):
        self.records = copy.deepcopy(records or [])
        self.columns = list(columns or (self.records[0] if self.records else []))

    def to_json(self, orient):
        assert orient == "records"
        return json.dumps(self.records)


class DatabaseError(Exception):
    pass


class ProjectConflictError(DatabaseError):
    pass


class StorageError(Exception):
    pass


class FakeDatabase:
    DatabaseError = DatabaseError
    ProjectConflictError = ProjectConflictError

    def __init__(self):
        self.data, self.metadata = {}, {}
        self.source = (["title"], [{"title": "Original study"}])
        self.version = 3
        self.failed = False
        self.writes = []
        self.deleted = []
        self.bundle_reads = 0

    def save_project(self, user, pid, data, expected_version=None, source=None, fulltext=None,
                     remove_fulltexts=False, remove_fulltext_uids=None,
                     import_legacy_runs=True, removed_paper_uids=None):
        if self.failed:
            raise DatabaseError("Unavailable")
        if expected_version != self.version:
            raise ProjectConflictError("Another session saved this project.")
        self.writes.append(copy.deepcopy((data, source, fulltext)))
        self.data = copy.deepcopy(data)
        if source is not None:
            self.source = copy.deepcopy(source)
        removals = list(self.metadata) if remove_fulltexts else remove_fulltext_uids or []
        for uid in removals:
            self.delete_fulltext(user, pid, uid)
        if fulltext is not None:
            self.metadata[fulltext["paper_uid"]] = copy.deepcopy(fulltext)
        self.version += 1
        return self.version

    def load_project_bundle(self, user, pid):
        self.bundle_reads += 1
        return copy.deepcopy(self.data), self.version, copy.deepcopy(self.source)

    def load_project_source(self, *args):
        raise AssertionError("The source must be read in the same bundle as the project.")

    def load_fulltexts(self, user, pid):
        return copy.deepcopy(self.metadata)

    def fulltext_key_in_use(self, user, key):
        return any(meta.get("storage_key") == key for meta in self.metadata.values())

    def delete_fulltext(self, user, pid, uid):
        self.deleted.append(uid)
        self.metadata.pop(uid, None)


class FakeStorage:
    FulltextStorageError = StorageError

    def __init__(self):
        self.files = {"old.pdf": b"old"}
        self.failed = set()
        self.saved = []
        self.deleted = []
        self.on_save = lambda key: None

    def object_key(self, user, pid, uid, suffix):
        return f"{pid}/{uid}/{suffix}.pdf"

    def save_pdf(self, key, data):
        self.on_save(key)
        self.saved.append(key)
        self.files[key] = data

    def delete_pdf(self, key):
        if key in self.failed:
            raise StorageError("Unavailable")
        self.deleted.append(key)
        self.files.pop(key, None)


class OfflineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.st, self.db, self.storage = FakeUI(), FakeDatabase(), FakeStorage()
        schema = load_functions("features/extraction/schema.py", {"InvalidModelResponse": ValueError})
        self.schema = SimpleNamespace(**schema)
        extraction = load_functions("features/extraction/state.py", {
            "spec_hash": self.schema.spec_hash, "validate_response": self.schema.validate_response,
        })
        self.extraction = SimpleNamespace(**extraction)
        package = ModuleType("features.extraction")
        package.state = self.extraction
        modules = patch.dict(sys.modules, {"features.extraction": package})
        modules.start()
        self.addCleanup(modules.stop)
        self.auth = SimpleNamespace(current_user=lambda: "reviewer@example.test")
        self.wf = load_functions("features/workflow/state.py", {
            "st": self.st, "db": self.db, "auth": self.auth, "pd": SimpleNamespace(DataFrame=Frame),
        })
        self.state = SimpleNamespace(**self.wf)
        self.state.set_active("project", "Offline test")
        data = self.state.new_project_data(self.state.MODE_DIRECT)
        self.paper = self.state.new_paper("", "Original study", "")
        self.paper["uid"] = "paper1"
        self.paper["stages"]["fulltext"] = {"ai_verdict": "include", "decision": "agree"}
        data["papers"] = [self.paper]
        self.state.load_into_session(data, version=3, source=self.db.source)
        self.db.data = copy.deepcopy(self.state.snapshot())
        self.metadata = {"paper1": {"storage_key": "old.pdf", "sha256": "old"}}
        self.db.metadata = copy.deepcopy(self.metadata)
        documents = load_functions("features/workflow/documents.py", {
            "db": self.db, "state": self.state, "fulltext_storage": self.storage,
        })
        self.documents = SimpleNamespace(**documents)
        self.fulltext = load_functions("views/fulltext_screening.py", {
            "st": self.st, "db": self.db, "state": self.state,
            "documents": self.documents,
            "fulltext_storage": self.storage, "user": "reviewer@example.test",
            "project_id": "project", "_cached_pdf": SimpleNamespace(clear=Mock()),
        }, {"_attach_prepared", "_cleanup_removed_pdfs"})

    def attach(self, digest="new"):
        return self.fulltext["_attach_prepared"](self.paper, b"new", "study.pdf", 2, digest, self.metadata)

    def test_failed_autosave_stays_blocked_until_retry_succeeds(self):
        spec = self.schema.build_spec("Use study evidence.", [
            {"id": "q1", "text": "Setting?", "type": "single_choice", "options": ["Forest"]}])
        page = load_functions("features/extraction/drafts.py", {
            "st": self.st, "state": self.state, "schema": self.schema, "extraction": self.extraction,
        }, {"review_key", "_keys", "_widget_answers", "_form_shape", "_comparable",
            "_provenance", "_current", "autosave"})
        prefix, digest = "extraction:project", "source"
        key = page["review_key"](prefix, self.paper, spec, digest)
        self.st.session_state.update({
            f"{prefix}:open_form": {"uid": "paper1", "digest": digest, "review_key": key,
                                    "pages": 2, "editable": True},
            f"{key}:q1:single": "Forest", f"{key}:q1:page": 1,
            f"{key}:q1:quote": "Typed evidence.", f"{key}:q1:checked": True,
        })
        self.db.failed = True
        with self.assertRaises(Rerun):
            page["autosave"](prefix, spec)
        record = self.extraction.get(self.paper)
        self.assertEqual(record["final_answers"]["q1"]["quote"], "Typed evidence.")
        self.assertEqual(record["checked_questions"], ["q1"])
        for click in (None, "Retry saving results"):
            self.st.clicked = click
            with self.assertRaises(Stop):
                self.state.require_saved_results()
            self.assertTrue(self.state.has_unsaved_results())
        self.assertEqual(self.db.writes, [])
        self.db.failed = False
        with self.assertRaises(Rerun):
            self.state.require_saved_results()
        self.assertFalse(self.state.has_unsaved_results())
        self.assertEqual(len(self.db.writes), 1)
        page["autosave"](prefix, spec)
        self.assertEqual(len(self.db.writes), 1)

    def test_conflict_cannot_force_overwrite_and_backup_keeps_matching_csv(self):
        original_source = copy.deepcopy(self.db.source)
        self.db.source = (["title"], [{"title": "Other window's CSV"}])
        self.db.version += 1
        self.assertFalse(self.state.save_result())
        self.assertNotIn("force", inspect.signature(self.state.save_active).parameters)
        with self.assertRaises(TypeError):
            self.state.save_active(force=True)
        with self.assertRaises(Stop):
            self.state.require_saved_results()
        backup = self.st.downloads[-1]
        self.assertEqual(backup["original_records"], original_source[1])
        self.assertEqual(backup["papers"][0]["title"], backup["original_records"][0]["title"])
        self.assertFalse(any("overwrit" in label for label in self.st.buttons))
        self.assertEqual(self.db.writes, [])
        self.assertEqual(self.db.source[1], [{"title": "Other window's CSV"}])

    def test_conflict_reload_reads_data_and_source_from_one_bundle(self):
        self.db.version += 1
        self.state.save_result()
        self.db.data["papers"][0]["title"] = "Replacement study"
        self.db.source = (["title"], [{"title": "Replacement study"}])
        self.st.session_state["extraction:project:open_form"] = {"uid": "paper1"}
        self.st.session_state["extraction:project:setup_draft"] = {"instructions": "Old draft"}
        self.st.session_state["llm:api_key:OpenAI"] = "synthetic-session-key"
        self.st.clicked, self.st.acknowledged = "Discard local changes and reload", True
        with self.assertRaises(Rerun):
            self.state.require_saved_results()
        self.assertEqual(self.db.bundle_reads, 1)
        self.assertEqual(self.state.papers()[0]["title"], "Replacement study")
        self.assertEqual(self.state._store()["original_df"].records, self.db.source[1])
        self.assertEqual(self.state.project_version(), self.db.version)
        self.assertFalse(self.state.has_unsaved_results())
        self.assertNotIn("extraction:project:open_form", self.st.session_state)
        self.assertNotIn("extraction:project:setup_draft", self.st.session_state)
        self.assertEqual(self.st.session_state["llm:api_key:OpenAI"], "synthetic-session-key")

    def test_pdf_replacement_commits_metadata_and_archived_decisions_together(self):
        before = copy.deepcopy(self.db.data)
        self.storage.failed.add("old.pdf")

        def while_uploading(key):
            self.assertEqual(self.db.metadata["paper1"]["storage_key"], "old.pdf")
            self.assertEqual(self.db.data, before)
            self.assertNotIn(key, self.storage.files)

        self.storage.on_save = while_uploading
        self.assertTrue(self.attach())
        saved_data, _, saved_pdf = self.db.writes[0]
        stage = saved_data["papers"][0]["stages"]["fulltext"]
        self.assertIsNone(stage["decision"])
        self.assertEqual(stage["history"][-1]["ai_verdict"], "include")
        self.assertEqual(saved_pdf["storage_key"], self.storage.saved[0])
        self.assertEqual(self.db.metadata["paper1"]["storage_key"], self.storage.saved[0])
        self.assertEqual(saved_data["pending_pdf_deletions"], [{"paper_uid": None, "storage_key": "old.pdf"}])
        self.assertEqual(self.state.pending_pdf_deletions(), saved_data["pending_pdf_deletions"])

    def test_stale_pdf_writer_rolls_back_and_cannot_delete_other_attempts(self):
        before_stages = copy.deepcopy(self.paper["stages"])
        before_metadata = copy.deepcopy(self.metadata)
        self.db.version += 1
        for _ in range(2):
            with self.assertRaises(ProjectConflictError):
                self.attach()
            self.assertEqual(self.paper["stages"], before_stages)
            self.assertEqual(self.metadata, before_metadata)
            self.assertEqual(self.db.metadata, before_metadata)
            self.assertEqual(self.state.pending_pdf_deletions(), [])
        self.assertNotEqual(*self.storage.saved)
        self.assertEqual(self.storage.files, {"old.pdf": b"old"})
        self.assertNotIn("old.pdf", self.storage.deleted)
        self.assertEqual(self.db.writes, [])

    def test_removal_commits_metadata_before_cleanup_and_retains_failed_targets(self):
        self.documents.remove(self.state.prepare_save(), "paper1")
        target = {"paper_uid": None, "storage_key": "old.pdf"}
        self.assertEqual(self.db.metadata, {})
        self.assertEqual(self.db.data["papers"], [])
        self.assertIn("old.pdf", self.storage.files)
        self.storage.failed.add("old.pdf")
        self.assertFalse(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertEqual(self.db.deleted, ["paper1"])
        self.assertEqual(self.state.pending_pdf_deletions(), [target])
        self.storage.failed.clear()
        self.assertTrue(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertEqual(self.db.deleted, ["paper1"])
        self.assertEqual(self.state.pending_pdf_deletions(), [])
        self.assertEqual(self.db.writes[-1][0]["pending_pdf_deletions"], [])

    def test_cleanup_commit_failure_restores_retry_queue(self):
        self.documents.remove(self.state.prepare_save(), "paper1")
        target = {"paper_uid": None, "storage_key": "old.pdf"}
        self.db.failed = True
        self.assertFalse(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertEqual(self.state.pending_pdf_deletions(), [target])
        self.assertEqual(self.db.data["pending_pdf_deletions"], [target])
        self.assertNotIn("old.pdf", self.storage.files)
        self.db.failed = False
        self.assertTrue(self.fulltext["_cleanup_removed_pdfs"]())

    def test_cleanup_continues_past_failure_and_retains_only_failed_targets(self):
        self.documents.remove(self.state.prepare_save(), "paper1")
        failed = {"paper_uid": None, "storage_key": "old.pdf"}
        completed = {"paper_uid": None, "storage_key": "other.pdf"}
        self.state.pending_pdf_deletions().append(completed)
        self.storage.files["other.pdf"] = b"other"
        self.storage.failed.add("old.pdf")
        self.assertFalse(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertEqual(self.db.deleted, ["paper1"])
        self.assertEqual(self.storage.deleted, ["other.pdf"])
        self.assertEqual(self.state.pending_pdf_deletions(), [failed])
        self.assertEqual(self.db.writes[-1][0]["pending_pdf_deletions"], [failed])

    def test_legacy_metadata_cleanup_failure_does_not_delete_the_file(self):
        target = {"paper_uid": "paper1", "storage_key": "old.pdf"}
        self.state._store()["papers"] = []
        self.state.pending_pdf_deletions().append(target)
        self.assertTrue(self.state.save_active())
        self.db.failed = True
        self.assertFalse(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertIn("old.pdf", self.storage.files)
        self.assertEqual(self.db.metadata["paper1"]["storage_key"], "old.pdf")
        self.assertEqual(self.state.pending_pdf_deletions(), [target])
        self.db.failed = False
        self.assertTrue(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertEqual(self.db.deleted, ["paper1"])

    def test_cleanup_never_deletes_a_currently_referenced_file(self):
        self.state.pending_pdf_deletions().append({"paper_uid": "paper1", "storage_key": "old.pdf"})
        self.assertTrue(self.fulltext["_cleanup_removed_pdfs"]())
        self.assertEqual(self.storage.deleted, [])
        self.assertEqual(self.db.deleted, [])
        self.assertEqual(self.storage.files, {"old.pdf": b"old"})
        self.assertEqual(self.db.metadata["paper1"]["storage_key"], "old.pdf")
        self.assertEqual(self.state.pending_pdf_deletions(), [])

    def test_pending_cleanup_survives_saved_snapshot_and_reload(self):
        self.documents.remove(self.state.prepare_save(), "paper1")
        saved = copy.deepcopy(self.db.data)
        self.assertEqual(saved["papers"], [])
        self.state.clear_session()
        self.state.reload_project("project")
        self.assertEqual(self.state.snapshot(), saved)
        self.assertEqual(self.state.pending_pdf_deletions(), [{"paper_uid": None, "storage_key": "old.pdf"}])
        self.assertEqual(self.db.metadata, {})

    def test_failed_csv_import_restores_the_source_pending_flag(self):
        before = copy.deepcopy(self.state.snapshot())
        self.db.failed = True
        def replace_spreadsheet():
            store = self.state._store()
            store.update(papers=[], original_df=Frame([{"title": "Replacement"}]), cursors={})
            store[self.state._SOURCE_KEY] = True

        with self.assertRaises(Stop):
            self.state.commit(replace_spreadsheet)
        self.assertEqual(self.state.snapshot(), before)
        self.assertFalse(self.state._store()[self.state._SOURCE_KEY])
        self.assertEqual(self.state._store()["original_df"].records, self.db.source[1])

    def test_failed_upload_cleanup_retains_its_unique_target(self):
        self.db.version += 1
        self.storage.on_save = self.storage.failed.add
        with self.assertRaises(ProjectConflictError):
            self.attach()
        self.assertEqual(self.st.session_state["_uncommitted_pdf_cleanup"], self.storage.saved)
        self.assertEqual(self.db.metadata["paper1"]["storage_key"], "old.pdf")

    def test_interrupted_attachment_leaves_saved_and_session_data_unchanged(self):
        before = copy.deepcopy(self.state.snapshot())
        with patch.object(self.db, "save_project", side_effect=Stop):
            with self.assertRaises(Stop):
                self.attach()
        self.assertEqual(self.state.snapshot(), before)
        self.assertEqual(self.db.data, before)
        self.assertEqual(self.storage.files, {"old.pdf": b"old"})
        self.assertEqual(self.db.metadata["paper1"]["storage_key"], "old.pdf")

    def test_rerun_after_attachment_keeps_the_committed_file_and_version(self):
        self.fulltext["_cached_pdf"].clear.side_effect = Rerun
        with self.assertRaises(Rerun):
            self.attach()
        key = self.db.metadata["paper1"]["storage_key"]
        self.assertNotEqual(key, "old.pdf")
        self.assertEqual(self.storage.files[key], b"new")
        self.assertEqual(self.state.project_version(), self.db.version)
        self.assertEqual(self.state.snapshot(), self.db.data)
        self.assertEqual(self.state.pending_pdf_deletions(), [{"paper_uid": None, "storage_key": "old.pdf"}])

    def test_shared_cleanup_notice_persists_and_retries_only_failed_keys(self):
        self.st.session_state["_uncommitted_pdf_cleanup"] = ["first.pdf", "second.pdf"]
        calls = []

        def retry(user, keys):
            calls.append((user, list(keys)))
            return ["second.pdf"] if len(calls) == 1 else []

        cleanup = Mock(side_effect=retry)
        core = ModuleType("core")
        core.auth = self.auth
        workflow = ModuleType("features.workflow")
        workflow.documents = SimpleNamespace(cleanup_keys=cleanup)
        notice = load_functions("core/ui.py", {"st": self.st}, {"pending_file_cleanup"})["pending_file_cleanup"]
        with patch.dict(sys.modules, {"core": core, "features.workflow": workflow}):
            notice()
            notice()
            self.assertEqual(self.st.warning.call_count, 2)
            cleanup.assert_not_called()
            self.st.clicked = "Retry pending file cleanup"
            with self.assertRaises(Rerun):
                notice()
            self.assertEqual(self.st.session_state["_uncommitted_pdf_cleanup"], ["second.pdf"])
            with self.assertRaises(Rerun):
                notice()
        self.assertEqual(cleanup.call_count, 2)
        self.assertEqual(calls, [
            ("reviewer@example.test", ["first.pdf", "second.pdf"]),
            ("reviewer@example.test", ["second.pdf"]),
        ])
        self.assertEqual(self.st.session_state["_uncommitted_pdf_cleanup"], [])


if __name__ == "__main__":
    unittest.main()
