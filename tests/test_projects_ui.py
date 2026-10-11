"""Project navigation preserves drafts and only activates successful loads."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core import auth, db, fulltext_storage, ui
from features.workflow import state


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def projects_ui(monkeypatch):
    monkeypatch.setattr(auth, "sidebar_user", lambda: None)
    monkeypatch.setattr(auth, "current_user", lambda: "reviewer@example.invalid")
    projects = {
        "first": state.new_project_data(state.MODE_PRISMA),
        "second": state.new_project_data(state.MODE_DIRECT),
    }
    monkeypatch.setattr(db, "list_projects", lambda user: [
        {"id": pid, "name": pid.title(), "updated_at": "2026-01-01"}
        for pid in projects
    ])
    load = Mock(side_effect=lambda user, pid: (deepcopy(projects[pid]), 3, ([], [])))
    monkeypatch.setattr(db, "load_project_bundle", load)
    # The open project is current unless a test says another session saved it.
    monkeypatch.setattr(db, "project_version", lambda user, pid: 0)

    def create(loaded=True):
        app = AppTest.from_file(str(ROOT / "views/projects.py"), default_timeout=10)
        app.session_state["active_project_id"] = "first"
        app.session_state["active_project_name"] = "First"
        if loaded:
            project = deepcopy(projects["first"])
            app.session_state["project_store"] = {
                "mode": project["mode"], "config": project["config"], "papers": [],
                "original_df": pd.DataFrame(), "cursors": {},
            }
        app.session_state["extraction:first:setup_draft"] = {
            "instructions": "Unsaved instructions", "questions": [],
        }
        return app.run()

    return create, load


def test_opening_active_project_preserves_unsaved_setup(projects_ui):
    create, load = projects_ui
    app = create()
    assert not app.exception
    draft = deepcopy(app.session_state["extraction:first:setup_draft"])

    app.button(key="open_first").click().run()

    assert not app.exception
    assert app.session_state["extraction:first:setup_draft"] == draft
    assert app.session_state["_go_workflow"] is True
    load.assert_not_called()


@pytest.mark.parametrize("loaded,target", [(True, "second"), (False, "first")])
def test_open_loads_another_or_uninitialized_project(projects_ui, loaded, target):
    create, load = projects_ui
    app = create(loaded)

    app.button(key=f"open_{target}").click().run()

    assert not app.exception
    load.assert_called_once_with("reviewer@example.invalid", target)
    assert app.session_state["active_project_id"] == target
    assert app.session_state["active_project_name"] == target.title()
    assert app.session_state["_go_workflow"] is True
    assert app.session_state["project_store"]["mode"] == (
        state.MODE_DIRECT if target == "second" else state.MODE_PRISMA
    )


def test_failed_project_open_preserves_active_project_and_draft(projects_ui):
    create, load = projects_ui
    app = create()
    before = deepcopy(app.session_state["project_store"]["config"])
    draft = deepcopy(app.session_state["extraction:first:setup_draft"])
    load.side_effect = db.DatabaseError("Offline read failure")

    app.button(key="open_second").click().run()

    assert not app.exception
    assert app.session_state["active_project_id"] == "first"
    assert app.session_state["project_store"]["config"] == before
    assert app.session_state["extraction:first:setup_draft"] == draft
    assert any("Offline read failure" in item.value for item in app.error)
    assert "_go_workflow" not in app.session_state


def test_project_names_are_literal_in_list_and_delete_confirmation(projects_ui, monkeypatch):
    create, _ = projects_ui
    name = '![project](https://example.invalid/pixel) <img src="https://example.invalid/image">'
    monkeypatch.setattr(db, "list_projects", lambda user: [
        {"id": "first", "name": name, "updated_at": "2026-01-01"},
    ])
    app = create()

    assert any(ui.escape_markdown(name) in item.value for item in app.markdown)
    app.button(key="del_first").click().run()

    assert not app.exception
    assert any(ui.escape_markdown(name) in item.value for item in app.warning)
    assert all(name not in item.value for item in [*app.markdown, *app.warning])


def test_project_load_errors_cannot_render_images(projects_ui):
    create, load = projects_ui
    app = create()
    message = "![error](https://example.invalid/pixel)"
    load.side_effect = db.DatabaseError(message)

    app.button(key="open_second").click().run()

    assert not app.exception
    assert any(item.value == ui.escape_markdown(message) for item in app.error)


def test_delete_cleans_only_keys_returned_by_the_locked_transaction(projects_ui, monkeypatch):
    create, _ = projects_ui
    remove = Mock(return_value=["newer.pdf", "pending.pdf"])
    cleanup = Mock()
    monkeypatch.setattr(db, "delete_project", remove)
    monkeypatch.setattr(fulltext_storage, "delete_many", cleanup)
    stale_read = Mock(side_effect=AssertionError("Do not read cleanup keys before the deletion lock."))
    monkeypatch.setattr(db, "load_fulltexts", stale_read)
    monkeypatch.setattr(db, "load_project_versioned", stale_read)
    app = create()
    app.button(key="del_first").click().run()

    app.button(key="yesdel_first").click().run()

    assert not app.exception
    remove.assert_called_once_with("reviewer@example.invalid", "first")
    cleanup.assert_called_once_with(["newer.pdf", "pending.pdf"])
    stale_read.assert_not_called()


def test_opening_the_active_project_refreshes_newer_saved_data_and_keeps_the_setup_draft(projects_ui, monkeypatch):
    create, load = projects_ui
    monkeypatch.setattr(db, "project_version", lambda user, pid: 3)
    app = create()
    app.session_state["project_store"][state._VERSION_KEY] = 2
    app.session_state["extraction:first:open_form"] = {"uid": "paper1"}
    draft = deepcopy(app.session_state["extraction:first:setup_draft"])

    app.button(key="open_first").click().run()

    assert not app.exception
    load.assert_called_once_with("reviewer@example.invalid", "first")
    assert app.session_state["project_store"][state._VERSION_KEY] == 3
    assert app.session_state["extraction:first:setup_draft"] == draft
    assert "extraction:first:open_form" not in app.session_state


def test_opening_a_project_deleted_elsewhere_reports_it_and_keeps_the_session(projects_ui, monkeypatch):
    create, load = projects_ui
    monkeypatch.setattr(db, "project_version", lambda user, pid: None)
    load.side_effect = lambda user, pid: ({}, 0, ([], []))
    app = create()

    app.button(key="open_first").click().run()

    assert not app.exception
    assert any("no longer exists" in error.value for error in app.error)
    assert "_go_workflow" not in app.session_state
