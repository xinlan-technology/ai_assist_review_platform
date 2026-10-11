"""Real transactions with Streamlit's interruptible session wrapper."""
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from streamlit.runtime.scriptrunner_utils.exceptions import RerunException, StopException
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.runtime.state import session_state_proxy
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.state.session_state import SessionState

from core import db
from features.workflow import state


@pytest.fixture
def project(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    db._metadata.create_all(engine)
    monkeypatch.setattr(db, "_get_engine", lambda: engine)
    control = {"interrupt": None}
    raw = SessionState()

    def yield_callback():
        interruption = control.pop("interrupt", None)
        if interruption is not None:
            raise interruption
        predicate = control.get("interrupt_when")
        if predicate is not None and predicate(raw):
            control.pop("interrupt_when")
            raise control.pop("conditional_interruption")

    safe = SafeSessionState(raw, yield_callback)
    monkeypatch.setattr(session_state_proxy, "get_session_state", lambda: safe)
    errors = []
    monkeypatch.setattr(state, "st", SimpleNamespace(
        session_state=session_state_proxy.SessionStateProxy(), error=errors.append,
    ))
    user = "reviewer@example.com"
    monkeypatch.setattr(state.auth, "current_user", lambda: user)
    data = state.new_project_data(state.MODE_PRISMA)
    pid = db.create_project(user, "Study", data)
    state.set_active(pid, "Study")
    state.load_into_session(data, version=1, source=([], []))
    yield SimpleNamespace(user=user, pid=pid, control=control, errors=errors)
    engine.dispose()


@pytest.mark.parametrize("interruption", [StopException(), RerunException(RerunData())])
def test_committed_save_is_acknowledged_before_the_next_streamlit_yield(project, monkeypatch,
                                                                      interruption):
    frame = pd.DataFrame([{"Title": "Study"}])
    state._store().update(papers=[state.new_paper("", "Study", "Abstract")], original_df=frame,
                          cursors={})
    state._store()[state._SOURCE_KEY] = True
    save = db.save_project

    def commit_then_interrupt(*args, **kwargs):
        version = save(*args, **kwargs)
        project.control["interrupt"] = interruption
        return version

    monkeypatch.setattr(db, "save_project", commit_then_interrupt)
    assert state.save_active()
    with pytest.raises(type(interruption)):
        state.project_version()
    assert state.project_version() == 2
    assert state._pending_source() is None
    assert db.load_project_source(project.user, project.pid) == (
        ["Title"], [{"Title": "Study"}],
    )
    monkeypatch.setattr(db, "save_project", save)
    assert state.save_active()
    assert state.project_version() == 3
    assert not project.errors


def test_recovered_session_still_refuses_a_genuine_concurrent_change(project, monkeypatch):
    save = db.save_project

    def commit_then_interrupt(*args, **kwargs):
        version = save(*args, **kwargs)
        project.control["interrupt"] = StopException()
        return version

    monkeypatch.setattr(db, "save_project", commit_then_interrupt)
    assert state.save_active()
    with pytest.raises(StopException):
        state.snapshot()
    monkeypatch.setattr(db, "save_project", save)
    other = state.new_project_data(state.MODE_PRISMA)
    other["config"]["abstract_criteria"] = "Another window's criteria"
    db.save_project(project.user, project.pid, other, expected_version=2)
    state.config()["abstract_criteria"] = "Stale local criteria"
    assert not state.save_active()
    assert state.project_version() == 2
    assert db.load_project(project.user, project.pid) == other
    assert state._store()[state._CONFLICT_KEY]


def test_detached_commit_changes_session_only_after_a_successful_transaction(project):
    prepared = state.prepare_save()
    prepared.data["config"]["abstract_criteria"] = "Detached criteria"
    assert state.config()["abstract_criteria"] != "Detached criteria"
    source = (["Title"], [{"Title": "New study"}])
    assert prepared.commit(prepared.data, source=source) == 2
    assert state.config()["abstract_criteria"] == "Detached criteria"
    assert state._store()["original_df"].to_dict("records") == source[1]
    assert db.load_project_bundle(project.user, project.pid) == (prepared.data, 2, source)


def test_failed_detached_commit_leaves_live_session_untouched(project):
    prepared = state.prepare_save()
    before = state.snapshot()
    prepared.data["papers"] = [state.new_paper("", "New study", "")]
    invalid = {"paper_uid": "new", "filename": "study.pdf", "sha256": "digest", "status": None}
    with pytest.raises(db.DatabaseError):
        prepared.commit(prepared.data, fulltext=invalid,
                        source=(["Title"], [{"Title": "New study"}]))
    assert state.snapshot() == before
    assert state.project_version() == 1
    assert db.load_project_source(project.user, project.pid) == ([], [])


@pytest.mark.parametrize("when", ["during_change", "before_save", "after_commit"])
def test_interrupted_edit_rolls_back_only_uncommitted_changes(project, monkeypatch, when):
    before = state.snapshot()
    before["config"] = dict(before["config"])
    saved_before = db.load_project(project.user, project.pid)
    save_active = state.save_active

    def changed_save():
        saved = save_active()
        if when == "after_commit":
            project.control["interrupt"] = StopException()
            state.project_version()
        return saved

    monkeypatch.setattr(state, "save_active", changed_save)

    def change():
        state.config()["abstract_criteria"] = "Changed criteria"
        if when != "after_commit":
            project.control["interrupt"] = StopException()
        if when == "during_change":
            state.config()

    with pytest.raises(StopException):
        state.commit(change)
    saved, version = db.load_project_versioned(project.user, project.pid)
    if when == "after_commit":
        assert state.config()["abstract_criteria"] == "Changed criteria"
        assert state.project_version() == version == 2
        assert saved == state.snapshot()
    else:
        assert state.snapshot() == before
        assert saved == saved_before
        assert state.project_version() == version == 1


def test_result_recovery_is_armed_before_the_first_checkpoint_yield(project):
    paper = state.new_paper("", "Study", "Abstract")
    state.papers().append(paper)
    assert state.save_active()
    state.begin_result()
    state.set_ai_result(paper, state.STAGE_ABSTRACT, "include", "Relevant", "test", "test",
                        state.criteria_hash(""), "test")
    project.control["interrupt"] = RerunException(RerunData())
    with pytest.raises(RerunException):
        state.save_result()
    assert state.has_unsaved_results()
    saved = db.load_project(project.user, project.pid)
    assert not saved["papers"][0]["stages"].get("abstract", {}).get("ai_verdict")
    assert state.papers()[0]["stages"]["abstract"]["ai_verdict"] == "include"
    assert state.save_result()
    assert not state.has_unsaved_results()
    assert db.load_project(project.user, project.pid)["papers"][0]["stages"]["abstract"]["ai_verdict"] == "include"


def test_committed_checkpoint_clears_the_marker_before_any_interruptible_access(project, monkeypatch):
    save = db.save_project

    def commit_then_interrupt(*args, **kwargs):
        version = save(*args, **kwargs)
        project.control["interrupt"] = StopException()
        return version

    state.begin_result()
    monkeypatch.setattr(db, "save_project", commit_then_interrupt)
    # The marker is cleared with the commit, before the first interruptible access.
    with pytest.raises(StopException):
        state.save_result()
    assert not state.has_unsaved_results()
    assert state.project_version() == 2


@pytest.mark.parametrize("interruption", [StopException(), RerunException(RerunData())])
@pytest.mark.parametrize("operation", ["open", "create"])
def test_interrupted_project_switch_cannot_save_one_project_into_another(project, operation,
                                                                        interruption):
    first_before = db.load_project_bundle(project.user, project.pid)
    other_data = state.new_project_data(state.MODE_DIRECT)
    other_data["config"]["fulltext_criteria"] = "Second project's criteria"
    other = db.create_project(project.user, "Second", other_data)
    other_before = db.load_project_bundle(project.user, other)
    # Equal versions are deliberate: a version check alone cannot protect identity.
    assert first_before[1] == other_before[1] == 1
    project.control["conditional_interruption"] = interruption
    if operation == "open":
        project.control["interrupt_when"] = lambda raw: (
            raw[state._KEY].get(state._PROJECT_ID_KEY) == other
        )
    else:
        project.control["interrupt_when"] = lambda raw: raw["active_project_id"] == other

    with pytest.raises(type(interruption)):
        if operation == "open":
            state.reload_project(other)
            state.set_active(other, "Second")
        else:
            state.set_active(other, "Second")
            state.load_into_session(other_data, version=1, project_id=other, source=([], []))

    assert not state.loaded()
    with pytest.raises(db.DatabaseError, match="switch was interrupted"):
        state.prepare_save()
    assert not state.save_active()
    assert "switch was interrupted" in project.errors[-1]
    assert db.load_project_bundle(project.user, project.pid) == first_before
    assert db.load_project_bundle(project.user, other) == other_before

    # The regular My Projects / Open path recognizes the mismatch and reloads.
    target = state.active_id()
    state.reload_project(target)
    state.set_active(target, "Reopened")
    assert state.loaded()
    assert state.save_active()
    assert db.load_project(project.user, target) == state.snapshot()


def test_switching_a_legacy_session_binds_its_old_identity_first(project):
    state._store().pop(state._PROJECT_ID_KEY)
    state.set_active("new-project", "New")
    assert not state.loaded()
    assert state._store()[state._PROJECT_ID_KEY] == project.pid
    assert not state.save_active()
    assert "switch was interrupted" in project.errors[-1]


def test_interrupted_widget_reset_leaves_the_previous_project_published(project):
    other_data = state.new_project_data(state.MODE_DIRECT)
    other = db.create_project(project.user, "Second", other_data)
    key = f"extraction:{other}:setup_draft"
    state.st.session_state[key] = {"instructions": "Discarded form"}
    before = state.snapshot()
    project.control["conditional_interruption"] = StopException()
    project.control["interrupt_when"] = lambda raw: key not in raw

    with pytest.raises(StopException):
        state.reload_project(other)

    assert state.active_id() == project.pid
    assert state.loaded()
    assert state.snapshot() == before
    assert state._store()[state._PROJECT_ID_KEY] == project.pid
