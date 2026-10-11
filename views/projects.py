import streamlit as st

from core import auth, db, fulltext_storage, ui
from features.workflow import state

ui.inject_base_css()
auth.sidebar_user()
ui.app_header("My Projects", "Create a project, open one to keep working, or delete it.")

user = auth.current_user()
state.require_saved_results()
ui.pending_file_cleanup()


def _stop_on_db_error(exc: db.DatabaseError) -> None:
    st.error(ui.escape_markdown(exc))
    st.stop()

with st.container(border=True):
    st.subheader("New project")
    name = st.text_input(
        "Project name *",
        placeholder="e.g., Flood social-media + remote-sensing review",
        help="A name is required before the project can be created.",
    )
    mode_keys = [state.MODE_PRISMA, state.MODE_DIRECT]
    mode_label = st.radio(
        "Workflow",
        [state.MODE_LABELS[m] for m in mode_keys],
        captions=[state.MODE_DESCRIPTIONS[m] for m in mode_keys],
    )
    mode = mode_keys[[state.MODE_LABELS[m] for m in mode_keys].index(mode_label)]
    next_step = (
        "After creation, you will go to Abstract Screening to upload the CSV."
        if mode == state.MODE_PRISMA
        else "After creation, you will go to Full-text Screening to upload the PDFs."
    )
    st.info(next_step)
    st.caption("The workflow mode is fixed once the project is created.")
    if st.button("Create and continue →", type="primary"):
        if not name.strip():
            st.error(
                "Enter a project name first. The CSV or PDF upload appears on the next page."
            )
        else:
            data = state.new_project_data(mode)
            try:
                pid = db.create_project(user, name.strip(), data)
            except db.DatabaseError as exc:
                _stop_on_db_error(exc)
            state.set_active(pid, name.strip())
            state.load_into_session(data, version=1, project_id=pid, source=([], []))
            st.session_state["_go_workflow"] = True
            st.rerun()

with st.container(border=True):
    st.subheader("Your projects")
    try:
        rows = db.list_projects(user)
    except db.DatabaseError as exc:
        _stop_on_db_error(exc)
    if not rows:
        st.caption("No projects yet — create one above.")
    for row in rows:
        pid, name = row["id"], row["name"]
        is_active = pid == state.active_id()

        info_col, open_col, ren_col, del_col = st.columns([6, 2, 2, 2])
        info_col.markdown(f"**{ui.escape_markdown(name)}**" + ("  ·  *open*" if is_active else ""))
        info_col.caption(f"updated {str(row['updated_at'])[:16].replace('T', ' ')}")

        if open_col.button("Open", key=f"open_{pid}", width="stretch"):
            try:
                if not is_active or not state.loaded():
                    state.reload_project(pid)
                elif db.project_version(user, pid) != state.project_version():
                    # Another session saved newer data; the unsaved question setup is kept.
                    state.reload_project(pid, keep_setup_draft=True)
                state.set_active(pid, name)
            except db.DatabaseError as exc:
                _stop_on_db_error(exc)
            except state.UnsupportedSchemaError as exc:
                st.error(ui.escape_markdown(exc))
                st.stop()
            st.session_state["_go_workflow"] = True
            st.rerun()
        if ren_col.button("Rename", key=f"ren_{pid}", width="stretch"):
            st.session_state[f"renaming_{pid}"] = True
        if del_col.button("Delete", key=f"del_{pid}", width="stretch"):
            st.session_state[f"confirm_del_{pid}"] = True

        if st.session_state.get(f"renaming_{pid}"):
            new_name = st.text_input("New name", value=name, key=f"rn_input_{pid}")
            if st.button("Save name", key=f"rn_save_{pid}"):
                try:
                    db.rename_project(user, pid, new_name.strip() or name)
                except db.DatabaseError as exc:
                    _stop_on_db_error(exc)
                if is_active:
                    state.set_active(pid, new_name.strip() or name)
                st.session_state.pop(f"renaming_{pid}", None)
                st.rerun()

        if st.session_state.get(f"confirm_del_{pid}"):
            st.warning(f"Delete “{ui.escape_markdown(name)}”? This permanently removes its saved work.")
            yes_col, no_col, _ = st.columns([2, 2, 6])
            if yes_col.button("Yes, delete", key=f"yesdel_{pid}", type="primary"):
                try:
                    cleanup_keys = db.delete_project(user, pid)
                except db.DatabaseError as exc:
                    _stop_on_db_error(exc)
                try:
                    fulltext_storage.delete_many(cleanup_keys)
                except fulltext_storage.FulltextStorageError as exc:
                    st.session_state.setdefault("_uncommitted_pdf_cleanup", []).extend(exc.failed_keys)
                if is_active:
                    state.set_active(None, None)
                    state.clear_session()
                st.session_state.pop(f"confirm_del_{pid}", None)
                st.rerun()
            if no_col.button("Cancel", key=f"nodel_{pid}"):
                st.session_state.pop(f"confirm_del_{pid}", None)
                st.rerun()
