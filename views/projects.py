import streamlit as st

from core import auth, db, ui
from features.screening import state

ui.inject_base_css()
auth.sidebar_user()
ui.app_header("My Projects", "Create a project, open one to keep working, or delete it.")

user = auth.current_user()


def _stop_on_db_error(exc: db.DatabaseError) -> None:
    st.error(str(exc))
    st.stop()

with st.container(border=True):
    st.subheader("New project")
    name = st.text_input("Project name", placeholder="e.g., Flood social-media + remote-sensing review")
    if st.button("Create", type="primary", disabled=not name.strip()):
        try:
            pid = db.create_project(user, name.strip(), {})
        except db.DatabaseError as exc:
            _stop_on_db_error(exc)
        state.set_active(pid, name.strip())
        state.restore({})
        st.session_state["_go_screening"] = True
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
        info_col.markdown(f"**{name}**" + ("  ·  *open*" if is_active else ""))
        info_col.caption(f"updated {str(row['updated_at'])[:16].replace('T', ' ')}")

        if open_col.button("Open", key=f"open_{pid}", use_container_width=True):
            state.set_active(pid, name)
            try:
                state.restore(db.load_project(user, pid))
            except db.DatabaseError as exc:
                _stop_on_db_error(exc)
            st.session_state["_go_screening"] = True
            st.rerun()
        if ren_col.button("Rename", key=f"ren_{pid}", use_container_width=True):
            st.session_state[f"renaming_{pid}"] = True
        if del_col.button("Delete", key=f"del_{pid}", use_container_width=True):
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
            st.warning(f"Delete “{name}”? This permanently removes its saved work.")
            yes_col, no_col, _ = st.columns([2, 2, 6])
            if yes_col.button("Yes, delete", key=f"yesdel_{pid}", type="primary"):
                try:
                    db.delete_project(user, pid)
                except db.DatabaseError as exc:
                    _stop_on_db_error(exc)
                if is_active:
                    state.set_active(None, None)
                    state.restore({})
                st.session_state.pop(f"confirm_del_{pid}", None)
                st.rerun()
            if no_col.button("Cancel", key=f"nodel_{pid}"):
                st.session_state.pop(f"confirm_del_{pid}", None)
                st.rerun()
