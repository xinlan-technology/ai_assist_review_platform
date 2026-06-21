from __future__ import annotations

import os

import streamlit as st

DEV_EMAIL = "dev@local"


def _has_secret(section: str) -> bool:
    try:
        return section in st.secrets
    except Exception:
        return False


def _auth_configured() -> bool:
    return _has_secret("auth")


def _dev_mode_allowed() -> bool:
    if os.environ.get("AIREVIEW_DEV") == "1":
        return True
    try:
        return bool(st.secrets["dev"]["allow_no_auth"])
    except Exception:
        return False


def _allowlist() -> list[str] | None:
    try:
        emails = st.secrets["access"]["allowed_emails"]
        return [str(e).lower() for e in emails]
    except Exception:
        return None


def current_user() -> str:
    if not _auth_configured():
        return DEV_EMAIL
    return st.user.email if getattr(st.user, "is_logged_in", False) else DEV_EMAIL


def require_login() -> str:
    if not _auth_configured():
        if _dev_mode_allowed():
            return DEV_EMAIL
        st.error(
            "Login is not configured. Add an [auth] section to secrets "
            "(or set AIREVIEW_DEV=1 for local development)."
        )
        st.stop()

    if not getattr(st.user, "is_logged_in", False):
        st.title("AI Literature Review Platform")
        st.write("Please sign in to continue.")
        st.button("Sign in with Google", type="primary", on_click=st.login)
        st.stop()

    email = st.user.email
    allow = _allowlist()
    if allow is not None and email.lower() not in allow:
        st.error("Your account is not authorized to use this app.")
        st.button("Log out", on_click=st.logout)
        st.stop()
    return email


def sidebar_user() -> None:
    with st.sidebar:
        st.caption(f"Signed in as {current_user()}")
        if _auth_configured():
            st.button("Log out", on_click=st.logout, use_container_width=True)
        else:
            st.caption("local dev mode (no login configured)")
        st.divider()
