from __future__ import annotations

import os

import streamlit as st

DEV_EMAIL = "dev@local"


class AccessConfigError(RuntimeError):
    """The access restriction exists but cannot be read. Fail closed."""


def _has_secret(section: str) -> bool:
    try:
        return section in st.secrets
    except Exception:
        return False


def _auth_configured() -> bool:
    if os.environ.get("AIREVIEW_DEV") == "1":
        return False
    return _has_secret("auth")


def _production_config_present() -> bool:
    for section, key in (("database", "url"), ("storage", "url")):
        try:
            if str(st.secrets[section][key]).strip():
                return True
        except Exception:
            continue
    return False


def dev_opt_in() -> bool:
    """Shared development opt-in for auth, database, and storage."""
    if os.environ.get("AIREVIEW_DEV") == "1":
        return True
    try:
        return st.secrets["dev"]["allow_no_auth"] is True
    except Exception:
        return False


def _dev_mode_allowed() -> bool:
    """Allow explicit dev mode; secret opt-in cannot override production settings."""
    if os.environ.get("AIREVIEW_DEV") == "1":
        return True
    return dev_opt_in() and not _production_config_present()


def _allowlist() -> list[str] | None:
    """Return allowed addresses, or None if absent; reject malformed restrictions."""
    if not _has_secret("access"):
        return None
    try:
        emails = st.secrets["access"]["allowed_emails"]
    except Exception as exc:
        raise AccessConfigError(
            "[access] is configured but allowed_emails could not be read."
        ) from exc
    if isinstance(emails, (str, bytes)) or not isinstance(emails, (list, tuple)):
        raise AccessConfigError(
            "[access].allowed_emails must be a list of addresses, "
            'for example allowed_emails = ["name@example.com"].'
        )
    cleaned = [str(email).strip().lower() for email in emails if str(email).strip()]
    if not cleaned:
        raise AccessConfigError("[access].allowed_emails is empty, so nobody could sign in.")
    return cleaned


def current_user() -> str:
    if not _auth_configured():
        return DEV_EMAIL
    return st.user.email if getattr(st.user, "is_logged_in", False) else DEV_EMAIL


def require_login() -> str:
    if not _auth_configured():
        if _dev_mode_allowed():
            return DEV_EMAIL
        st.error(
            "Login is not configured. Add an [auth] section to secrets. "
            "(The no-login development mode needs AIREVIEW_DEV=1, or "
            "[dev] allow_no_auth = true with no database or storage configured.)"
        )
        st.stop()

    if not getattr(st.user, "is_logged_in", False):
        st.title("AI Literature Review Platform")
        st.write("Please sign in to continue.")
        st.button("Sign in with Google", type="primary", on_click=st.login)
        st.stop()

    email = st.user.email
    try:
        allow = _allowlist()
    except AccessConfigError as exc:
        st.error(f"{exc} Nobody can sign in until this is fixed.")
        st.button("Log out", on_click=st.logout)
        st.stop()
        return DEV_EMAIL
    if allow is not None and email.lower() not in allow:
        st.error("Your account is not authorized to use this app.")
        st.button("Log out", on_click=st.logout)
        st.stop()
    return email


def sidebar_user() -> None:
    with st.sidebar:
        st.caption(f"Signed in as {current_user()}")
        if _auth_configured():
            st.button("Log out", on_click=st.logout, width="stretch")
        else:
            st.caption("local dev mode (no login configured)")
        st.divider()
