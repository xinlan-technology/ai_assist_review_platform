from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from sqlalchemy import (
    JSON,
    Column,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.exc import SQLAlchemyError


class DatabaseError(RuntimeError):
    pass


_DB_ERROR_MESSAGE = (
    "Could not connect to the project database. Check the [database] URL in "
    ".streamlit/secrets.toml, especially the Supabase database password and host."
)

_DB_MISSING_MESSAGE = (
    "Project database is not configured. Add [database].url to secrets "
    "(or set AIREVIEW_DEV=1 for local SQLite development)."
)


def _raise_database_error() -> None:
    raise DatabaseError(_DB_ERROR_MESSAGE) from None


def _dev_mode_allowed() -> bool:
    if os.environ.get("AIREVIEW_DEV") == "1":
        return True
    try:
        return bool(st.secrets["dev"]["allow_no_auth"])
    except Exception:
        return False


_metadata = MetaData()
projects = Table(
    "projects",
    _metadata,
    Column("id", String, primary_key=True),
    Column("user_email", String, nullable=False, index=True),
    Column("name", String, nullable=False),
    Column("data", JSON, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

_engine = None


def _conn_string() -> str:
    try:
        url = st.secrets["database"]["url"]
        if url:
            return url
    except Exception:
        pass
    if not _dev_mode_allowed():
        raise DatabaseError(_DB_MISSING_MESSAGE)
    local_dir = Path(__file__).resolve().parent.parent / ".local"
    local_dir.mkdir(exist_ok=True)
    return f"sqlite:///{local_dir / 'projects.db'}"


def _get_engine():
    global _engine
    if _engine is None:
        url = _conn_string()
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        _engine = create_engine(url, **kwargs)
        try:
            _metadata.create_all(_engine)
        except SQLAlchemyError:
            _engine = None
            _raise_database_error()
    return _engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_projects(user_email: str) -> list[dict]:
    stmt = (
        select(projects.c.id, projects.c.name, projects.c.updated_at)
        .where(projects.c.user_email == user_email)
        .order_by(projects.c.updated_at.desc())
    )
    try:
        with _get_engine().connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt)]
    except SQLAlchemyError:
        _raise_database_error()


def create_project(user_email: str, name: str, data: dict) -> str:
    pid = uuid.uuid4().hex
    now = _now()
    stmt = insert(projects).values(
        id=pid, user_email=user_email, name=name, data=data, created_at=now, updated_at=now
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()
    return pid


def load_project(user_email: str, pid: str) -> dict:
    stmt = select(projects.c.data).where(
        projects.c.id == pid, projects.c.user_email == user_email
    )
    try:
        with _get_engine().connect() as conn:
            row = conn.execute(stmt).first()
    except SQLAlchemyError:
        _raise_database_error()
    return dict(row._mapping["data"]) if row else {}


def save_project(user_email: str, pid: str, data: dict) -> None:
    stmt = (
        update(projects)
        .where(projects.c.id == pid, projects.c.user_email == user_email)
        .values(data=data, updated_at=_now())
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()


def rename_project(user_email: str, pid: str, name: str) -> None:
    stmt = (
        update(projects)
        .where(projects.c.id == pid, projects.c.user_email == user_email)
        .values(name=name, updated_at=_now())
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()


def delete_project(user_email: str, pid: str) -> None:
    stmt = delete(projects).where(
        projects.c.id == pid, projects.c.user_email == user_email
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()
