from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    func,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    inspect as sa_inspect,
    insert,
    select,
    text as sql_text,
    update,
)
from sqlalchemy.exc import SQLAlchemyError

from core import auth


class DatabaseError(RuntimeError):
    pass


class ProjectConflictError(DatabaseError):
    """Another session saved this project since it was loaded here."""


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
    # Reject stale writes using a monotonically increasing row version.
    Column("version", Integer, nullable=False, default=1),
)

# Keep authoritative PDF metadata separate from frequently saved review state.
fulltexts = Table(
    "fulltexts",
    _metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("user_email", String, nullable=False, index=True),
    Column("paper_uid", String, nullable=False),
    Column("filename", String),
    Column("storage_key", String),
    Column("sha256", String),
    Column("file_size", Integer),
    Column("page_count", Integer),
    Column("status", String, nullable=False),  # ok | failed
    Column("error", Text),
    Column("text", Text),  # unused by native-PDF screening; kept for compatibility
    Column("created_at", String, nullable=False),
    UniqueConstraint("project_id", "paper_uid", name="uq_fulltexts_project_paper"),
)

# Store imported rows separately to avoid rewriting them for each decision.
project_sources = Table(
    "project_sources",
    _metadata,
    Column("project_id", String, primary_key=True),
    Column("user_email", String, nullable=False, index=True),
    Column("columns", JSON, nullable=False),
    Column("records", JSON, nullable=False),
    Column("created_at", String, nullable=False),
)

_engine = None

_FULLTEXT_ADDED_COLUMNS = {
    "storage_key": "VARCHAR",
    "file_size": "INTEGER",
    "page_count": "INTEGER",
    "error": "TEXT",
}


def _upgrade_projects_schema(engine) -> None:
    """Add the optimistic-locking column to a table created without it."""
    inspector = sa_inspect(engine)
    if not inspector.has_table("projects"):
        return
    existing = {column["name"] for column in inspector.get_columns("projects")}
    if "version" in existing:
        return
    with engine.begin() as conn:
        conn.execute(sql_text("ALTER TABLE projects ADD COLUMN version INTEGER"))
        conn.execute(sql_text("UPDATE projects SET version = 1 WHERE version IS NULL"))


def _upgrade_fulltexts_schema(engine) -> None:
    """Add nullable columns to legacy tables; create_all() does not alter them."""
    inspector = sa_inspect(engine)
    if not inspector.has_table("fulltexts"):
        return
    existing = {column["name"] for column in inspector.get_columns("fulltexts")}
    missing = [
        (name, sql_type)
        for name, sql_type in _FULLTEXT_ADDED_COLUMNS.items()
        if name not in existing
    ]
    if not missing:
        return
    with engine.begin() as conn:
        for name, sql_type in missing:
            conn.execute(
                sql_text(f"ALTER TABLE fulltexts ADD COLUMN {name} {sql_type}")
            )


def _conn_string() -> str:
    # Explicit dev mode overrides deployment secrets with isolated local data.
    if os.environ.get("AIREVIEW_DEV") == "1":
        local_dir = Path(__file__).resolve().parent.parent / ".local"
        local_dir.mkdir(exist_ok=True)
        return f"sqlite:///{local_dir / 'projects.db'}"
    try:
        url = st.secrets["database"]["url"]
        if url:
            return url
    except Exception:
        pass
    if not auth.dev_opt_in():
        raise DatabaseError(_DB_MISSING_MESSAGE)
    local_dir = Path(__file__).resolve().parent.parent / ".local"
    local_dir.mkdir(exist_ok=True)
    return f"sqlite:///{local_dir / 'projects.db'}"


def _get_engine():
    global _engine
    if _engine is None:
        url = _conn_string()
        kwargs = (
            {"connect_args": {"check_same_thread": False}}
            if url.startswith("sqlite")
            # Refresh connections closed by Supabase while idle.
            else {"pool_pre_ping": True, "pool_recycle": 1800}
        )
        _engine = create_engine(url, **kwargs)
        try:
            _metadata.create_all(_engine)
            _upgrade_projects_schema(_engine)
            _upgrade_fulltexts_schema(_engine)
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
        id=pid, user_email=user_email, name=name, data=data,
        created_at=now, updated_at=now, version=1,
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()
    return pid


def load_project(user_email: str, pid: str) -> dict:
    return load_project_versioned(user_email, pid)[0]


def load_project_versioned(user_email: str, pid: str) -> tuple[dict, int]:
    """Project data plus the version it was read at, for conflict detection."""
    stmt = select(projects.c.data, projects.c.version).where(
        projects.c.id == pid, projects.c.user_email == user_email
    )
    try:
        with _get_engine().connect() as conn:
            row = conn.execute(stmt).first()
    except SQLAlchemyError:
        _raise_database_error()
    if not row:
        return {}, 0
    return dict(row._mapping["data"]), int(row._mapping["version"] or 1)


def load_project_bundle(user_email: str, pid: str) -> tuple[dict, int, tuple[list, list]]:
    """Read the document, version, and imported rows from one database snapshot."""
    stmt = select(
        projects.c.data, projects.c.version,
        project_sources.c.columns, project_sources.c.records,
    ).select_from(projects.outerjoin(
        project_sources,
        (project_sources.c.project_id == projects.c.id)
        & (project_sources.c.user_email == projects.c.user_email),
    )).where(projects.c.id == pid, projects.c.user_email == user_email)
    try:
        with _get_engine().connect() as conn:
            row = conn.execute(stmt).first()
    except SQLAlchemyError:
        _raise_database_error()
    if row is None:
        return {}, 0, ([], [])
    values = row._mapping
    return (
        dict(values["data"]), int(values["version"] or 1),
        (list(values["columns"] or []), list(values["records"] or [])),
    )


def save_project(user_email: str, pid: str, data: dict,
                 expected_version: int | None = None,
                 source: tuple[list, list] | None = None,
                 fulltext: dict | None = None) -> int:
    """Atomically save review data and optional source/PDF metadata; return version.

    ``expected_version=None`` skips the version guard.
    """
    conditions = [projects.c.id == pid, projects.c.user_email == user_email]
    if expected_version is not None:
        conditions.append(func.coalesce(projects.c.version, 1) == expected_version)
    stmt = (
        update(projects)
        .where(*conditions)
        .values(data=data, updated_at=_now(),
                version=func.coalesce(projects.c.version, 1) + 1)
    )
    try:
        with _get_engine().begin() as conn:
            result = conn.execute(stmt)
            if result.rowcount != 1:
                if expected_version is not None:
                    current = conn.execute(
                        select(projects.c.version).where(
                            projects.c.id == pid, projects.c.user_email == user_email
                        )
                    ).first()
                    if current is not None:
                        raise ProjectConflictError(
                            "This project was changed in another tab or window, so your "
                            "changes were not saved. Reload the page to continue from the "
                            "saved version."
                        )
                raise DatabaseError("Project no longer exists or is not accessible. Changes were not saved.")
            if source is not None:
                columns, records = source
                conn.execute(
                    delete(project_sources).where(
                        project_sources.c.project_id == pid,
                        project_sources.c.user_email == user_email,
                    )
                )
                conn.execute(
                    insert(project_sources).values(
                        project_id=pid, user_email=user_email,
                        columns=list(columns), records=list(records),
                        created_at=_now(),
                    )
                )
            if fulltext is not None:
                _write_fulltext(conn, user_email, pid, **fulltext)
            new_version = conn.execute(
                select(projects.c.version).where(
                    projects.c.id == pid, projects.c.user_email == user_email
                )
            ).scalar()
    except SQLAlchemyError:
        _raise_database_error()
    return int(new_version or 0)


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


def save_project_source(user_email: str, project_id: str,
                        columns: list, records: list) -> None:
    """Store imported rows outside the project document."""
    try:
        with _get_engine().begin() as conn:
            conn.execute(
                delete(project_sources).where(
                    project_sources.c.project_id == project_id,
                    project_sources.c.user_email == user_email,
                )
            )
            conn.execute(
                insert(project_sources).values(
                    project_id=project_id,
                    user_email=user_email,
                    columns=list(columns),
                    records=list(records),
                    created_at=_now(),
                )
            )
    except SQLAlchemyError:
        _raise_database_error()


def load_project_source(user_email: str, project_id: str) -> tuple[list, list]:
    stmt = select(project_sources.c.columns, project_sources.c.records).where(
        project_sources.c.project_id == project_id,
        project_sources.c.user_email == user_email,
    )
    try:
        with _get_engine().connect() as conn:
            row = conn.execute(stmt).first()
    except SQLAlchemyError:
        _raise_database_error()
    if not row:
        return [], []
    return list(row._mapping["columns"] or []), list(row._mapping["records"] or [])


def delete_project(user_email: str, pid: str) -> None:
    try:
        with _get_engine().begin() as conn:
            conn.execute(
                delete(project_sources).where(
                    project_sources.c.project_id == pid,
                    project_sources.c.user_email == user_email,
                )
            )
            conn.execute(
                delete(fulltexts).where(
                    fulltexts.c.project_id == pid, fulltexts.c.user_email == user_email
                )
            )
            conn.execute(
                delete(projects).where(
                    projects.c.id == pid, projects.c.user_email == user_email
                )
            )
    except SQLAlchemyError:
        _raise_database_error()


def upsert_fulltext(
    user_email: str,
    project_id: str,
    paper_uid: str,
    filename: str,
    sha256: str,
    status: str,
    text: str | None = None,
    storage_key: str | None = None,
    file_size: int | None = None,
    page_count: int | None = None,
    error: str | None = None,
) -> None:
    try:
        with _get_engine().begin() as conn:
            _write_fulltext(
                conn, user_email, project_id, paper_uid, filename, sha256, status,
                text=text, storage_key=storage_key, file_size=file_size,
                page_count=page_count, error=error,
            )
    except SQLAlchemyError:
        _raise_database_error()


def _write_fulltext(conn, user_email: str, project_id: str, paper_uid: str,
                    filename: str, sha256: str, status: str,
                    text: str | None = None, storage_key: str | None = None,
                    file_size: int | None = None, page_count: int | None = None,
                    error: str | None = None) -> None:
    conn.execute(delete(fulltexts).where(
        fulltexts.c.project_id == project_id,
        fulltexts.c.user_email == user_email,
        fulltexts.c.paper_uid == paper_uid,
    ))
    conn.execute(insert(fulltexts).values(
        id=uuid.uuid4().hex, project_id=project_id, user_email=user_email,
        paper_uid=paper_uid, filename=filename, storage_key=storage_key,
        sha256=sha256, file_size=file_size, page_count=page_count,
        status=status, error=error, text=text, created_at=_now(),
    ))


def delete_fulltext(user_email: str, project_id: str, paper_uid: str) -> None:
    stmt = delete(fulltexts).where(
        fulltexts.c.project_id == project_id,
        fulltexts.c.user_email == user_email,
        fulltexts.c.paper_uid == paper_uid,
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()


def delete_project_fulltexts(user_email: str, project_id: str) -> None:
    stmt = delete(fulltexts).where(
        fulltexts.c.project_id == project_id,
        fulltexts.c.user_email == user_email,
    )
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()


def load_fulltexts(user_email: str, project_id: str) -> dict[str, dict]:
    """All full texts for a project, keyed by paper_uid."""
    stmt = select(
        fulltexts.c.paper_uid,
        fulltexts.c.filename,
        fulltexts.c.storage_key,
        fulltexts.c.sha256,
        fulltexts.c.file_size,
        fulltexts.c.page_count,
        fulltexts.c.status,
        fulltexts.c.error,
        fulltexts.c.text,
    ).where(
        fulltexts.c.project_id == project_id, fulltexts.c.user_email == user_email
    )
    try:
        with _get_engine().connect() as conn:
            return {
                row.paper_uid: {
                    "filename": row.filename,
                    "storage_key": row.storage_key,
                    "sha256": row.sha256,
                    "file_size": row.file_size,
                    "page_count": row.page_count,
                    "status": row.status,
                    "error": row.error,
                    "text": row.text,
                }
                for row in conn.execute(stmt)
            }
    except SQLAlchemyError:
        _raise_database_error()
