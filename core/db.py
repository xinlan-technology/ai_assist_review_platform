from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from contextlib import nullcontext, suppress
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import streamlit as st
from sqlalchemy import (
    JSON,
    Column,
    Float,
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
from sqlalchemy.engine import Connection

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

# Attempts are durable records, independent of the current review decision.
ai_runs = Table(
    "ai_runs",
    _metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("user_email", String, nullable=False, index=True),
    Column("paper_uid", String, nullable=False, index=True),
    Column("stage", String, nullable=False),
    Column("batch_id", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("model", String, nullable=False),
    Column("config_hash", String, nullable=False),
    Column("source_hash", String, nullable=False),
    Column("prompt_version", String, nullable=False),
    Column("prompt_snapshot", JSON, nullable=False),
    Column("status", String, nullable=False),
    Column("result", JSON),
    Column("started_at", String, nullable=False),
    Column("completed_at", String),
    Column("duration_seconds", Float),
)

_engine = None
_engine_lock = Lock()

_FULLTEXT_ADDED_COLUMNS = {
    "storage_key": "VARCHAR",
    "file_size": "INTEGER",
    "page_count": "INTEGER",
    "error": "TEXT",
}


def _schema_connection(bind):
    return nullcontext(bind) if isinstance(bind, Connection) else bind.begin()


def _upgrade_projects_schema(engine) -> None:
    """Add the optimistic-locking column to a table created without it."""
    inspector = sa_inspect(engine)
    if not inspector.has_table("projects"):
        return
    existing = {column["name"] for column in inspector.get_columns("projects")}
    if "version" in existing:
        return
    with _schema_connection(engine) as conn:
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
    with _schema_connection(engine) as conn:
        for name, sql_type in missing:
            conn.execute(
                sql_text(f"ALTER TABLE fulltexts ADD COLUMN {name} {sql_type}")
            )


def _protect_app_tables(engine) -> None:
    if engine.dialect.name != "postgresql":
        return
    with _schema_connection(engine) as conn:
        for table in ("projects", "fulltexts", "project_sources", "ai_runs"):
            conn.execute(sql_text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        roles = set(conn.execute(sql_text(
            "SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated')"
        )).scalars())
        grantees = "PUBLIC" + "".join(
            f', "{role}"' for role in ("anon", "authenticated") if role in roles
        )
        # Revoke direct public grants without changing existing policies.
        conn.execute(sql_text(
            "REVOKE ALL PRIVILEGES ON TABLE projects, fulltexts, project_sources, ai_runs "
            f"FROM {grantees}"
        ))


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
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        candidate = None
        try:
            url = _conn_string()
            kwargs = (
                {"connect_args": {"check_same_thread": False}}
                if url.startswith("sqlite")
                # Refresh connections closed by Supabase while idle.
                else {"pool_pre_ping": True, "pool_recycle": 1800}
            )
            candidate = create_engine(url, **kwargs)
            # PostgreSQL publishes new tables and their protection together.
            with candidate.begin() as conn:
                _metadata.create_all(conn)
                _upgrade_projects_schema(conn)
                _upgrade_fulltexts_schema(conn)
                _protect_app_tables(conn)
        except BaseException as exc:
            if candidate is not None:
                with suppress(Exception):
                    candidate.dispose()
            if isinstance(exc, SQLAlchemyError):
                _raise_database_error()
            raise
        _engine = candidate
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
                 fulltext: dict | None = None,
                 remove_fulltexts: bool = False,
                 remove_fulltext_uids: list[str] | None = None,
                 import_legacy_runs: bool = True,
                 removed_paper_uids: list[str] | None = None) -> int:
    """Atomically save review data and optional source/PDF metadata; return version.

    ``expected_version=None`` skips the version guard. With
    ``import_legacy_runs=False`` the stored document is not read back: the
    caller names the papers it removed in ``removed_paper_uids``.
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
            previous_data = (_lock_project(conn, user_email, pid, expected_version)
                             if import_legacy_runs else None)
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
            removed_uids = set(removed_paper_uids or [])
            if import_legacy_runs:
                existing_run_ids = set(conn.execute(select(ai_runs.c.id).where(
                    ai_runs.c.project_id == pid, ai_runs.c.user_email == user_email,
                )).scalars())
                _import_legacy_ai_runs(conn, user_email, pid, previous_data, existing_run_ids)
                _import_legacy_ai_runs(conn, user_email, pid, data, existing_run_ids)
                removed_uids |= _paper_uids(previous_data) - _paper_uids(data)
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
            if remove_fulltexts or remove_fulltext_uids:
                removal = delete(fulltexts).where(
                    fulltexts.c.project_id == pid,
                    fulltexts.c.user_email == user_email,
                )
                if not remove_fulltexts:
                    removal = removal.where(fulltexts.c.paper_uid.in_(remove_fulltext_uids))
                conn.execute(removal)
            if fulltext is not None:
                _write_fulltext(conn, user_email, pid, **fulltext)
            removed_uids = sorted(removed_uids)
            for offset in range(0, len(removed_uids), 500):
                conn.execute(delete(ai_runs).where(
                    ai_runs.c.project_id == pid, ai_runs.c.user_email == user_email,
                    ai_runs.c.paper_uid.in_(removed_uids[offset:offset + 500]),
                ))
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
    try:
        with _get_engine().begin() as conn:
            _lock_project(conn, user_email, project_id, read=False)
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


def delete_project(user_email: str, pid: str) -> list[str]:
    """Delete an owned project and return its locked snapshot of PDF keys."""
    try:
        with _get_engine().begin() as conn:
            # Serialize deletion with saves and AI attempts before reading keys.
            locked = conn.execute(update(projects).where(
                projects.c.id == pid, projects.c.user_email == user_email,
            ).values(version=projects.c.version))
            if locked.rowcount != 1:
                return []
            data = conn.execute(select(projects.c.data).where(
                projects.c.id == pid, projects.c.user_email == user_email,
            )).scalar_one()
            keys = list(conn.execute(select(fulltexts.c.storage_key).where(
                fulltexts.c.project_id == pid, fulltexts.c.user_email == user_email,
            )).scalars())
            keys.extend(target.get("storage_key")
                        for target in data.get("pending_pdf_deletions") or []
                        if isinstance(target, dict))
            keys = list(dict.fromkeys(key for key in keys if isinstance(key, str) and key))
            conn.execute(delete(ai_runs).where(
                ai_runs.c.project_id == pid, ai_runs.c.user_email == user_email,
            ))
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
    return keys


def _paper_uids(data: dict) -> set[str]:
    return {
        paper["uid"] for paper in data.get("papers") or []
        if isinstance(paper, dict) and isinstance(paper.get("uid"), str)
    }


def _import_legacy_ai_runs(conn, user_email: str, pid: str, data: dict,
                           existing_run_ids: set[str]) -> None:
    """Promote available AI snapshots without interpreting human revisions as runs."""
    common = {"ai_error", "ai_error_kind", "provider", "model", "prompt_version",
              "completed_at", "started_at", "source_hash", "source_sha256"}
    screening = common | {"ai_verdict", "ai_reason", "criteria_hash"}
    extraction = common | {"ai_answers", "field_errors", "invalid_answers",
                           "spec", "spec_hash", "page_count"}
    config = data.get("config") or {}
    for paper in data.get("papers") or []:
        if not isinstance(paper, dict) or not paper.get("uid"):
            continue
        stages = paper.get("stages") or {}
        for stage in ("abstract", "fulltext", "extraction"):
            current = stages.get(stage) or {}
            if not isinstance(current, dict):
                continue
            for snapshot in [*current.get("history", []), current]:
                if (not isinstance(snapshot, dict) or snapshot.get("source_run_id")
                        or not (snapshot.get("ai_verdict") or snapshot.get("ai_error")
                                or "ai_answers" in snapshot or snapshot.get("field_errors"))):
                    continue
                fields = extraction if stage == "extraction" else screening
                result = _run_json({key: snapshot.get(key) for key in fields})
                identity = json.dumps([pid, paper["uid"], stage, result],
                                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                run_id = "legacy-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
                if run_id in existing_run_ids:
                    continue
                prompt_snapshot = {"legacy": True}
                if stage == "extraction":
                    if isinstance(result.get("spec"), dict):
                        prompt_snapshot["spec"] = result["spec"]
                    config_hash = result.get("spec_hash") or ""
                else:
                    config_hash = result.get("criteria_hash") or ""
                    criteria = config.get(f"{stage}_criteria")
                    if (isinstance(criteria, str) and config_hash
                            and hashlib.sha256(criteria.strip().encode("utf-8")).hexdigest()[:16]
                            == config_hash):
                        prompt_snapshot["criteria"] = criteria
                status = "succeeded"
                if result.get("field_errors") or result.get("ai_error_kind") == "invalid_response":
                    status = "invalid_response"
                elif result.get("ai_error"):
                    status = "call_failed"
                completed_at = result.get("completed_at") or _now()
                conn.execute(insert(ai_runs).values(
                    id=run_id, project_id=pid, user_email=user_email,
                    paper_uid=paper["uid"], stage=stage, batch_id="legacy",
                    provider=result.get("provider") or "unknown",
                    model=result.get("model") or "unknown", config_hash=config_hash,
                    source_hash=result.get("source_sha256") or result.get("source_hash") or "",
                    prompt_version=result.get("prompt_version") or "legacy-unknown",
                    prompt_snapshot=prompt_snapshot, result=result, status=status,
                    started_at=result.get("started_at") or completed_at,
                    completed_at=completed_at, duration_seconds=None,
                ))
                existing_run_ids.add(run_id)


def _lock_project(conn, user_email: str, pid: str,
                     expected_version: int | None = None, *, read: bool = True) -> dict | None:
    """Lock the owned project without changing its document or version.

    ``read=False`` confirms ownership without transferring the document.
    """
    conditions = [projects.c.id == pid, projects.c.user_email == user_email]
    if expected_version is not None:
        conditions.append(func.coalesce(projects.c.version, 1) == expected_version)
    locked = conn.execute(update(projects).where(*conditions).values(
        version=projects.c.version,
    ))
    row = conn.execute(select(projects.c.data if read else projects.c.id).where(
        projects.c.id == pid, projects.c.user_email == user_email,
    )).first()
    if row is None:
        raise DatabaseError("Project no longer exists or is not accessible.")
    if locked.rowcount != 1:
        raise ProjectConflictError(
            "This project was changed in another tab or window. Reload the page to continue."
        )
    return dict(row._mapping["data"]) if read else None


def _run_json(value: dict) -> dict:
    """Copy JSON data and reject credentials in structured snapshot fields."""
    credential_keys = {"apikey", "apikeys", "authorization", "accesstoken", "secretkey"}

    def check(item, question_ids=False):
        if isinstance(item, dict):
            for key, nested in item.items():
                normalized = str(key).lower().replace("_", "").replace("-", "")
                if normalized in credential_keys and not question_ids:
                    raise DatabaseError("AI run records must not contain credentials.")
                # These mapping keys are reviewer-defined question IDs, not settings.
                check(nested, key in {"ai_answers", "field_errors", "invalid_answers"})
        elif isinstance(item, (list, tuple)):
            for nested in item:
                check(nested)

    if not isinstance(value, dict):
        raise DatabaseError("AI run snapshots and results must be JSON objects.")
    check(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        raise DatabaseError("AI run data must be JSON serializable.") from None


def start_ai_run(user_email: str, pid: str, *, paper_uid: str, stage: str,
                 batch_id: str, provider: str, model: str, config_hash: str,
                 source_hash: str, prompt_version: str, prompt_snapshot: dict,
                 expected_version: int | None = None,
                 run_id: str | None = None) -> dict:
    """Persist an attempt before calling a provider; stable IDs make retries safe.

    A matching ``expected_version`` proves the caller holds the current paper
    list, so the document is read back only when no version is supplied.
    """
    if stage not in {"abstract", "fulltext", "extraction"}:
        raise DatabaseError("Unknown AI run stage.")
    values = {
        "id": run_id if run_id is not None else uuid.uuid4().hex,
        "project_id": pid, "user_email": user_email, "paper_uid": paper_uid,
        "stage": stage, "batch_id": batch_id, "provider": provider, "model": model,
        "config_hash": config_hash, "source_hash": source_hash,
        "prompt_version": prompt_version, "prompt_snapshot": _run_json(prompt_snapshot),
    }
    if any(not isinstance(value, str) or not value for key, value in values.items()
           if key != "prompt_snapshot"):
        raise DatabaseError("AI run metadata must contain non-empty strings.")
    try:
        with _get_engine().begin() as conn:
            data = _lock_project(conn, user_email, pid, expected_version,
                                 read=expected_version is None)
            if data is not None and paper_uid not in _paper_uids(data):
                raise DatabaseError("Paper no longer exists in this project.")
            previous = conn.execute(select(ai_runs).where(
                ai_runs.c.id == values["id"],
            )).first()
            if previous is not None:
                stored = dict(previous._mapping)
                if any(stored[key] != value for key, value in values.items()):
                    raise DatabaseError("AI run ID is already used by a different attempt.")
                return stored
            conn.execute(insert(ai_runs).values(
                **values, status="running", started_at=_now(),
                result=None, completed_at=None, duration_seconds=None,
            ))
            return dict(conn.execute(select(ai_runs).where(
                ai_runs.c.id == values["id"],
            )).one()._mapping)
    except SQLAlchemyError:
        _raise_database_error()


def finish_ai_run(user_email: str, pid: str, run_id: str, *, status: str,
                  result: dict, duration_seconds: float, verify_paper: bool = True) -> dict:
    """Complete a running attempt once; never replace a completed result.

    ``verify_paper=False`` skips reading the document: saving a project already
    deletes the attempts of the papers it removes.
    """
    if status not in {"succeeded", "invalid_response", "call_failed"}:
        raise DatabaseError("Unknown terminal AI run status.")
    if (isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, (int, float))
            or not math.isfinite(duration_seconds) or duration_seconds < 0):
        raise DatabaseError("AI run duration must be a finite non-negative number.")
    result = _run_json(result)
    conditions = [ai_runs.c.id == run_id, ai_runs.c.project_id == pid,
                  ai_runs.c.user_email == user_email]
    try:
        with _get_engine().begin() as conn:
            data = _lock_project(conn, user_email, pid, read=verify_paper)
            row = conn.execute(select(ai_runs).where(*conditions)).first()
            if row is None:
                raise DatabaseError("AI run no longer exists or is not accessible.")
            stored = dict(row._mapping)
            if data is not None and stored["paper_uid"] not in _paper_uids(data):
                raise DatabaseError("Paper no longer exists in this project.")
            if stored["status"] != "running":
                if (stored["status"] == status and stored["result"] == result
                        and stored["duration_seconds"] == duration_seconds):
                    return stored
                raise DatabaseError("A completed AI run cannot be changed.")
            conn.execute(update(ai_runs).where(*conditions).values(
                status=status, result=result, duration_seconds=duration_seconds,
                completed_at=_now(),
            ))
            return dict(conn.execute(select(ai_runs).where(*conditions)).one()._mapping)
    except SQLAlchemyError:
        _raise_database_error()


def load_ai_runs(user_email: str, pid: str, stage: str | None = None,
                 paper_uid: str | None = None) -> list[dict]:
    """Return all owned attempts in stable chronological order, without a cap."""
    stmt = select(ai_runs).join(projects, (
        (projects.c.id == ai_runs.c.project_id)
        & (projects.c.user_email == ai_runs.c.user_email)
    )).where(ai_runs.c.project_id == pid, ai_runs.c.user_email == user_email)
    if stage is not None:
        stmt = stmt.where(ai_runs.c.stage == stage)
    if paper_uid is not None:
        stmt = stmt.where(ai_runs.c.paper_uid == paper_uid)
    stmt = stmt.order_by(ai_runs.c.started_at, ai_runs.c.id)
    try:
        with _get_engine().connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt)]
    except SQLAlchemyError:
        _raise_database_error()


_RUN_INDEX_COLUMNS = ("id", "paper_uid", "stage", "provider", "model", "config_hash",
                      "source_hash", "prompt_version", "status", "started_at")


def load_ai_run_index(user_email: str, pid: str, stage: str | None = None) -> list[dict]:
    """Return attempt identities and statuses without answers or prompt snapshots."""
    stmt = select(*(ai_runs.c[name] for name in _RUN_INDEX_COLUMNS)).join(projects, (
        (projects.c.id == ai_runs.c.project_id)
        & (projects.c.user_email == ai_runs.c.user_email)
    )).where(ai_runs.c.project_id == pid, ai_runs.c.user_email == user_email)
    if stage is not None:
        stmt = stmt.where(ai_runs.c.stage == stage)
    stmt = stmt.order_by(ai_runs.c.started_at, ai_runs.c.id)
    try:
        with _get_engine().connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt)]
    except SQLAlchemyError:
        _raise_database_error()


def project_version(user_email: str, pid: str) -> int | None:
    """Return the saved row version, or None when the project is not accessible."""
    stmt = select(func.coalesce(projects.c.version, 1)).where(
        projects.c.id == pid, projects.c.user_email == user_email,
    )
    try:
        with _get_engine().connect() as conn:
            row = conn.execute(stmt).first()
    except SQLAlchemyError:
        _raise_database_error()
    return None if row is None else int(row[0])


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
            _lock_project(conn, user_email, project_id, read=False)
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


def delete_fulltext(user_email: str, project_id: str, paper_uid: str,
                    expected_storage_key: str | None = None) -> None:
    stmt = delete(fulltexts).where(
        fulltexts.c.project_id == project_id,
        fulltexts.c.user_email == user_email,
        fulltexts.c.paper_uid == paper_uid,
    )
    if expected_storage_key is not None:
        stmt = stmt.where(fulltexts.c.storage_key == expected_storage_key)
    try:
        with _get_engine().begin() as conn:
            conn.execute(stmt)
    except SQLAlchemyError:
        _raise_database_error()


def fulltext_key_in_use(user_email: str, storage_key: str) -> bool:
    stmt = select(fulltexts.c.id).where(
        fulltexts.c.user_email == user_email,
        fulltexts.c.storage_key == storage_key,
    ).limit(1)
    try:
        with _get_engine().connect() as conn:
            return conn.execute(stmt).first() is not None
    except SQLAlchemyError:
        _raise_database_error()


def load_fulltexts(user_email: str, project_id: str) -> dict[str, dict]:
    """PDF metadata for a project, keyed by paper_uid."""
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
