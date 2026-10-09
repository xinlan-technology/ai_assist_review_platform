"""Original-PDF storage using a private Supabase bucket or isolated local files."""
from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path, PurePosixPath

import streamlit as st

from core import auth


class FulltextStorageError(RuntimeError):
    pass


def _local_storage_allowed() -> bool:
    """Local PDFs are valid only for an explicitly isolated/local database."""
    if os.environ.get("AIREVIEW_DEV") == "1":
        return True
    try:
        database_url = str(st.secrets["database"]["url"]).strip()
    except Exception:
        database_url = ""
    return auth.dev_opt_in() and not database_url


def _remote_config() -> tuple[str, str, str] | None:
    if os.environ.get("AIREVIEW_DEV") == "1":
        return None
    try:
        url = str(st.secrets["storage"]["url"]).strip().rstrip("/")
        key = str(st.secrets["storage"]["service_role_key"]).strip()
        bucket = str(st.secrets["storage"]["bucket"]).strip()
    except Exception:
        return None
    return (url, key, bucket) if url and key and bucket else None


def backend_label() -> str:
    if _remote_config():
        return "private Supabase Storage"
    if _local_storage_allowed():
        return "local development storage"
    return "not configured"


def ensure_configured() -> None:
    if _remote_config() or _local_storage_allowed():
        return
    raise FulltextStorageError(
        "PDF storage is not configured. Add [storage].url, "
        "[storage].service_role_key, and [storage].bucket to Streamlit secrets."
    )


def inspect_pdf(data: bytes) -> int | None:
    """Check PDF signature and encryption; return page count when readable."""
    if not data or not data.lstrip().startswith(b"%PDF-"):
        raise FulltextStorageError("The uploaded file is not a valid PDF.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:
                unlocked = 0
            if not unlocked:
                raise FulltextStorageError(
                    "Password-protected PDFs are not supported. Upload an unlocked copy."
                )
        return len(reader.pages)
    except FulltextStorageError:
        raise
    except Exception:
        # Tolerate parser failures; page count is diagnostic only.
        return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_key(user_email: str, project_id: str, paper_uid: str, digest: str) -> str:
    user_key = hashlib.sha256(user_email.lower().encode("utf-8")).hexdigest()[:20]
    return str(PurePosixPath(user_key, project_id, paper_uid, f"{digest}.pdf"))


def _local_root() -> Path:
    return Path(__file__).resolve().parent.parent / ".local" / "fulltext-pdfs"


def _safe_local_path(key: str) -> Path:
    root = _local_root().resolve()
    candidate = (root / key).resolve()
    if root not in candidate.parents:
        raise FulltextStorageError("Invalid PDF storage key.")
    return candidate


def _client():
    config = _remote_config()
    if not config:
        raise FulltextStorageError("Supabase Storage is not configured.")
    url, key, _ = config
    try:
        from supabase import create_client

        return create_client(url, key)
    except Exception as exc:
        raise FulltextStorageError(f"Could not initialize Supabase Storage: {exc}") from exc


def save_pdf(key: str, data: bytes) -> None:
    ensure_configured()
    config = _remote_config()
    if config:
        _, _, bucket = config
        try:
            _client().storage.from_(bucket).upload(
                path=key,
                file=data,
                file_options={"content-type": "application/pdf", "upsert": "true"},
            )
        except Exception as exc:
            raise FulltextStorageError(f"Could not upload the PDF: {exc}") from exc
        return

    path = _safe_local_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(data)
    except OSError as exc:
        raise FulltextStorageError(f"Could not save the PDF locally: {exc}") from exc


def load_pdf(key: str) -> bytes:
    ensure_configured()
    config = _remote_config()
    if config:
        _, _, bucket = config
        try:
            return bytes(_client().storage.from_(bucket).download(key))
        except Exception as exc:
            raise FulltextStorageError(f"Could not download the PDF: {exc}") from exc

    path = _safe_local_path(key)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FulltextStorageError(f"Could not open the stored PDF: {exc}") from exc


def delete_pdf(key: str | None) -> None:
    if not key:
        return
    ensure_configured()
    config = _remote_config()
    if config:
        _, _, bucket = config
        try:
            _client().storage.from_(bucket).remove([key])
        except Exception as exc:
            raise FulltextStorageError(f"Could not delete the stored PDF: {exc}") from exc
        return

    path = _safe_local_path(key)
    try:
        path.unlink(missing_ok=True)
        # Remove now-empty paper/project/user folders, never the storage root.
        root = _local_root().resolve()
        parent = path.parent
        while parent != root and root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    except OSError as exc:
        raise FulltextStorageError(f"Could not delete the local PDF: {exc}") from exc


def delete_many(keys: list[str]) -> None:
    """Attempt all unique objects; retain failed keys without exposing error details."""
    unique = list(dict.fromkeys(key for key in keys if key))
    failed = []
    for key in unique:
        try:
            delete_pdf(key)
        except Exception:
            failed.append(key)
    if failed:
        error = FulltextStorageError(f"Could not remove {len(failed)} of {len(unique)} PDF objects.")
        error.failed_keys = failed
        raise error from None
