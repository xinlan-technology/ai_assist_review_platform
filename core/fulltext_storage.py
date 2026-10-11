"""Original-PDF storage using a private Supabase bucket or isolated local files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

import streamlit as st

from core import auth


class FulltextStorageError(RuntimeError):
    pass


MAX_UPLOAD_PDF_BYTES = 50 * 1024 * 1024
PDF_INSPECTION_TIMEOUT = 10
_PDF_INSPECTION_ERROR = "The PDF could not be inspected safely. Try a repaired or smaller copy."


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
    """Inspect untrusted PDFs in a bounded child, never in the app process."""
    if not isinstance(data, (bytes, bytearray)) or not data or not data.lstrip().startswith(b"%PDF-"):
        raise FulltextStorageError("The uploaded file is not a valid PDF.")
    if len(data) > MAX_UPLOAD_PDF_BYTES:
        raise FulltextStorageError("The PDF is larger than 50 MB. Compress it before uploading.")
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("_pdf_inspector.py"))],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=PDF_INSPECTION_TIMEOUT, check=False,
        )
        if result.returncode or len(result.stdout) > 1024:
            raise ValueError("PDF inspection worker failed")
        report = json.loads(result.stdout)
        if not isinstance(report, dict):
            raise ValueError("Invalid PDF inspection report")
        if report == {"error": "password"}:
            raise FulltextStorageError(
                "Password-protected PDFs are not supported. Upload an unlocked copy."
            )
        pages = report.get("pages")
        if set(report) != {"pages"} or (pages is not None and (type(pages) is not int or pages < 0)):
            raise ValueError("Invalid PDF inspection report")
        return pages
    except (OSError, subprocess.SubprocessError, ValueError):
        # subprocess.run kills and reaps the worker on timeout.
        raise FulltextStorageError(_PDF_INSPECTION_ERROR) from None


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
    except Exception:
        raise FulltextStorageError("Could not initialize PDF storage. Check the storage configuration.") from None


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
        except Exception:
            raise FulltextStorageError("Could not upload the PDF. Check storage or try again.") from None
        return

    try:
        path = _safe_local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError:
        raise FulltextStorageError("Could not save the PDF locally. Check storage permissions.") from None


def load_pdf(key: str) -> bytes:
    ensure_configured()
    config = _remote_config()
    if config:
        _, _, bucket = config
        try:
            return bytes(_client().storage.from_(bucket).download(key))
        except Exception:
            raise FulltextStorageError("Could not download the PDF. Check storage or try again.") from None

    try:
        path = _safe_local_path(key)
        return path.read_bytes()
    except OSError:
        raise FulltextStorageError("Could not open the stored PDF. Reattach it or try again.") from None


def delete_pdf(key: str | None) -> None:
    if not key:
        return
    ensure_configured()
    config = _remote_config()
    if config:
        _, _, bucket = config
        try:
            _client().storage.from_(bucket).remove([key])
        except Exception:
            raise FulltextStorageError("Could not delete the stored PDF. Check storage or try again.") from None
        return

    try:
        path = _safe_local_path(key)
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
    except OSError:
        raise FulltextStorageError("Could not delete the local PDF. Check storage permissions.") from None


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
