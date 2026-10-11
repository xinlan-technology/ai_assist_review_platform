"""Local-backend tests for original PDF persistence (no network required)."""
from __future__ import annotations

import io
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest
from pypdf import PdfWriter
from core import fulltext_storage
from core.fulltext_storage import _client as initialize_storage_client


def _one_page_pdf() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(output)
    return output.getvalue()


def test_inspect_pdf_counts_pages_without_text_extraction():
    assert fulltext_storage.inspect_pdf(_one_page_pdf()) == 1
    try:
        fulltext_storage.inspect_pdf(b"not a pdf")
    except fulltext_storage.FulltextStorageError:
        pass
    else:
        raise AssertionError("non-PDF data must be rejected")


@pytest.mark.parametrize("password", ["fixture-password", ""])
def test_inspection_preserves_password_checks(password):
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(password, owner_password="fixture-owner-password")
    writer.write(output)
    if password:
        with pytest.raises(fulltext_storage.FulltextStorageError, match="Password-protected"):
            fulltext_storage.inspect_pdf(output.getvalue())
    else:
        assert fulltext_storage.inspect_pdf(output.getvalue()) == 1


def test_malformed_pdf_keeps_unknown_page_count():
    assert fulltext_storage.inspect_pdf(b"%PDF-1.7\nnot a readable document") is None


def test_inspection_timeout_kills_and_reaps_child(monkeypatch):
    processes = []
    popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        child = popen(*args, **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(fulltext_storage, "PDF_INSPECTION_TIMEOUT", 0.000001)
    with pytest.raises(fulltext_storage.FulltextStorageError, match="inspected safely"):
        fulltext_storage.inspect_pdf(_one_page_pdf())
    assert len(processes) == 1
    assert processes[0].returncode is not None
    assert processes[0].poll() is not None


@pytest.mark.parametrize("returncode,stdout", [
    (-9, b""), (1, b"private backend detail"), (0, b"not json"),
    (0, b'{"error":"resource"}'), (0, b'{"pages":true}'),
    (0, b'{"pages":-1}'), (0, b'{"pages":1,"unexpected":2}'),
    (0, b"x" * 1025),
])
def test_worker_failure_and_invalid_reports_are_safe(monkeypatch, returncode, stdout):
    def run(command, **kwargs):
        assert command[:2] == [sys.executable, "-I"]
        assert command[2].endswith("/_pdf_inspector.py")
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["timeout"] == fulltext_storage.PDF_INSPECTION_TIMEOUT
        assert not kwargs.get("shell")
        return subprocess.CompletedProcess(command, returncode, stdout)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(fulltext_storage.FulltextStorageError) as captured:
        fulltext_storage.inspect_pdf(b"%PDF-fixture")
    assert str(captured.value) == fulltext_storage._PDF_INSPECTION_ERROR


def test_inspection_rejects_oversize_before_starting_child(monkeypatch):
    monkeypatch.setattr(fulltext_storage, "MAX_UPLOAD_PDF_BYTES", 10)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with pytest.raises(fulltext_storage.FulltextStorageError, match="larger than 50 MB"):
        fulltext_storage.inspect_pdf(b"%PDF-" + b"x" * 6)


@pytest.mark.parametrize("platform,expected", [
    ("linux", [(1, (5, 5)), (2, (512 * 1024 * 1024,) * 2)]),
    ("darwin", [(1, (5, 5))]), ("win32", []),
])
def test_worker_resource_limits_are_platform_scoped(monkeypatch, platform, expected):
    from core import _pdf_inspector

    calls = []
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RLIMIT_CPU=1, RLIMIT_AS=2,
        setrlimit=lambda kind, limit: calls.append((kind, limit)),
    ))
    monkeypatch.setattr(_pdf_inspector, "sys", SimpleNamespace(platform=platform))
    _pdf_inspector._limits()
    assert calls == expected


@pytest.mark.parametrize("operation", ["initialize", "save", "load", "delete"])
def test_remote_errors_never_expose_service_credentials(monkeypatch, operation):
    secret = "synthetic-service-role-secret"

    def fail(*args, **kwargs):
        raise RuntimeError(f"Authorization: Bearer {secret}; private request details")

    monkeypatch.setattr(fulltext_storage, "_remote_config", lambda: (
        "https://example.invalid", secret, "fixture-bucket",
    ))
    # Never instantiate the real SDK, including the direct initialization test.
    monkeypatch.setitem(sys.modules, "supabase", SimpleNamespace(create_client=fail))
    bucket = SimpleNamespace(upload=fail, download=fail, remove=fail)
    monkeypatch.setattr(fulltext_storage, "_client", lambda: SimpleNamespace(
        storage=SimpleNamespace(from_=lambda name: bucket),
    ))
    with pytest.raises(fulltext_storage.FulltextStorageError) as captured:
        if operation == "initialize":
            initialize_storage_client()
        elif operation == "save":
            fulltext_storage.save_pdf("fixture.pdf", b"fixture")
        elif operation == "load":
            fulltext_storage.load_pdf("fixture.pdf")
        else:
            fulltext_storage.delete_pdf("fixture.pdf")
    assert secret not in str(captured.value)
    assert "private request details" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__


@pytest.mark.parametrize("operation", ["mkdir", "save", "load", "delete"])
def test_local_io_errors_never_expose_private_paths(monkeypatch, operation):
    def fail(*args, **kwargs):
        raise OSError("Cannot access /private/synthetic-user/research/secret.pdf")

    parent = SimpleNamespace(mkdir=fail if operation == "mkdir" else lambda **kwargs: None)
    path = SimpleNamespace(parent=parent, write_bytes=fail, read_bytes=fail, unlink=fail)
    monkeypatch.setattr(fulltext_storage, "_safe_local_path", lambda key: path)
    with pytest.raises(fulltext_storage.FulltextStorageError) as captured:
        if operation in {"mkdir", "save"}:
            fulltext_storage.save_pdf("fixture.pdf", b"fixture")
        elif operation == "load":
            fulltext_storage.load_pdf("fixture.pdf")
        else:
            fulltext_storage.delete_pdf("fixture.pdf")
    assert "synthetic-user" not in str(captured.value)
    assert "secret.pdf" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__


def test_local_storage_round_trip_and_private_key():
    data = _one_page_pdf()
    digest = fulltext_storage.sha256(data)
    unique = uuid.uuid4().hex
    email = "person@example.edu"
    key = fulltext_storage.object_key(email, f"project-{unique}", "paper-1", digest)
    assert email not in key
    fulltext_storage.save_pdf(key, data)
    try:
        assert fulltext_storage.load_pdf(key) == data
    finally:
        fulltext_storage.delete_pdf(key)
