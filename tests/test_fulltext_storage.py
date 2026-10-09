"""Local-backend tests for original PDF persistence (no network required)."""
from __future__ import annotations

import io
import uuid


from pypdf import PdfWriter
from core import fulltext_storage


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
