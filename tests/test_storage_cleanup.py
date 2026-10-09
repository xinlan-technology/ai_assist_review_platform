"""Best-effort storage cleanup using mocks only."""

import pytest


from core import fulltext_storage


def test_cleanup_attempts_every_unique_key_and_hides_backend_details(monkeypatch):
    attempted = []

    def remove(key):
        attempted.append(key)
        if key in {"first.pdf", "third.pdf"}:
            raise fulltext_storage.FulltextStorageError("Synthetic credential-bearing backend error")

    monkeypatch.setattr(fulltext_storage, "delete_pdf", remove)
    with pytest.raises(fulltext_storage.FulltextStorageError) as captured:
        fulltext_storage.delete_many(["first.pdf", "second.pdf", "first.pdf", "", None, "third.pdf"])
    assert attempted == ["first.pdf", "second.pdf", "third.pdf"]
    assert captured.value.failed_keys == ["first.pdf", "third.pdf"]
    assert str(captured.value) == "Could not remove 2 of 3 PDF objects."
    assert "credential" not in str(captured.value)


def test_cleanup_continues_after_unexpected_backend_exception(monkeypatch):
    attempted = []

    def remove(key):
        attempted.append(key)
        if key == "first.pdf":
            raise OSError("Synthetic I/O failure")

    monkeypatch.setattr(fulltext_storage, "delete_pdf", remove)
    with pytest.raises(fulltext_storage.FulltextStorageError):
        fulltext_storage.delete_many(["first.pdf", "second.pdf"])
    assert attempted == ["first.pdf", "second.pdf"]


def test_cleanup_success_and_empty_input(monkeypatch):
    attempted = []
    monkeypatch.setattr(fulltext_storage, "delete_pdf", attempted.append)
    fulltext_storage.delete_many(["one.pdf", "one.pdf", "two.pdf"])
    fulltext_storage.delete_many(["", None])
    assert attempted == ["one.pdf", "two.pdf"]
