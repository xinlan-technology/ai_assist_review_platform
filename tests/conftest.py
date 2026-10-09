"""Keep regression tests isolated from credentials, live databases, and APIs."""
import socket

import pytest

from core import db, fulltext_storage


@pytest.fixture(autouse=True)
def offline_only(monkeypatch, tmp_path):
    monkeypatch.setenv("AIREVIEW_DEV", "1")

    def blocked(*args, **kwargs):
        raise AssertionError("Live I/O is disabled in tests; provide an explicit mock.")

    monkeypatch.setattr(db, "_get_engine", blocked)
    monkeypatch.setattr(fulltext_storage, "_client", blocked)
    monkeypatch.setattr(fulltext_storage, "_local_root", lambda: tmp_path / "pdfs")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
