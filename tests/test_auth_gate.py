"""The no-login fallback and the e-mail allowlist must both fail closed."""
import pytest

from core import auth


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("AIREVIEW_DEV", raising=False)


def _secrets(monkeypatch, values):
    monkeypatch.setattr(auth.st, "secrets", values)


def test_dev_fallback_refused_when_a_database_is_configured(no_env, monkeypatch):
    _secrets(monkeypatch, {
        "dev": {"allow_no_auth": True},
        "database": {"url": "postgresql+psycopg2://user:pw@host/db"},
    })
    assert auth._dev_mode_allowed() is False


def test_dev_fallback_refused_when_storage_is_configured(no_env, monkeypatch):
    _secrets(monkeypatch, {
        "dev": {"allow_no_auth": True},
        "storage": {"url": "https://ref.supabase.co", "bucket": "fulltext-pdfs"},
    })
    assert auth._dev_mode_allowed() is False


def test_dev_fallback_requires_a_literal_true(no_env, monkeypatch):
    for value in ("true", "yes", 1, [1]):
        _secrets(monkeypatch, {"dev": {"allow_no_auth": value}})
        assert auth._dev_mode_allowed() is False, value
    _secrets(monkeypatch, {"dev": {"allow_no_auth": True}})
    assert auth._dev_mode_allowed() is True


def test_dev_env_var_still_wins_for_local_development(monkeypatch):
    monkeypatch.setenv("AIREVIEW_DEV", "1")
    _secrets(monkeypatch, {"database": {"url": "postgresql://x"}})
    assert auth._dev_mode_allowed() is True
    assert auth._auth_configured() is False


def test_absent_access_section_means_no_restriction(no_env, monkeypatch):
    _secrets(monkeypatch, {"auth": {"client_id": "x"}})
    assert auth._allowlist() is None


def test_allowlist_is_normalized(no_env, monkeypatch):
    _secrets(monkeypatch, {"access": {"allowed_emails": [" Name@Example.com ", "b@x.org"]}})
    assert auth._allowlist() == ["name@example.com", "b@x.org"]


@pytest.mark.parametrize("section", [
    {"allowed_emails": "name@example.com"},
    {"allowed_email": ["name@example.com"]},
    {"allowed_emails": []},
    {"allowed_emails": ["  "]},
    {},
])
def test_broken_allowlist_fails_closed(no_env, monkeypatch, section):
    _secrets(monkeypatch, {"access": section})
    with pytest.raises(auth.AccessConfigError):
        auth._allowlist()
