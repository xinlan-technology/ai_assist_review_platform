"""The read-only database check reports the conditions that stop start-up."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "postgres_preflight.py"
spec = importlib.util.spec_from_file_location("postgres_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

OWNER = {"name": "app", "superuser": False, "bypass_rls": False, "can_create": True}


def _table(name, **changes):
    return {"name": name, "owner": "app", "owned": True, "rls": True, "forced": False} | changes


def test_owner_with_protected_tables_can_start():
    problems, notes = preflight.assess(OWNER, [_table(name) for name in preflight.TABLES], [])
    assert problems == [] and notes == []


def test_fresh_database_is_created_and_public_grants_are_reported():
    grants = [{"table_name": "projects", "grantee": "anon", "privileges": "SELECT"}]
    problems, notes = preflight.assess(OWNER, [_table("projects", rls=False)], grants)
    assert problems == []
    assert any("Will be created" in note and "ai_runs" in note for note in notes)
    assert any("no row-level security yet" in note for note in notes)
    assert any("anon currently holds SELECT" in note for note in notes)


@pytest.mark.parametrize("role, table, expected", [
    (OWNER | {"can_create": False}, None, "cannot be created"),
    (OWNER, _table("projects", owner="postgres", owned=False), "only the"),
    (OWNER | {"bypass_rls": True}, _table("projects", owner="postgres", owned=False), "only the"),
    (OWNER, _table("projects", forced=True), "forces row-level security"),
    # Security is still off, but start-up turns it on and the forced flag then applies.
    (OWNER, _table("projects", rls=False, forced=True), "forces row-level security"),
])
def test_conditions_that_stop_the_application_are_problems(role, table, expected):
    tables = [_table(name) for name in preflight.TABLES if not table or name != table["name"]]
    if table:
        tables.append(table)
    else:
        tables = []
    problems, _ = preflight.assess(role, tables, [])
    assert any(expected in problem for problem in problems)


def test_bypass_role_reads_forced_tables_and_superuser_needs_no_ownership():
    forced = [_table(name, forced=True) for name in preflight.TABLES]
    assert preflight.assess(OWNER | {"bypass_rls": True}, forced, [])[0] == []
    foreign = [_table(name, owner="postgres", owned=False) for name in preflight.TABLES]
    assert preflight.assess(OWNER | {"superuser": True}, foreign, [])[0] == []


def test_missing_connection_string_never_connects(monkeypatch, capsys):
    monkeypatch.delenv("AIREVIEW_DATABASE_URL", raising=False)
    monkeypatch.setattr(preflight, "inspect", lambda url: pytest.fail("must not connect"))
    assert preflight.main() == 2
    assert "AIREVIEW_DATABASE_URL" in capsys.readouterr().out


def test_connection_failure_never_prints_the_connection_string(monkeypatch, capsys):
    secret = "postgresql+psycopg2://app:very-secret@db.example.invalid/postgres"
    monkeypatch.setenv("AIREVIEW_DATABASE_URL", secret)

    def fail(url):
        raise RuntimeError(f"could not connect to {url}")

    monkeypatch.setattr(preflight, "inspect", fail)
    assert preflight.main() == 2
    output = capsys.readouterr().out
    assert "very-secret" not in output and "example.invalid" not in output


def test_tables_are_found_through_the_search_path_and_named_with_their_schema():
    assert "pg_table_is_visible(c.oid)" in preflight._TABLES
    assert "pg_table_is_visible(c.oid)" in preflight._GRANTS
    assert "current_schema()" not in preflight._TABLES + preflight._GRANTS
    foreign = _table("projects", owner="postgres", owned=False) | {"schema": "shared"}
    others = [_table(name) for name in preflight.TABLES if name != "projects"]
    problems, _ = preflight.assess(OWNER, [foreign, *others], [])
    assert len(problems) == 1 and "shared.projects" in problems[0]
