"""Private app-table initialization without contacting PostgreSQL."""
from unittest.mock import MagicMock

import pytest
from sqlalchemy.engine import Connection

from core import db


@pytest.mark.parametrize("roles", [[], ["anon"], ["anon", "authenticated"]])
def test_all_app_tables_deny_public_roles_without_removing_policies(roles):
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    connection = MagicMock(spec=Connection)
    connection.dialect = engine.dialect
    connection.execute.return_value.scalars.return_value = roles
    engine.begin.return_value.__enter__.return_value = connection

    db._protect_app_tables(engine)

    statements = [str(call.args[0]) for call in connection.execute.call_args_list]
    assert set(statements[:4]) == {
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"
        for table in ("projects", "fulltexts", "project_sources", "ai_runs")
    }
    assert statements[4] == "SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated')"
    assert statements[5] == (
        "REVOKE ALL PRIVILEGES ON TABLE projects, fulltexts, project_sources, ai_runs FROM PUBLIC"
        + "".join(f', "{role}"' for role in roles)
    )
    assert len(statements) == 6
    connection.begin.assert_not_called()


def test_sqlite_app_tables_do_not_execute_postgres_ddl():
    engine = MagicMock()
    engine.dialect.name = "sqlite"
    db._protect_app_tables(engine)
    engine.begin.assert_not_called()
