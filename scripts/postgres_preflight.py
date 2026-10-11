"""Read-only check for conditions that stop this application on PostgreSQL.

On start-up the application creates its tables if needed, enables row-level
security on them and revokes public privileges, all in one transaction. If the
connecting role may not do that, the application cannot start. This script
inspects the database without changing anything and reports the blocking
conditions it can see. A clean result is not a deployment guarantee: it does
not cover the storage bucket, policies added later, or network access.

Usage:
    AIREVIEW_DATABASE_URL='postgresql+psycopg2://…' python scripts/postgres_preflight.py

The connection string is read from the environment only and is never printed.
"""
from __future__ import annotations

import os
import sys

TABLES = ("projects", "fulltexts", "project_sources", "ai_runs")
PUBLIC_ROLES = ("PUBLIC", "anon", "authenticated")

# The application uses unqualified table names. An existing table is whichever
# one the search path makes visible; a missing table is created in the current
# schema. Neither is necessarily "public".
_ROLE = """
SELECT current_user AS name, current_schema() AS schema, r.rolsuper AS superuser,
       r.rolbypassrls AS bypass_rls,
       has_schema_privilege(current_user, current_schema(), 'CREATE') AS can_create
FROM pg_roles r WHERE r.rolname = current_user
"""
# 'USAGE' is true only when the owner's privileges apply without SET ROLE; a
# NOINHERIT membership does not make this role the owner for ALTER TABLE.
_TABLES = """
SELECT c.relname AS name, n.nspname AS schema, pg_get_userbyid(c.relowner) AS owner,
       pg_has_role(current_user, c.relowner, 'USAGE') AS owned,
       c.relrowsecurity AS rls, c.relforcerowsecurity AS forced
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r' AND c.relname = ANY(:tables) AND pg_table_is_visible(c.oid)
"""
# Read the access-control lists directly: the information schema lists a grant
# only when the current role is its grantor or grantee. Grantee 0 is PUBLIC.
_GRANTS = """
SELECT c.relname AS table_name,
       CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE pg_get_userbyid(a.grantee) END AS grantee,
       string_agg(a.privilege_type, ', ' ORDER BY a.privilege_type) AS privileges
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
     CROSS JOIN LATERAL aclexplode(c.relacl) AS a
WHERE c.relkind = 'r' AND c.relname = ANY(:tables) AND pg_table_is_visible(c.oid)
  AND (a.grantee = 0 OR pg_get_userbyid(a.grantee) = ANY(:grantees))
GROUP BY 1, 2 ORDER BY 1, 2
"""


def assess(role: dict, tables: list[dict], grants: list[dict]) -> tuple[list[str], list[str]]:
    """Return (problems, notes) for the inspected role, tables and public grants."""
    problems, notes = [], []
    found = {table["name"]: table for table in tables}
    missing = [name for name in TABLES if name not in found]
    privileged = bool(role.get("superuser"))
    schema = role.get("schema") or "the current schema"
    if missing:
        if role.get("can_create") or privileged:
            notes.append(f"Will be created in {schema} on first start: " + ", ".join(missing) + ".")
        else:
            problems.append("Missing tables (" + ", ".join(missing) + ") cannot be created: the role "
                            f"has no CREATE privilege on {schema}. The application will not start.")
    for table in found.values():
        name = f"{table['schema']}.{table['name']}" if table.get("schema") else table["name"]
        if not (table["owned"] or privileged):
            problems.append(f"Table {name} is owned by {table['owner']} and this role does not hold "
                            "the owner's privileges. Start-up enables row-level security and revokes "
                            "public grants, which only the owner may do, so the application will "
                            "not start.")
        # Start-up leaves every table with row-level security on, so the forced
        # flag matters even where security is still off today.
        elif table["forced"] and not (role.get("bypass_rls") or privileged):
            problems.append(f"Table {name} forces row-level security on its owner and this role has "
                            "no BYPASSRLS: once security is on, every query returns no rows.")
        if not table["rls"]:
            notes.append(f"Table {name} has no row-level security yet; start-up will enable it.")
    for grant in grants:
        notes.append(f"{grant['grantee']} currently holds {grant['privileges']} on "
                     f"{grant['table_name']}; start-up will revoke it.")
    return problems, notes


def inspect(url: str) -> tuple[dict, list[dict], list[dict]]:
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            # Nothing below can write, whatever the role is allowed to do.
            conn.execute(text("SET TRANSACTION READ ONLY"))
            role = dict(conn.execute(text(_ROLE)).mappings().one())
            tables = [dict(row) for row in conn.execute(text(_TABLES), {"tables": list(TABLES)}).mappings()]
            grants = [dict(row) for row in conn.execute(
                text(_GRANTS), {"tables": list(TABLES), "grantees": list(PUBLIC_ROLES)}).mappings()]
    finally:
        engine.dispose()
    return role, tables, grants


def main() -> int:
    url = os.environ.get("AIREVIEW_DATABASE_URL", "").strip()
    if not url.startswith("postgresql"):
        print("Set AIREVIEW_DATABASE_URL to the PostgreSQL connection string the application will use.")
        return 2
    try:
        role, tables, grants = inspect(url)
    except Exception as exc:
        # Driver messages can repeat the connection string; report the kind only.
        print(f"Could not inspect the database ({type(exc).__name__}). Check the connection string and network.")
        return 2
    problems, notes = assess(role, tables, grants)
    print(f"Role: {role['name']}  (superuser: {role['superuser']}, BYPASSRLS: {role['bypass_rls']})")
    for note in notes:
        print(f"  note     {note}")
    for problem in problems:
        print(f"  PROBLEM  {problem}")
    print("Result: " + ("the application cannot work with this role." if problems else
                        "no blocking condition found. Nothing was changed. This checks the database "
                        "role only and is not a deployment guarantee."))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
