"""Read-only access to the MediAssist SQLite database."""

import re
import sqlite3
from pathlib import Path

from app.core.config import DATABASE_PATH
from app.core.permissions import SQL_ALLOWED_TABLES


# This custom exception tells callers that SQL broke our read-only safety policy.
class UnsafeQueryError(ValueError):
    """Raised when a query is outside the repository's read-only policy."""


_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|VACUUM|PRAGMA|"
    r"ATTACH|DETACH|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE"
    r")\b",
    re.IGNORECASE,
)
_TABLE_REFERENCE = re.compile(r"\b(?:FROM|JOIN)\s+([\w.\"`\[\]]+)", re.IGNORECASE)


# This private helper validates one SQL string before it is sent to SQLite.
def _normalise_sql(sql: str) -> str:
    """Validate and return one safe, read-only SELECT statement."""
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeQueryError("SQL must be a non-empty string.")

    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()

    if not statement:
        raise UnsafeQueryError("SQL must contain a statement.")
    if ";" in statement:
        raise UnsafeQueryError("Only one SQL statement is allowed.")
    if "--" in statement or "/*" in statement or "*/" in statement:
        raise UnsafeQueryError("SQL comments are not allowed.")
    if not statement.upper().startswith("SELECT"):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if _FORBIDDEN_KEYWORDS.search(statement):
        raise UnsafeQueryError("The query contains a forbidden SQL operation.")

    referenced_tables = {
        match.group(1).split(".")[-1].strip('"`[]').lower()
        for match in _TABLE_REFERENCE.finditer(statement)
    }
    unapproved_tables = referenced_tables - SQL_ALLOWED_TABLES
    if unapproved_tables:
        names = ", ".join(sorted(unapproved_tables))
        raise UnsafeQueryError(f"Query references an unapproved table: {names}.")

    return statement


# This is the repository's public method for safely reading rows from MediAssist.
def execute_read_only_query(
    sql: str, database_path: Path = DATABASE_PATH
) -> list[dict]:
    """Execute one validated SELECT query and return its rows as dictionaries.

    The SQLite URI uses ``mode=ro`` as a second safeguard: the database cannot
    be changed even if validation is accidentally weakened in the future.
    """
    statement = _normalise_sql(sql)
    resolved_database_path = Path(database_path).resolve()

    if not resolved_database_path.is_file():
        raise FileNotFoundError(f"Database file does not exist: {resolved_database_path}")

    connection_uri = f"file:{resolved_database_path.as_posix()}?mode=ro"
    with sqlite3.connect(connection_uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(statement).fetchall()

    return [dict(row) for row in rows]
