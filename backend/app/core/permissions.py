"""Central authorization policy for MediBot backend capabilities."""

# Roles permitted to use the SQL-RAG route.
SQL_RAG_ALLOWED_ROLES = {"billing_executive", "admin"}

# SQLite tables that SQL-RAG may read from.
SQL_ALLOWED_TABLES = {"claims", "maintenance_tickets"}


# This is the central role check used before a caller can enter the SQL-RAG flow.
def can_use_sql_rag(role: str) -> bool:
    """Return whether a role may use the SQL-RAG capability."""
    return role in SQL_RAG_ALLOWED_ROLES
