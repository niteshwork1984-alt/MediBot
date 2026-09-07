"""SQLite persistence for MediBot users; no authentication policy or token creation."""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.collection_policy import ALL_ROLES


# This immutable object represents a user record with its password hash but never plaintext password.
@dataclass(frozen=True)
class UserRecord:
    """Persistent MediBot user identity, role, active state, and password hash."""

    user_id: str
    username: str
    password_hash: str
    role: str
    is_active: bool


# This class contains only SQLite schema and user-record operations for the authentication database.
class UserRepository:
    """Create and query MediBot user records in one dedicated SQLite database."""

    # This constructor stores the authentication database path without opening a connection yet.
    def __init__(self, database_path: Path) -> None:
        """Create a user repository for the supplied SQLite database file."""
        self._database_path = database_path

    # This helper opens a SQLite connection with dictionary-like row access for one repository operation.
    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection after ensuring the parent data directory exists."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    # This method creates the authentication schema only when it does not already exist.
    def initialize_schema(self) -> None:
        """Create the users table and username index for explicit bootstrap operations."""
        allowed_roles = ", ".join(f"'{role}'" for role in sorted(ALL_ROLES))
        with self._connect() as connection:
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ({allowed_roles})),
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                    created_at TEXT NOT NULL
                )
                """
            )

    # This method returns a user by exact username without exposing SQLite details to the service layer.
    def find_by_username(self, username: str) -> UserRecord | None:
        """Return one user record or None when no matching username exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return self._to_user_record(row) if row is not None else None

    # This method creates one user only when its username is not already provisioned.
    def create_if_missing(
        self,
        username: str,
        password_hash: str,
        role: str,
    ) -> bool:
        """Insert one active user and return whether this call created it."""
        if role not in ALL_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        if not isinstance(username, str) or not username.strip():
            raise ValueError("Username must be a non-empty string.")
        with self._connect() as connection:
            existing_user = connection.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            if existing_user is not None:
                return False
            connection.execute(
                """
                INSERT INTO users (id, username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    username,
                    password_hash,
                    role,
                    1,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return True

    # This helper translates one SQLite row into the repository's immutable user model.
    def _to_user_record(self, row: sqlite3.Row) -> UserRecord:
        """Return a UserRecord from one selected users-table row."""
        return UserRecord(
            user_id=str(row["id"]),
            username=str(row["username"]),
            password_hash=str(row["password_hash"]),
            role=str(row["role"]),
            is_active=bool(row["is_active"]),
        )
