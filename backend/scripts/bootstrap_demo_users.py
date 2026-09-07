"""Explicit command that provisions the five MediBot assignment demo accounts once."""

import argparse
import logging
import os
from pathlib import Path

from app.core.password_hasher import BcryptPasswordHasher
from app.core.config import BACKEND_DIRECTORY
from app.repositories.user_repository import UserRepository


DEMO_USERS = (
    ("dr.mehta", "doctor"),
    ("nurse.priya", "nurse"),
    ("billing.ravi", "billing_executive"),
    ("tech.anand", "technician"),
    ("admin.sys", "admin"),
)


# This function reads the database location and password environment variable name for a controlled bootstrap run.
def _parse_arguments() -> argparse.Namespace:
    """Return bootstrap settings without accepting passwords as command-line text."""
    parser = argparse.ArgumentParser(description="Provision the five MediBot demo user accounts.")
    parser.add_argument(
        "--database-path",
        type=Path,
        default=BACKEND_DIRECTORY / "data" / "medibot_auth.db",
        help="SQLite database that will hold password hashes and user roles.",
    )
    parser.add_argument(
        "--password-env",
        default="MEDIBOT_DEMO_PASSWORD",
        help="Environment variable containing the non-empty demo password.",
    )
    return parser.parse_args()


# This function configures concise bootstrap logs without recording any password or token value.
def _configure_logging() -> None:
    """Configure timestamped terminal logs for the explicit demo-user bootstrap command."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# This function reads a non-empty password from an environment variable without displaying it.
def _demo_password(environment_variable_name: str) -> str:
    """Return the configured demo password or fail without exposing its value."""
    password = os.getenv(environment_variable_name, "")
    if not password:
        raise ValueError(
            f"Missing non-empty environment variable for demo password: {environment_variable_name}"
        )
    return password


# This function creates only missing demo users so it can be run safely more than once.
def main() -> None:
    """Initialize the authentication schema and seed the five assignment demo users."""
    _configure_logging()
    arguments = _parse_arguments()
    password_hasher = BcryptPasswordHasher()
    repository = UserRepository(arguments.database_path)
    repository.initialize_schema()
    password = _demo_password(arguments.password_env)
    created_count = 0
    for username, role in DEMO_USERS:
        if repository.create_if_missing(username, password_hasher.hash_password(password), role):
            created_count += 1
    logging.getLogger(__name__).info(
        "Demo user bootstrap completed created_users=%d existing_users=%d",
        created_count,
        len(DEMO_USERS) - created_count,
    )


# This guard starts the explicit provisioning command only when Python executes this module directly.
if __name__ == "__main__":
    main()
