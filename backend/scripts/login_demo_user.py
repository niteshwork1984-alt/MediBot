"""Explicit terminal command for testing MediBot login without exposing the session token."""

import argparse
import logging
import os

from app.core.config import BACKEND_DIRECTORY
from app.core.password_hasher import BcryptPasswordHasher
from app.core.session_tokens import SessionTokenService
from app.repositories.user_repository import UserRepository
from app.services.authentication_service import AuthenticationService


# This function reads terminal values required to test a provisioned MediBot user login.
def _parse_arguments() -> argparse.Namespace:
    """Return username and password environment variable name for one login attempt."""
    parser = argparse.ArgumentParser(description="Test a MediBot demo user login.")
    parser.add_argument("--username", required=True, help="Provisioned MediBot demo username.")
    parser.add_argument(
        "--password-env",
        default="MEDIBOT_DEMO_PASSWORD",
        help="Environment variable containing the password to verify.",
    )
    return parser.parse_args()


# This function configures concise terminal logs without recording credentials or session tokens.
def _configure_logging() -> None:
    """Configure timestamped terminal logs for the explicit login test command."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# This function reads a required secret-like setting without including its value in an exception or log.
def _required_environment_value(name: str) -> str:
    """Return one non-empty environment value required by the local login test."""
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"Missing non-empty environment variable: {name}")
    return value


# This function runs one login test and reports only the authenticated identity, never the issued token.
def main() -> None:
    """Authenticate a provisioned demo user using the configured password and signing secret."""
    _configure_logging()
    arguments = _parse_arguments()
    password = _required_environment_value(arguments.password_env)
    signing_secret = _required_environment_value("JWT_SECRET_KEY")
    service = AuthenticationService(
        user_repository=UserRepository(BACKEND_DIRECTORY / "data" / "medibot_auth.db"),
        password_hasher=BcryptPasswordHasher(),
        token_service=SessionTokenService(signing_secret),
    )
    result = service.login(arguments.username, password)
    print(
        "Login succeeded: "
        f"username={result.user.username} role={result.user.role} session_token_created=true"
    )


# This guard starts the explicit login test command only when Python executes this module directly.
if __name__ == "__main__":
    main()
