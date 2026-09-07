"""Login orchestration for MediBot users without FastAPI transport concerns."""

import logging
from dataclasses import dataclass

from app.core.password_hasher import PasswordHasher
from app.core.session_tokens import AuthenticatedUser, SessionTokenService
from app.repositories.user_repository import UserRepository


LOGGER = logging.getLogger(__name__)


# This exception represents a deliberately generic failed-login result that does not reveal which credential failed.
class AuthenticationError(ValueError):
    """Raised when username, password, or user active state does not permit login."""


# This immutable object carries a successful identity and its signed session token to a future transport layer.
@dataclass(frozen=True)
class LoginResult:
    """Successful login result with authenticated identity and an expiring session token."""

    user: AuthenticatedUser
    session_token: str


# This class coordinates user lookup, password verification, and token issuance without exposing HTTP details.
class AuthenticationService:
    """Authenticate one provisioned internal user and issue a signed session token."""

    # This constructor accepts replaceable dependencies so tests avoid real databases and cryptographic libraries.
    def __init__(
        self,
        user_repository: UserRepository,
        password_hasher: PasswordHasher,
        token_service: SessionTokenService,
    ) -> None:
        """Create the authentication service with repository, hash, and token dependencies."""
        self._user_repository = user_repository
        self._password_hasher = password_hasher
        self._token_service = token_service

    # This method verifies supplied login credentials and creates a role-tagged token only after successful verification.
    def login(self, username: str, password: str) -> LoginResult:
        """Return a signed session token for one active user with a matching password."""
        if not isinstance(username, str) or not username.strip():
            raise AuthenticationError("Invalid username or password.")
        user = self._user_repository.find_by_username(username)
        if user is None or not user.is_active:
            LOGGER.warning("Login rejected username=%s", username)
            raise AuthenticationError("Invalid username or password.")
        if not self._password_hasher.verify_password(password, user.password_hash):
            LOGGER.warning("Login rejected username=%s", username)
            raise AuthenticationError("Invalid username or password.")

        authenticated_user = AuthenticatedUser(
            user_id=user.user_id,
            username=user.username,
            role=user.role,
        )
        session_token = self._token_service.create_token(authenticated_user)
        LOGGER.info("Login accepted username=%s role=%s", user.username, user.role)
        return LoginResult(user=authenticated_user, session_token=session_token)
