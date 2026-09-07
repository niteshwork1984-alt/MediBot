"""Signed JWT session-token creation and verification for MediBot users."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.ingestion.collection_policy import ALL_ROLES


# This immutable object contains only the trusted identity information extracted from a verified token.
@dataclass(frozen=True)
class AuthenticatedUser:
    """Authenticated MediBot identity and role used by future protected endpoints."""

    user_id: str
    username: str
    role: str


# This class creates and verifies signed role-tagged JWTs without keeping server-side session state.
class SessionTokenService:
    """Issue expiring HS256 JWT session tokens and verify their identity claims."""

    # This constructor stores the signing configuration supplied from runtime environment settings.
    def __init__(
        self,
        signing_secret: str,
        expiration_minutes: int = 60,
        issuer: str = "medibot",
    ) -> None:
        """Create a token service using a non-empty signing secret and positive expiry duration."""
        if not isinstance(signing_secret, str) or len(signing_secret.encode("utf-8")) < 32:
            raise ValueError("JWT signing secret must contain at least 32 bytes.")
        if (
            not isinstance(expiration_minutes, int)
            or isinstance(expiration_minutes, bool)
            or expiration_minutes < 1
        ):
            raise ValueError("Session expiration minutes must be a positive integer.")
        self._signing_secret = signing_secret
        self._expiration_minutes = expiration_minutes
        self._issuer = issuer

    # This helper imports PyJWT only when a token operation is requested.
    def _jwt(self) -> Any:
        """Return the PyJWT module required for signed token operations."""
        try:
            import jwt
        except ImportError as error:
            raise RuntimeError(
                "PyJWT is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error
        return jwt

    # This method creates an expiring token that carries the server-approved user ID, username, and role.
    def create_token(self, user: AuthenticatedUser) -> str:
        """Return a signed JWT session token for one authenticated MediBot user."""
        if user.role not in ALL_ROLES:
            raise ValueError(f"Unsupported role: {user.role}")
        now = datetime.now(timezone.utc)
        claims = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role,
            "iss": self._issuer,
            "iat": now,
            "exp": now + timedelta(minutes=self._expiration_minutes),
        }
        return self._jwt().encode(claims, self._signing_secret, algorithm="HS256")

    # This method verifies signature, expiry, issuer, and required identity claims before returning a role.
    def verify_token(self, token: str) -> AuthenticatedUser:
        """Return trusted identity claims only after successful JWT signature and expiry validation."""
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Session token must be a non-empty string.")
        jwt = self._jwt()
        try:
            claims = jwt.decode(
                token,
                self._signing_secret,
                algorithms=["HS256"],
                issuer=self._issuer,
                options={"require": ["sub", "username", "role", "exp", "iss"]},
            )
        except jwt.PyJWTError as error:
            raise ValueError("Invalid or expired session token.") from error
        return self._authenticated_user_from_claims(claims)

    # This helper validates decoded JWT values before application code can use them as an identity.
    def _authenticated_user_from_claims(self, claims: dict[str, Any]) -> AuthenticatedUser:
        """Return a typed user only when all required JWT claims have valid values."""
        user_id = claims.get("sub")
        username = claims.get("username")
        role = claims.get("role")
        if not all(isinstance(value, str) and value for value in (user_id, username, role)):
            raise ValueError("Session token contains invalid identity claims.")
        if role not in ALL_ROLES:
            raise ValueError("Session token contains an unsupported role.")
        return AuthenticatedUser(user_id=user_id, username=username, role=role)
