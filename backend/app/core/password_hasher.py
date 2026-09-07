"""Password-hashing abstractions for MediBot authentication."""

from abc import ABC, abstractmethod


# This abstract base class is the Java-like interface for replaceable password-hashing implementations.
class PasswordHasher(ABC):
    """Hash and verify passwords without exposing plaintext values after login."""

    # This abstract method converts one plaintext password into a persistent one-way hash.
    @abstractmethod
    def hash_password(self, password: str) -> str:
        """Return a one-way password hash for persistent storage."""
        raise NotImplementedError

    # This abstract method verifies a login password against its stored one-way hash.
    @abstractmethod
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Return whether a plaintext password matches the stored hash."""
        raise NotImplementedError


# This concrete implementation uses bcrypt, which includes a random salt in every generated hash.
class BcryptPasswordHasher(PasswordHasher):
    """Use bcrypt to hash and verify MediBot user passwords."""

    # This constructor imports bcrypt only when authentication is used.
    def __init__(self) -> None:
        """Create the bcrypt-backed password hasher."""
        try:
            import bcrypt
        except ImportError as error:
            raise RuntimeError(
                "bcrypt is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error
        self._bcrypt = bcrypt

    # This method validates input and generates a salted bcrypt hash instead of persisting the password.
    def hash_password(self, password: str) -> str:
        """Return a UTF-8 bcrypt hash for one non-empty plaintext password."""
        if not isinstance(password, str) or not password:
            raise ValueError("Password must be a non-empty string.")
        return self._bcrypt.hashpw(
            password.encode("utf-8"),
            self._bcrypt.gensalt(),
        ).decode("utf-8")

    # This method safely compares a login password to the stored bcrypt hash.
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Return whether a non-empty plaintext password matches a bcrypt hash."""
        if not isinstance(password, str) or not password:
            return False
        try:
            return self._bcrypt.checkpw(
                password.encode("utf-8"),
                password_hash.encode("utf-8"),
            )
        except (TypeError, ValueError):
            return False
