"""Unit tests for provisioned-user login, bcrypt hashing, and signed session tokens."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.core.password_hasher import BcryptPasswordHasher, PasswordHasher
from app.core.session_tokens import AuthenticatedUser, SessionTokenService
from app.repositories.user_repository import UserRepository
from app.services.authentication_service import AuthenticationError, AuthenticationService


TEST_SIGNING_SECRET = "test-signing-secret-with-at-least-32-bytes"


# This fake hasher lets authentication-service tests focus on orchestration without bcrypt work factors.
class FakePasswordHasher(PasswordHasher):
    """Deterministic password-hasher substitute used only by unit tests."""

    # This method returns a visibly fake test hash without using cryptography.
    def hash_password(self, password: str) -> str:
        """Return a deterministic fake hash for one test password."""
        return f"test-hash:{password}"

    # This method verifies a test password against the corresponding fake hash.
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Return whether the fake hash matches the supplied test password."""
        return password_hash == self.hash_password(password)


# This test class verifies password hashing and repository-backed authentication behavior.
class AuthenticationServiceTests(unittest.TestCase):
    # This helper creates an initialized temporary users database with one active nurse account.
    def _repository_with_nurse(self, directory: str) -> UserRepository:
        """Return a user repository containing one deterministically hashed nurse user."""
        repository = UserRepository(Path(directory) / "medibot_auth.db")
        repository.initialize_schema()
        repository.create_if_missing(
            "nurse.priya",
            FakePasswordHasher().hash_password("correct-password"),
            "nurse",
        )
        return repository

    # This test verifies bcrypt creates a salted non-plaintext hash and verifies the original password.
    def test_bcrypt_hashes_and_verifies_password(self) -> None:
        hasher = BcryptPasswordHasher()
        password_hash = hasher.hash_password("correct-password")

        self.assertNotEqual(password_hash, "correct-password")
        self.assertTrue(hasher.verify_password("correct-password", password_hash))
        self.assertFalse(hasher.verify_password("wrong-password", password_hash))

    # This test verifies a matching provisioned user receives a token whose role can be verified later.
    def test_login_issues_verified_role_tagged_token(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            service = AuthenticationService(
                self._repository_with_nurse(temporary_directory),
                FakePasswordHasher(),
                SessionTokenService(TEST_SIGNING_SECRET),
            )

            result = service.login("nurse.priya", "correct-password")
            verified_user = SessionTokenService(TEST_SIGNING_SECRET).verify_token(
                result.session_token
            )

        self.assertEqual(verified_user.username, "nurse.priya")
        self.assertEqual(verified_user.role, "nurse")

    # This test verifies a wrong password produces a generic error and no signed token.
    def test_login_rejects_wrong_password(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            service = AuthenticationService(
                self._repository_with_nurse(temporary_directory),
                FakePasswordHasher(),
                SessionTokenService(TEST_SIGNING_SECRET),
            )

            with self.assertRaisesRegex(AuthenticationError, "Invalid username or password"):
                service.login("nurse.priya", "wrong-password")

    # This test verifies a token signed with a different secret cannot be trusted by the application.
    def test_session_token_rejects_wrong_signing_secret(self) -> None:
        token = SessionTokenService("first-signing-secret-with-at-least-32-bytes").create_token(
            AuthenticatedUser("user-1", "nurse.priya", "nurse")
        )

        with self.assertRaisesRegex(ValueError, "Invalid or expired"):
            SessionTokenService("second-signing-secret-with-at-least-32-bytes").verify_token(token)

    # This test verifies weak signing configuration is rejected before any token can be created.
    def test_session_token_rejects_short_signing_secret(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 32 bytes"):
            SessionTokenService("too-short")

    # This test verifies the repository's idempotent provisioning does not duplicate an existing username.
    def test_repository_creates_demo_user_only_once(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            repository = UserRepository(Path(temporary_directory) / "medibot_auth.db")
            repository.initialize_schema()

            first_created = repository.create_if_missing("tech.anand", "hash", "technician")
            second_created = repository.create_if_missing("tech.anand", "new-hash", "technician")

        self.assertTrue(first_created)
        self.assertFalse(second_created)
