"""Endpoint tests for login and JWT-protected chat without real RAG dependencies."""

import unittest

from fastapi.testclient import TestClient

from app.api.application import create_application
from app.core.session_tokens import AuthenticatedUser, SessionTokenService
from app.services.authentication_service import AuthenticationError, LoginResult
from app.services.chat_service import ChatResult, ChatService, ChatSource


TEST_SIGNING_SECRET = "api-test-signing-secret-with-at-least-32-bytes"


# This fake authentication service returns a prepared login outcome without SQLite or bcrypt.
class FakeAuthenticationService:
    """In-memory login replacement used to test FastAPI transport behavior."""

    # This constructor stores a token service used to create a valid token for the fixed test user.
    def __init__(self, token_service: SessionTokenService, should_fail: bool = False) -> None:
        """Create a fake authentication service with optional failed-login behavior."""
        self._token_service = token_service
        self._should_fail = should_fail
        self.calls: list[tuple[str, str]] = []

    # This method records credentials and returns a valid signed token or the expected generic login error.
    def login(self, username: str, password: str) -> LoginResult:
        """Return a fixed nurse identity for successful test credentials."""
        self.calls.append((username, password))
        if self._should_fail:
            raise AuthenticationError("Invalid username or password.")
        user = AuthenticatedUser("user-1", "nurse.priya", "nurse")
        return LoginResult(user=user, session_token=self._token_service.create_token(user))


# This fake chat service records the role received from the verified token and returns one safe answer.
class FakeChatService(ChatService):
    """In-memory chat replacement used to test routing endpoint behavior."""

    # This constructor prepares an empty call log.
    def __init__(self) -> None:
        """Create a deterministic fake chat service."""
        self.calls: list[tuple[str, str]] = []

    # This method records the request and returns a fixed Hybrid RAG answer with a trusted-looking source.
    def answer(self, question: str, role: str) -> ChatResult:
        """Return one deterministic chat result without Qdrant, reranking, or an LLM call."""
        self.calls.append((question, role))
        return ChatResult(
            retrieval_type="hybrid_rag",
            answer="Use contact precautions.",
            sources=[ChatSource("infection_control.pdf", "MRSA", "nursing")],
        )


# This test class verifies the HTTP boundary, including JWT-derived role handling.
class MediBotApiTests(unittest.TestCase):
    # This helper creates a test client and its inspectable fake dependencies.
    def _client(
        self,
        should_fail_login: bool = False,
    ) -> tuple[TestClient, FakeAuthenticationService, FakeChatService]:
        """Return an in-memory FastAPI client with deterministic fake services."""
        token_service = SessionTokenService(TEST_SIGNING_SECRET)
        authentication_service = FakeAuthenticationService(token_service, should_fail_login)
        chat_service = FakeChatService()
        application = create_application(authentication_service, chat_service, token_service)
        return TestClient(application), authentication_service, chat_service

    # This test verifies login returns a bearer token and safe identity information for valid credentials.
    def test_login_returns_bearer_token(self) -> None:
        client, authentication_service, _ = self._client()

        response = client.post(
            "/login",
            json={"username": "nurse.priya", "password": "demo-password"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["token_type"], "bearer")
        self.assertEqual(response.json()["username"], "nurse.priya")
        self.assertEqual(response.json()["role"], "nurse")
        self.assertTrue(response.json()["access_token"])
        self.assertEqual(authentication_service.calls, [("nurse.priya", "demo-password")])

    # This test verifies login returns a generic 401 result when authentication fails.
    def test_login_rejects_invalid_credentials(self) -> None:
        client, _, _ = self._client(should_fail_login=True)

        response = client.post("/login", json={"username": "nurse.priya", "password": "wrong"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password.")

    # This test verifies chat rejects a request with no bearer token before calling any RAG service.
    def test_chat_rejects_missing_bearer_token(self) -> None:
        client, _, chat_service = self._client()

        response = client.post("/chat", json={"question": "What are MRSA precautions?"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(chat_service.calls, [])

    # This test verifies chat sends the role from a verified JWT to the chat service.
    def test_chat_uses_role_from_verified_bearer_token(self) -> None:
        client, _, chat_service = self._client()
        login_response = client.post(
            "/login",
            json={"username": "nurse.priya", "password": "demo-password"},
        )
        token = login_response.json()["access_token"]

        response = client.post(
            "/chat",
            json={"question": "What are MRSA precautions?"},
            headers={"Authorization": f"Bearer {token}"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(chat_service.calls, [("What are MRSA precautions?", "nurse")])
        self.assertEqual(response.json()["retrieval_type"], "hybrid_rag")
        self.assertEqual(response.json()["sources"][0]["collection"], "nursing")

    # This test verifies the request schema rejects a role field so the JWT remains the only role source.
    def test_chat_rejects_client_supplied_role(self) -> None:
        client, _, chat_service = self._client()
        login_response = client.post(
            "/login",
            json={"username": "nurse.priya", "password": "demo-password"},
        )

        response = client.post(
            "/chat",
            json={"question": "What are MRSA precautions?", "role": "admin"},
            headers={"Authorization": f"Bearer {login_response.json()['access_token']}"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(chat_service.calls, [])

    # This test verifies a malformed bearer token cannot access the chat service.
    def test_chat_rejects_invalid_bearer_token(self) -> None:
        client, _, chat_service = self._client()

        response = client.post(
            "/chat",
            json={"question": "What are MRSA precautions?"},
            headers={"Authorization": "Bearer invalid-token"},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(chat_service.calls, [])
