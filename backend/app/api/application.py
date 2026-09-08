"""FastAPI application factory and JWT-protected login and chat endpoints."""

import os
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.schemas import ChatRequest, ChatResponse, ChatSourceResponse, LoginRequest, LoginResponse
from app.core.config import BACKEND_DIRECTORY
from app.core.password_hasher import BcryptPasswordHasher
from app.core.session_tokens import AuthenticatedUser, SessionTokenService
from app.repositories.user_repository import UserRepository
from app.services.authentication_service import AuthenticationError, AuthenticationService
from app.services.chat_service import ChatService, create_medibot_chat_service


BEARER_SCHEME = HTTPBearer(auto_error=False)


# This function reads the signing secret only while composing the production HTTP application.
def _session_token_service_from_environment() -> SessionTokenService:
    """Create the production JWT service without logging the signing secret."""
    signing_secret = os.getenv("JWT_SECRET_KEY", "")
    if not signing_secret:
        raise RuntimeError("Missing required configuration: JWT_SECRET_KEY")
    return SessionTokenService(signing_secret)


# This function composes repository, bcrypt, and JWT dependencies for the login endpoint.
def _authentication_service_from_environment(
    token_service: SessionTokenService,
) -> AuthenticationService:
    """Create the production authentication service for provisioned demo users."""
    return AuthenticationService(
        user_repository=UserRepository(BACKEND_DIRECTORY / "data" / "medibot_auth.db"),
        password_hasher=BcryptPasswordHasher(),
        token_service=token_service,
    )


# This function creates the dependency that reads a role only from a verified bearer token.
def _authenticated_user_dependency(
    token_service: SessionTokenService,
) -> Callable[[HTTPAuthorizationCredentials | None], AuthenticatedUser]:
    """Return a FastAPI dependency that rejects missing, invalid, or expired JWTs."""
    # This nested function performs request-time bearer token verification.
    def current_user(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(BEARER_SCHEME),
        ],
    ) -> AuthenticatedUser:
        """Return the identity extracted from a verified bearer token."""
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            return token_service.verify_token(credentials.credentials)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session token.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None

    return current_user


# This factory builds an application with injectable dependencies for isolated endpoint tests.
def create_application(
    authentication_service: AuthenticationService | None = None,
    chat_service: ChatService | None = None,
    token_service: SessionTokenService | None = None,
) -> FastAPI:
    """Create MediBot's HTTP application without making a provider or Qdrant request at startup."""
    selected_token_service = token_service or _session_token_service_from_environment()
    selected_authentication_service = authentication_service or _authentication_service_from_environment(
        selected_token_service
    )
    selected_chat_service = chat_service or create_medibot_chat_service()
    application = FastAPI(title="MediBot API", version="1.0.0")
    authenticated_user = _authenticated_user_dependency(selected_token_service)

    # This endpoint verifies provisioned-user credentials and returns a signed bearer token.
    @application.post("/login", response_model=LoginResponse)
    def login(request: LoginRequest) -> LoginResponse:
        """Return a role-tagged JWT after a successful provisioned-user login."""
        try:
            result = selected_authentication_service.login(request.username, request.password)
        except AuthenticationError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        return LoginResponse(
            access_token=result.session_token,
            username=result.user.username,
            role=result.user.role,
        )

    # This endpoint uses the role from the verified JWT and routes one question to the correct RAG service.
    @application.post("/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        user: Annotated[AuthenticatedUser, Depends(authenticated_user)],
    ) -> ChatResponse:
        """Return an SQL RAG or Hybrid RAG answer for the JWT-authenticated user."""
        result = selected_chat_service.answer(request.question, user.role)
        return ChatResponse(
            retrieval_type=result.retrieval_type,
            answer=result.answer,
            sources=[
                ChatSourceResponse(
                    source_document=source.source_document,
                    section_title=source.section_title,
                    collection=source.collection,
                )
                for source in result.sources
            ],
        )

    return application
