"""Pydantic request and response contracts for MediBot's HTTP API."""

from pydantic import BaseModel, ConfigDict, Field


# This request model accepts login credentials only at the login boundary.
class LoginRequest(BaseModel):
    """Username and password supplied to the login endpoint."""

    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


# This response model returns a bearer token and safe identity fields after successful login.
class LoginResponse(BaseModel):
    """Successful login response for a provisioned demo user."""

    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


# This request model deliberately contains no role because the server extracts it from the JWT.
class ChatRequest(BaseModel):
    """One authenticated user question for the chat endpoint."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4_000)


# This response model represents one trusted document citation without internal chunk contents or scores.
class ChatSourceResponse(BaseModel):
    """Citation metadata returned with a Hybrid RAG answer."""

    source_document: str
    section_title: str
    collection: str


# This response model returns the selected RAG route, final answer, and any trusted citations.
class ChatResponse(BaseModel):
    """Successful routed chat response for an authenticated session."""

    retrieval_type: str
    answer: str
    sources: list[ChatSourceResponse]
