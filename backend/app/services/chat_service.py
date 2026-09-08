"""Provider-independent chat orchestration that selects SQL RAG or Hybrid RAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import QDRANT_COLLECTION, QDRANT_URL
from app.services.hybrid_rag_answer_service import (
    HybridRagAnswerService,
    HybridRagSource,
)
from app.services.llm.llm_client import create_llm_client
from app.services.reranking.fastembed_cross_encoder_reranker import FastEmbedCrossEncoderReranker
from app.services.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever
from app.services.routing_service import RetrievalType, route_question
from app.services.sql_rag_service import SQLRagService


# This immutable value is one transport-safe citation returned from the Hybrid RAG path.
@dataclass(frozen=True)
class ChatSource:
    """Trusted document citation included with a Hybrid RAG chat answer."""

    source_document: str
    section_title: str
    collection: str


# This immutable value is the common response returned from either route.
@dataclass(frozen=True)
class ChatResult:
    """One routed chat answer and its optional trusted document citations."""

    retrieval_type: RetrievalType
    answer: str
    sources: list[ChatSource]


# This abstract base class is the Java-like interface used by the FastAPI transport layer.
class ChatService(ABC):
    """Route one authenticated question and return a transport-safe answer."""

    # This abstract method defines the common chat operation for real and test implementations.
    @abstractmethod
    def answer(self, question: str, role: str) -> ChatResult:
        """Return a routed answer using the supplied server-verified role."""
        raise NotImplementedError


# This function translates Hybrid RAG source objects into the common chat response type.
def _chat_sources(sources: list[HybridRagSource]) -> list[ChatSource]:
    """Return transport-safe sources without exposing internal reranking scores or chunk text."""
    return [
        ChatSource(
            source_document=source.source_document,
            section_title=source.section_title,
            collection=source.collection,
        )
        for source in sources
    ]


# This concrete service composes the existing router, SQL RAG service, and Hybrid RAG answer service.
class MediBotChatService(ChatService):
    """Route authenticated questions to SQL RAG or Hybrid RAG."""

    # This constructor receives both RAG services so tests can use deterministic substitutes.
    def __init__(
        self,
        sql_rag_service: SQLRagService,
        hybrid_rag_service: HybridRagAnswerService,
    ) -> None:
        """Store the already-constructed SQL and Hybrid RAG services."""
        self._sql_rag_service = sql_rag_service
        self._hybrid_rag_service = hybrid_rag_service

    # This method chooses the route once and returns the common response object expected by FastAPI.
    def answer(self, question: str, role: str) -> ChatResult:
        """Route one authenticated question without accepting a client-provided role elsewhere."""
        retrieval_type = route_question(question, role)
        if retrieval_type == "sql_rag":
            return ChatResult(
                retrieval_type=retrieval_type,
                answer=self._sql_rag_service.answer(question, role),
                sources=[],
            )

        hybrid_result = self._hybrid_rag_service.answer_with_details(question, role)
        return ChatResult(
            retrieval_type=retrieval_type,
            answer=hybrid_result.answer,
            sources=_chat_sources(hybrid_result.sources),
        )


# This factory assembles production dependencies while keeping construction outside FastAPI endpoints.
def create_medibot_chat_service() -> MediBotChatService:
    """Create the configured RAG services for one application process."""
    llm_client = create_llm_client()
    return MediBotChatService(
        sql_rag_service=SQLRagService(llm_client),
        hybrid_rag_service=HybridRagAnswerService(
            retriever=QdrantHybridRetriever(
                url=QDRANT_URL,
                collection_name=QDRANT_COLLECTION,
            ),
            reranker=FastEmbedCrossEncoderReranker(),
            llm_client=llm_client,
        ),
    )
