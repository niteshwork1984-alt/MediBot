"""Unit tests for SQL-versus-Hybrid RAG chat routing without external dependencies."""

import unittest
from unittest.mock import patch

from app.core.config import QDRANT_COLLECTION, QDRANT_URL
from app.services.chat_service import MediBotChatService
from app.services.hybrid_rag_answer_service import HybridRagResult, HybridRagSource


# This fake SQL RAG service returns a deterministic analytical answer and records its calls.
class FakeSQLRagService:
    """In-memory SQL RAG replacement used to test chat route composition."""

    # This constructor prepares an empty call log.
    def __init__(self) -> None:
        """Create a fake SQL RAG service."""
        self.calls: list[tuple[str, str]] = []

    # This method records a SQL-route request and returns a deterministic answer.
    def answer(self, question: str, role: str) -> str:
        """Return a SQL RAG answer without an LLM or SQLite request."""
        self.calls.append((question, role))
        return "There are 4 claims."


# This fake Hybrid RAG service returns a deterministic cited answer and records its calls.
class FakeHybridRagService:
    """In-memory Hybrid RAG replacement used to test chat route composition."""

    # This constructor prepares an empty call log.
    def __init__(self) -> None:
        """Create a fake Hybrid RAG service."""
        self.calls: list[tuple[str, str]] = []

    # This method records a Hybrid-route request and returns an answer with trusted source metadata.
    def answer_with_details(self, question: str, role: str) -> HybridRagResult:
        """Return a Hybrid RAG answer without Qdrant, reranking, or an LLM request."""
        self.calls.append((question, role))
        return HybridRagResult(
            answer="Use contact precautions.",
            sources=[HybridRagSource("infection_control.pdf", "MRSA", "nursing")],
        )


# This test class verifies the existing keyword router is correctly wired through chat orchestration.
class MediBotChatServiceTests(unittest.TestCase):
    # This helper creates the chat service and exposes both fake route targets for assertions.
    def _service(self) -> tuple[MediBotChatService, FakeSQLRagService, FakeHybridRagService]:
        """Return a chat service composed from deterministic route substitutes."""
        sql_rag_service = FakeSQLRagService()
        hybrid_rag_service = FakeHybridRagService()
        return (
            MediBotChatService(sql_rag_service, hybrid_rag_service),
            sql_rag_service,
            hybrid_rag_service,
        )

    # This test verifies an authorized analytical claims question reaches SQL RAG only.
    def test_routes_authorized_analytical_question_to_sql_rag(self) -> None:
        service, sql_rag_service, hybrid_rag_service = self._service()

        result = service.answer("How many claims are there?", "billing_executive")

        self.assertEqual(result.retrieval_type, "sql_rag")
        self.assertEqual(result.answer, "There are 4 claims.")
        self.assertEqual(result.sources, [])
        self.assertEqual(sql_rag_service.calls, [("How many claims are there?", "billing_executive")])
        self.assertEqual(hybrid_rag_service.calls, [])

    # This test verifies a document question reaches Hybrid RAG and keeps its trusted citations.
    def test_routes_document_question_to_hybrid_rag(self) -> None:
        service, sql_rag_service, hybrid_rag_service = self._service()

        result = service.answer("What are MRSA contact precautions?", "nurse")

        self.assertEqual(result.retrieval_type, "hybrid_rag")
        self.assertEqual(result.answer, "Use contact precautions.")
        self.assertEqual(result.sources[0].source_document, "infection_control.pdf")
        self.assertEqual(sql_rag_service.calls, [])
        self.assertEqual(hybrid_rag_service.calls, [("What are MRSA contact precautions?", "nurse")])

    # This test verifies production dependency composition passes the configured Qdrant connection settings.
    def test_factory_uses_configured_qdrant_url_and_collection(self) -> None:
        with (
            patch("app.services.chat_service.create_llm_client", return_value=object()),
            patch("app.services.chat_service.QdrantHybridRetriever") as retriever_class,
            patch("app.services.chat_service.FastEmbedCrossEncoderReranker"),
        ):
            from app.services.chat_service import create_medibot_chat_service

            create_medibot_chat_service()

        retriever_class.assert_called_once_with(
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
        )
