"""Unit tests for Hybrid RAG orchestration without Qdrant, reranker inference, or provider calls."""

import unittest

from app.core.config import LLMModelPurpose
from app.services.hybrid_rag_answer_service import (
    HybridRagAnswerService,
    build_answer_context,
    build_trusted_sources,
)
from app.services.llm.llm_client import LLMClient
from app.services.reranking.reranker import RerankedChunk, Reranker
from app.services.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk


# This fake retriever returns prepared candidates and records the requested role and limit.
class FakeRetriever(HybridRetriever):
    """In-memory retrieval replacement for Hybrid RAG service tests."""

    # This constructor stores the prepared candidates for a deterministic retrieval result.
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        """Create a fake retriever with one prepared candidate result."""
        self.candidates = candidates
        self.calls: list[tuple[str, str, int]] = []

    # This method records the request and returns the prepared role-authorized candidates.
    def search(self, question: str, role: str, limit: int = 5) -> list[RetrievedChunk]:
        """Return prepared candidates without connecting to Qdrant."""
        self.calls.append((question, role, limit))
        return self.candidates


# This fake reranker returns prepared final chunks and records the candidate batch it receives.
class FakeReranker(Reranker):
    """In-memory reranker replacement for Hybrid RAG service tests."""

    # This constructor stores prepared reranked chunks for a deterministic result.
    def __init__(self, results: list[RerankedChunk]) -> None:
        """Create a fake reranker with one prepared final context."""
        self.results = results
        self.calls: list[tuple[str, list[RetrievedChunk], int]] = []

    # This method records the request and returns prepared reranked chunks.
    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        limit: int,
    ) -> list[RerankedChunk]:
        """Return prepared chunks without running a cross-encoder model."""
        self.calls.append((question, candidates, limit))
        return self.results


# This fake client records prompts and returns a deterministic answer without a network call.
class FakeLLMClient(LLMClient):
    """In-memory LLM replacement for Hybrid RAG service tests."""

    # This constructor stores one response and prepares a call log.
    def __init__(self, response: str) -> None:
        """Create a fake LLM client with a deterministic answer."""
        self.response = response
        self.calls: list[dict[str, object]] = []

    # This method records the answer prompt and returns the configured response.
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: LLMModelPurpose,
    ) -> str:
        """Return a deterministic answer without a provider request."""
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt, "purpose": purpose}
        )
        return self.response


# This test class verifies the retrieve-rerank-answer pipeline and safe citation handling.
class HybridRagAnswerServiceTests(unittest.TestCase):
    # This helper creates a trusted retrieval record with controllable source metadata.
    def _chunk(self, text: str, section_title: str = "MRSA precautions") -> RetrievedChunk:
        """Return one deterministic retrieved chunk for test context."""
        return RetrievedChunk(
            text=text,
            source_document="infection_control.pdf",
            document_key="nursing/infection_control.pdf",
            collection="nursing",
            section_title=section_title,
            chunk_type="text",
            fusion_score=0.8,
        )

    # This test verifies only reranked chunks enter the provider prompt and citations use trusted metadata.
    def test_answers_from_reranked_context_and_returns_trusted_sources(self) -> None:
        candidate = self._chunk("candidate text")
        final_chunk = self._chunk("MRSA requires contact precautions.")
        retriever = FakeRetriever([candidate])
        reranker = FakeReranker([RerankedChunk(chunk=final_chunk, reranker_score=0.9)])
        llm_client = FakeLLMClient("Use contact precautions for MRSA.")
        service = HybridRagAnswerService(
            retriever,
            reranker,
            llm_client,
            candidate_limit=10,
            rerank_limit=3,
        )

        result = service.answer_with_details("What are MRSA precautions?", "nurse")

        self.assertEqual(result.answer, "Use contact precautions for MRSA.")
        self.assertEqual(retriever.calls, [("What are MRSA precautions?", "nurse", 10)])
        self.assertEqual(reranker.calls, [("What are MRSA precautions?", [candidate], 3)])
        self.assertEqual(llm_client.calls[0]["purpose"], LLMModelPurpose.ANSWER)
        self.assertIn("MRSA requires contact precautions.", llm_client.calls[0]["user_prompt"])
        self.assertNotIn("candidate text", llm_client.calls[0]["user_prompt"])
        self.assertEqual(result.sources[0].source_document, "infection_control.pdf")
        self.assertEqual(result.sources[0].section_title, "MRSA precautions")

    # This test verifies no candidate result avoids reranking and a provider request.
    def test_returns_no_source_answer_without_candidates(self) -> None:
        retriever = FakeRetriever([])
        reranker = FakeReranker([])
        llm_client = FakeLLMClient("unused")
        service = HybridRagAnswerService(retriever, reranker, llm_client)

        result = service.answer_with_details("What is the policy?", "nurse")

        self.assertEqual(result.sources, [])
        self.assertIn("could not find", result.answer)
        self.assertEqual(reranker.calls, [])
        self.assertEqual(llm_client.calls, [])

    # This test verifies source values are deduplicated using trusted metadata fields.
    def test_build_trusted_sources_deduplicates_same_document_section_and_collection(self) -> None:
        first = RerankedChunk(self._chunk("first"), 0.9)
        duplicate = RerankedChunk(self._chunk("second"), 0.8)
        different_section = RerankedChunk(self._chunk("third", "Hand hygiene"), 0.7)

        sources = build_trusted_sources([first, duplicate, different_section])

        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[1].section_title, "Hand hygiene")

    # This test verifies context labels preserve trusted metadata and the reranked text.
    def test_build_answer_context_includes_source_metadata_and_text(self) -> None:
        context = build_answer_context([RerankedChunk(self._chunk("MRSA text"), 0.9)])

        self.assertIn("document=infection_control.pdf", context)
        self.assertIn("section=MRSA precautions", context)
        self.assertIn("MRSA text", context)

    # This test verifies invalid user input stops the workflow before external dependencies are called.
    def test_rejects_empty_question_before_retrieval(self) -> None:
        retriever = FakeRetriever([])
        service = HybridRagAnswerService(retriever, FakeReranker([]), FakeLLMClient("unused"))

        with self.assertRaisesRegex(ValueError, "Question must be a non-empty"):
            service.answer_with_details("  ", "nurse")

        self.assertEqual(retriever.calls, [])
