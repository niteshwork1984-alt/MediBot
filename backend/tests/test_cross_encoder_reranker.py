"""Unit tests for cross-encoder reranking without model downloads or Qdrant access."""

import unittest

from app.services.reranking.fastembed_cross_encoder_reranker import FastEmbedCrossEncoderReranker
from app.services.reranking.reranker import Reranker
from app.services.retrieval.hybrid_retriever import RetrievedChunk


# This fake cross-encoder returns prepared relevance scores in the same order as received candidate documents.
class FakeCrossEncoder:
    """In-memory cross-encoder replacement used to test sorting and limit behavior."""

    # This constructor stores prepared scores and records each reranking request.
    def __init__(self, scores: list[float]) -> None:
        """Create a fake cross-encoder with deterministic candidate scores."""
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    # This method records question-document pairs and yields deterministic scores without model inference.
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return the prepared score list for one candidate batch."""
        self.calls.append((query, documents))
        return self.scores


# This test class verifies interface enforcement, joint scoring order, and input validation.
class FastEmbedCrossEncoderRerankerTests(unittest.TestCase):
    # This helper creates a retrieval candidate with distinct text and RRF fusion score for sorting checks.
    def _candidate(self, text: str, fusion_score: float) -> RetrievedChunk:
        """Return one deterministic retrieved chunk for reranking tests."""
        return RetrievedChunk(
            text=text,
            source_document="source.pdf",
            document_key="nursing/source.pdf",
            collection="nursing",
            section_title="Procedure",
            chunk_type="text",
            fusion_score=fusion_score,
        )

    # This test verifies the concrete FastEmbed implementation explicitly extends the shared reranking interface.
    def test_fastembed_reranker_implements_interface(self) -> None:
        reranker = FastEmbedCrossEncoderReranker(cross_encoder=FakeCrossEncoder([]))

        self.assertIsInstance(reranker, Reranker)

    # This test verifies cross-encoder scores reorder candidates and keep only the requested top results.
    def test_rerank_sorts_by_cross_encoder_score_before_fusion_score(self) -> None:
        candidates = [
            self._candidate("first candidate", 0.9),
            self._candidate("second candidate", 0.1),
            self._candidate("third candidate", 0.5),
        ]
        encoder = FakeCrossEncoder([0.2, 0.9, 0.6])
        reranker = FastEmbedCrossEncoderReranker(cross_encoder=encoder)

        reranked_chunks = reranker.rerank("MRSA procedure", candidates, limit=2)

        self.assertEqual([result.chunk.text for result in reranked_chunks], ["second candidate", "third candidate"])
        self.assertEqual([result.reranker_score for result in reranked_chunks], [0.9, 0.6])
        self.assertEqual(encoder.calls, [("MRSA procedure", [candidate.text for candidate in candidates])])

    # This test verifies equal cross-encoder scores fall back to the existing RRF fusion order deterministically.
    def test_rerank_uses_fusion_score_to_break_equal_cross_encoder_scores(self) -> None:
        candidates = [
            self._candidate("lower fusion", 0.1),
            self._candidate("higher fusion", 0.9),
        ]
        reranker = FastEmbedCrossEncoderReranker(cross_encoder=FakeCrossEncoder([0.5, 0.5]))

        reranked_chunks = reranker.rerank("MRSA procedure", candidates, limit=2)

        self.assertEqual(reranked_chunks[0].chunk.text, "higher fusion")

    # This test verifies empty candidates do not invoke the cross-encoder or produce results.
    def test_rerank_returns_empty_list_without_candidates(self) -> None:
        encoder = FakeCrossEncoder([])
        reranker = FastEmbedCrossEncoderReranker(cross_encoder=encoder)

        self.assertEqual(reranker.rerank("MRSA procedure", [], limit=3), [])
        self.assertEqual(encoder.calls, [])

    # This test verifies a model scoring mismatch fails instead of silently pairing incorrect chunks and scores.
    def test_rerank_rejects_score_count_mismatch(self) -> None:
        reranker = FastEmbedCrossEncoderReranker(cross_encoder=FakeCrossEncoder([0.5]))

        with self.assertRaisesRegex(RuntimeError, "score count"):
            reranker.rerank(
                "MRSA procedure",
                [self._candidate("first", 0.1), self._candidate("second", 0.2)],
                limit=2,
            )
