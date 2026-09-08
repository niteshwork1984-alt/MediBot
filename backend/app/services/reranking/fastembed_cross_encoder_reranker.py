"""FastEmbed cross-encoder reranking for role-authorized Hybrid RAG candidates."""

import logging
import time
from typing import Any

from app.core.config import RERANKER_MODEL
from app.services.reranking.reranker import RerankedChunk, Reranker
from app.services.retrieval.hybrid_retriever import RetrievedChunk


LOGGER = logging.getLogger(__name__)


# This concrete implementation scores the full question-and-chunk pair with a local FastEmbed cross-encoder.
class FastEmbedCrossEncoderReranker(Reranker):
    """Rerank Hybrid RAG candidates using a jointly evaluated local cross-encoder model."""

    # This constructor accepts a replacement model so tests avoid model downloads and production can change providers.
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        cross_encoder: Any | None = None,
    ) -> None:
        """Create a reranker with one configured local cross-encoder model."""
        self.model_name = model_name
        if cross_encoder is not None:
            self._cross_encoder = cross_encoder
            return
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "FastEmbed is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error
        self._cross_encoder = TextCrossEncoder(model_name=model_name, lazy_load=True)
        LOGGER.info("Configured cross-encoder reranker model=%s", model_name)

    # This method evaluates each question-and-chunk pair, then retains the highest cross-encoder scores.
    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        limit: int,
    ) -> list[RerankedChunk]:
        """Return up to limit candidates sorted by cross-encoder relevance, not fusion score."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Limit must be a positive integer.")
        if not candidates:
            return []

        reranking_started_at = time.monotonic()
        LOGGER.info(
            "Starting cross-encoder reranking model=%s candidate_count=%d result_limit=%d",
            self.model_name,
            len(candidates),
            limit,
        )
        scores = list(
            self._cross_encoder.rerank(
                query=question,
                documents=[candidate.text for candidate in candidates],
            )
        )
        if len(scores) != len(candidates):
            raise RuntimeError(
                "Cross-encoder returned a score count different from the candidate count."
            )
        reranked_chunks = [
            RerankedChunk(chunk=candidate, reranker_score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked_chunks.sort(
            key=lambda result: (result.reranker_score, result.chunk.fusion_score),
            reverse=True,
        )
        selected_chunks = reranked_chunks[:limit]
        LOGGER.info(
            "Finished cross-encoder reranking model=%s candidate_count=%d selected_count=%d duration_ms=%d",
            self.model_name,
            len(candidates),
            len(selected_chunks),
            (time.monotonic() - reranking_started_at) * 1000,
        )
        return selected_chunks
