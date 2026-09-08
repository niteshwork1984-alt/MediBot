"""Provider-independent contracts for cross-encoder reranking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.services.retrieval.hybrid_retriever import RetrievedChunk


# This immutable object preserves a trusted retrieved chunk with its separate cross-encoder relevance score.
@dataclass(frozen=True)
class RerankedChunk:
    """One retrieved chunk reordered by a cross-encoder relevance score."""

    chunk: RetrievedChunk
    reranker_score: float


# This abstract base class is the Java-like interface every reranking implementation must extend.
class Reranker(ABC):
    """Score question-and-chunk pairs jointly and return the best candidates."""

    # This abstract method defines the common reranking operation for local or hosted implementations.
    @abstractmethod
    def rerank(
        self,
        question: str,
        candidates: list[RetrievedChunk],
        limit: int,
    ) -> list[RerankedChunk]:
        """Return the highest-scoring candidate chunks for one non-empty question."""
        raise NotImplementedError
