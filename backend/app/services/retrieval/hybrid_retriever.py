"""Provider-independent contracts for role-filtered hybrid document retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


# This immutable object is the safe structured result returned before any LLM answer generation.
@dataclass(frozen=True)
class RetrievedChunk:
    """One role-authorized hybrid-search result with citation metadata."""

    text: str
    source_document: str
    document_key: str
    collection: str
    section_title: str
    chunk_type: str
    fusion_score: float


# This abstract base class is the Java-like interface every hybrid retriever must extend.
class HybridRetriever(ABC):
    """Retrieve only documents authorized for the supplied role."""

    # This abstract method defines the common retrieval operation for all vector-store implementations.
    @abstractmethod
    def search(self, question: str, role: str, limit: int = 5) -> list[RetrievedChunk]:
        """Return the best authorized chunks for one non-empty user question."""
        raise NotImplementedError
