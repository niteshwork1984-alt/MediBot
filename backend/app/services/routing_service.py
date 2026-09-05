"""Question-routing strategies for MediBot retrieval workflows."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from app.core.permissions import can_use_sql_rag


RetrievalType = Literal["sql_rag", "hybrid_rag"]


# This abstract base class is the Java-like interface every router must extend.
class RoutingStrategy(ABC):
    """Java-like abstract interface for every routing implementation."""

    # This abstract method defines the common routing operation for all routers.
    @abstractmethod
    def route(self, question: str, role: str) -> RetrievalType:
        """Return the retrieval workflow appropriate for a question and role."""
        raise NotImplementedError


# This concrete implementation routes with simple keyword rules in phase one.
@dataclass(frozen=True)
class KeywordRoutingStrategy(RoutingStrategy):
    """Route analytical database questions using transparent phrase matching."""

    analytical_phrases: frozenset[str] = frozenset(
        {
            "how many",
            "count",
            "total",
            "sum",
            "average",
            "most",
            "least",
            "highest",
            "lowest",
            "by department",
            "by category",
            "last month",
        }
    )
    database_terms: frozenset[str] = frozenset(
        {
            "claim",
            "claims",
            "maintenance ticket",
            "maintenance tickets",
            "equipment",
            "approved amount",
            "claimed amount",
        }
    )

    # This method applies authorization and keyword checks to choose one route.
    def route(self, question: str, role: str) -> RetrievalType:
        """Route only authorised analytical database questions to SQL RAG."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if not can_use_sql_rag(role):
            return "hybrid_rag"

        normalized_question = question.lower()
        is_analytical = any(
            phrase in normalized_question for phrase in self.analytical_phrases
        )
        mentions_database_data = any(
            term in normalized_question for term in self.database_terms
        )

        return "sql_rag" if is_analytical and mentions_database_data else "hybrid_rag"


# This public helper accepts any approved router, making future replacements easy.
def route_question(
    question: str,
    role: str,
    strategy: RoutingStrategy | None = None,
) -> RetrievalType:
    """Route a question through the selected strategy.

    The default is intentionally simple. A semantic or LLM strategy can later
    be supplied without changing callers of this function.
    """
    selected_strategy = strategy if strategy is not None else KeywordRoutingStrategy()
    if not isinstance(selected_strategy, RoutingStrategy):
        raise TypeError("strategy must extend RoutingStrategy.")

    return selected_strategy.route(question, role)
