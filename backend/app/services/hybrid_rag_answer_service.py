"""Hybrid RAG orchestration: retrieve, rerank, generate an answer, and preserve trusted sources."""

import logging
from dataclasses import dataclass

from app.core.config import (
    HYBRID_RAG_CANDIDATE_LIMIT,
    HYBRID_RAG_RERANK_LIMIT,
    LLMModelPurpose,
)
from app.services.llm.llm_client import LLMClient, create_llm_client
from app.services.reranking.reranker import RerankedChunk, Reranker
from app.services.retrieval.hybrid_retriever import HybridRetriever


LOGGER = logging.getLogger(__name__)


# This immutable value exposes a citation built from trusted Qdrant payload metadata, not LLM output.
@dataclass(frozen=True)
class HybridRagSource:
    """One safe citation for a chunk used as answer context."""

    source_document: str
    section_title: str
    collection: str


# This immutable value keeps the final answer separate from its independently produced citations.
@dataclass(frozen=True)
class HybridRagResult:
    """The natural-language answer and trusted sources for one Hybrid RAG request."""

    answer: str
    sources: list[HybridRagSource]


# This function creates source objects from reranked chunk metadata and removes duplicate citations.
def build_trusted_sources(reranked_chunks: list[RerankedChunk]) -> list[HybridRagSource]:
    """Return unique citations in the same order as the final reranked context."""
    sources: list[HybridRagSource] = []
    seen_sources: set[tuple[str, str, str]] = set()
    for reranked_chunk in reranked_chunks:
        chunk = reranked_chunk.chunk
        source_key = (chunk.source_document, chunk.section_title, chunk.collection)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(HybridRagSource(*source_key))
    return sources


# This function formats only approved reranked chunks into the LLM's context window.
def build_answer_context(reranked_chunks: list[RerankedChunk]) -> str:
    """Return labelled source text that the answer model may use as its sole evidence."""
    context_parts: list[str] = []
    for position, reranked_chunk in enumerate(reranked_chunks, start=1):
        chunk = reranked_chunk.chunk
        context_parts.append(
            f"[Source {position}] document={chunk.source_document}; "
            f"section={chunk.section_title}; collection={chunk.collection}\n{chunk.text}"
        )
    return "\n\n".join(context_parts)


# This service coordinates role-filtered retrieval, reranking, and provider-independent answer generation.
class HybridRagAnswerService:
    """Answer document questions using only role-authorized, reranked document context."""

    # This constructor accepts dependencies so unit tests avoid Qdrant, a reranker model, and provider calls.
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm_client: LLMClient,
        candidate_limit: int = HYBRID_RAG_CANDIDATE_LIMIT,
        rerank_limit: int = HYBRID_RAG_RERANK_LIMIT,
    ) -> None:
        """Store the three stages and validate the configured result limits."""
        if candidate_limit < 1:
            raise ValueError("Candidate limit must be a positive integer.")
        if rerank_limit < 1 or rerank_limit > candidate_limit:
            raise ValueError("Rerank limit must be positive and cannot exceed candidate limit.")
        self._retriever = retriever
        self._reranker = reranker
        self._llm_client = llm_client
        self._candidate_limit = candidate_limit
        self._rerank_limit = rerank_limit

    # This method performs the complete Hybrid RAG flow for a supplied authenticated role.
    def answer_with_details(self, question: str, role: str) -> HybridRagResult:
        """Return an answer and metadata-derived citations for one non-empty question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Role must be a non-empty string.")

        candidates = self._retriever.search(question, role, self._candidate_limit)
        LOGGER.info("Hybrid RAG retrieved authorized_candidates=%s role=%s", len(candidates), role)
        if not candidates:
            return HybridRagResult(
                answer="I could not find an authorized document that answers this question.",
                sources=[],
            )

        reranked_chunks = self._reranker.rerank(question, candidates, self._rerank_limit)
        LOGGER.info("Hybrid RAG reranked final_context_chunks=%s role=%s", len(reranked_chunks), role)
        if not reranked_chunks:
            return HybridRagResult(
                answer="I could not find an authorized document that answers this question.",
                sources=[],
            )

        answer = self._llm_client.generate(
            system_prompt=(
                "You are MediBot. Answer only from the supplied authorized document context. "
                "Do not invent facts or sources. If the context does not answer the question, "
                "say that the provided documents do not contain the answer."
            ),
            user_prompt=(
                f"Question: {question}\n\n"
                f"Authorized document context:\n{build_answer_context(reranked_chunks)}"
            ),
            purpose=LLMModelPurpose.ANSWER,
        )
        sources = build_trusted_sources(reranked_chunks)
        LOGGER.info("Hybrid RAG answer generated source_count=%s role=%s", len(sources), role)
        return HybridRagResult(answer=answer, sources=sources)

    # This convenience method returns only the answer text when source details are not needed.
    def answer(self, question: str, role: str) -> str:
        """Return the natural-language Hybrid RAG answer for one authorized role."""
        return self.answer_with_details(question, role).answer


# This function creates the configured provider client for a future API composition root.
def create_hybrid_rag_answer_service(
    retriever: HybridRetriever,
    reranker: Reranker,
) -> HybridRagAnswerService:
    """Create a Hybrid RAG answer service with the LLM provider selected by configuration."""
    return HybridRagAnswerService(retriever, reranker, create_llm_client())
