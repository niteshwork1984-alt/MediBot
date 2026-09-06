"""LangChain Qdrant hybrid retrieval with Qdrant-enforced role filtering."""

import logging
import time
from typing import Any

from app.core.config import HYBRID_RAG_MAX_LIMIT
from app.ingestion.collection_policy import ALL_ROLES
from app.ingestion.embedding_service import FastEmbedEmbeddingService
from app.services.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk


LOGGER = logging.getLogger(__name__)


# This concrete implementation runs dense and BM25 search together against one Qdrant collection.
class QdrantHybridRetriever(HybridRetriever):
    """Retrieve role-authorized chunks using LangChain HYBRID mode and reciprocal-rank fusion."""

    # This constructor accepts an optional vector store so unit tests avoid Qdrant and model downloads.
    def __init__(
        self,
        url: str,
        collection_name: str,
        embedding: FastEmbedEmbeddingService | None = None,
        vector_store: Any | None = None,
    ) -> None:
        """Connect to an existing hybrid Qdrant collection without changing its data."""
        self.collection_name = collection_name
        if vector_store is not None:
            self._vector_store = vector_store
            return

        try:
            from langchain_qdrant import QdrantVectorStore, RetrievalMode
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError(
                "Qdrant and LangChain dependencies are not installed. "
                "Run: python3 -m pip install -r requirements.txt"
            ) from error

        selected_embedding = embedding or FastEmbedEmbeddingService()
        self._vector_store = QdrantVectorStore(
            client=QdrantClient(url=url),
            collection_name=collection_name,
            embedding=selected_embedding,
            sparse_embedding=selected_embedding.sparse_embeddings(),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="text",
        )

    # This helper builds a Qdrant filter evaluated before either dense or sparse candidates are returned.
    def _role_filter(self, role: str) -> Any:
        """Return a Qdrant payload filter that permits only chunks containing the role."""
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.access_roles",
                    match=models.MatchAny(any=[role]),
                )
            ]
        )

    # This helper converts LangChain documents into the result contract used by callers and the future answer layer.
    def _to_retrieved_chunk(self, document: Any, fusion_score: float) -> RetrievedChunk:
        """Map one LangChain document and RRF fusion score into a citation-ready result."""
        metadata = document.metadata
        return RetrievedChunk(
            text=document.page_content,
            source_document=str(metadata["source_document"]),
            document_key=str(metadata["document_key"]),
            collection=str(metadata["collection"]),
            section_title=str(metadata["section_title"]),
            chunk_type=str(metadata["chunk_type"]),
            fusion_score=float(fusion_score),
        )

    # This method runs both configured vector searches and combines them with explicit reciprocal-rank fusion.
    def search(self, question: str, role: str, limit: int = 5) -> list[RetrievedChunk]:
        """Return up to limit role-authorized chunks without sending the question to an LLM."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        if role not in ALL_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("Limit must be a positive integer.")
        if limit > HYBRID_RAG_MAX_LIMIT:
            raise ValueError(
                f"Limit cannot exceed HYBRID_RAG_MAX_LIMIT ({HYBRID_RAG_MAX_LIMIT})."
            )

        from qdrant_client import models

        retrieval_started_at = time.monotonic()
        LOGGER.info(
            "Starting hybrid retrieval collection=%s role=%s limit=%d",
            self.collection_name,
            role,
            limit,
        )
        results = self._vector_store.similarity_search_with_score(
            question,
            k=limit,
            filter=self._role_filter(role),
            hybrid_fusion=models.FusionQuery(fusion=models.Fusion.RRF),
        )
        retrieved_chunks = [
            self._to_retrieved_chunk(document, fusion_score)
            for document, fusion_score in results
        ]
        LOGGER.info(
            "Finished hybrid retrieval collection=%s role=%s result_count=%d duration_ms=%d",
            self.collection_name,
            role,
            len(retrieved_chunks),
            (time.monotonic() - retrieval_started_at) * 1000,
        )
        return retrieved_chunks
