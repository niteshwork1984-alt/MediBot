"""Unit tests for role-filtered LangChain/Qdrant hybrid retrieval."""

import unittest

from langchain_core.documents import Document

from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever


# This fake vector store records Hybrid RAG search arguments without querying Qdrant.
class FakeHybridVectorStore:
    """In-memory LangChain vector-store substitute for retrieval tests."""

    # This constructor stores deterministic results and an empty call record.
    def __init__(self, results: list[tuple[Document, float]]) -> None:
        """Create the fake store with prepared document-score pairs."""
        self.results = results
        self.calls: list[dict[str, object]] = []

    # This method records the hybrid search request and returns its prepared results.
    def similarity_search_with_score(self, question: str, **kwargs: object) -> list[tuple[Document, float]]:
        """Return prepared results without evaluating vectors or contacting Qdrant."""
        self.calls.append({"question": question, **kwargs})
        return self.results


# This test class verifies interface enforcement, RBAC filtering, and citation result mapping.
class QdrantHybridRetrieverTests(unittest.TestCase):
    # This helper makes one LangChain document shaped exactly like the ingestion payload metadata.
    def _document(self) -> Document:
        """Return one deterministic role-authorized retrieval result document."""
        return Document(
            page_content="MRSA isolation procedure text",
            metadata={
                "source_document": "nursing_procedures.pdf",
                "document_key": "nursing/nursing_procedures.pdf",
                "collection": "nursing",
                "access_roles": ["admin", "doctor", "nurse"],
                "section_title": "Isolation Procedure",
                "chunk_type": "text",
                "document_hash": "not-returned",
                "index_version": "v1",
            },
        )

    # This test verifies the Qdrant implementation explicitly extends the common retriever interface.
    def test_qdrant_retriever_implements_interface(self) -> None:
        store = FakeHybridVectorStore([])
        retriever = QdrantHybridRetriever("http://unused", "test_collection", vector_store=store)

        self.assertIsInstance(retriever, HybridRetriever)

    # This test verifies Qdrant receives the role condition before hybrid candidates are selected.
    def test_search_filters_qdrant_candidates_by_role_and_returns_citation_metadata(self) -> None:
        store = FakeHybridVectorStore([(self._document(), 0.75)])
        retriever = QdrantHybridRetriever("http://unused", "test_collection", vector_store=store)

        results = retriever.search("What is the MRSA procedure?", "nurse", limit=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_key, "nursing/nursing_procedures.pdf")
        self.assertEqual(results[0].section_title, "Isolation Procedure")
        self.assertEqual(results[0].fusion_score, 0.75)
        self.assertEqual(store.calls[0]["k"], 3)
        role_filter = store.calls[0]["filter"]
        condition = role_filter.must[0]  # type: ignore[union-attr]
        self.assertEqual(condition.key, "metadata.access_roles")
        self.assertEqual(condition.match.any, ["nurse"])  # type: ignore[union-attr]

    # This test verifies invalid roles are rejected before any vector-store query can occur.
    def test_search_rejects_unknown_role_before_querying(self) -> None:
        store = FakeHybridVectorStore([])
        retriever = QdrantHybridRetriever("http://unused", "test_collection", vector_store=store)

        with self.assertRaisesRegex(ValueError, "Unsupported role"):
            retriever.search("What is the MRSA procedure?", "visitor")

        self.assertEqual(store.calls, [])

    # This test verifies a retrieval limit must be a positive integer.
    def test_search_rejects_invalid_limit(self) -> None:
        store = FakeHybridVectorStore([])
        retriever = QdrantHybridRetriever("http://unused", "test_collection", vector_store=store)

        with self.assertRaisesRegex(ValueError, "Limit must be a positive integer"):
            retriever.search("What is the MRSA procedure?", "nurse", limit=0)

    # This test verifies a caller cannot request an unbounded hybrid candidate list.
    def test_search_rejects_limit_above_configured_maximum(self) -> None:
        store = FakeHybridVectorStore([])
        retriever = QdrantHybridRetriever("http://unused", "test_collection", vector_store=store)

        with self.assertRaisesRegex(ValueError, "HYBRID_RAG_MAX_LIMIT"):
            retriever.search("What is the MRSA procedure?", "nurse", limit=21)

        self.assertEqual(store.calls, [])
