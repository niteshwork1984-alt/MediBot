"""Qdrant collection creation, incremental-state checks, and document point writes."""

import logging
from typing import Any

from app.ingestion.models import PreparedChunk


LOGGER = logging.getLogger(__name__)


# This class owns the write-only ingestion operations against one Qdrant collection.
class QdrantDocumentIndex:
    """Persist MediBot document points and inspect existing index state in Qdrant."""

    # This constructor connects to the configured Qdrant server without creating data yet.
    def __init__(self, url: str, collection_name: str) -> None:
        """Create a Qdrant client for one versioned document collection."""
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RuntimeError(
                "Qdrant client is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error

        self._client = QdrantClient(url=url)
        self.collection_name = collection_name
        self._vector_store: Any | None = None

    # This helper imports Qdrant models only when a database operation is requested.
    def _models(self) -> Any:
        """Return the official Qdrant model classes used to build requests."""
        from qdrant_client import models

        return models

    # This method creates the named dense and sparse vector schema once for a new collection.
    def ensure_collection(self, dense_vector_size: int, embedding: Any, sparse_embedding: Any) -> None:
        """Create the collection and payload indexes needed by ingestion and later RBAC retrieval."""
        models = self._models()
        if not self._client.collection_exists(self.collection_name):
            LOGGER.info("Creating Qdrant collection collection=%s dense_vector_size=%d", self.collection_name, dense_vector_size)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            for field_name in (
                "metadata.document_key", "metadata.collection", "metadata.access_roles",
                "metadata.document_hash", "metadata.index_version",
            ):
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            LOGGER.info("Created Qdrant collection and payload indexes collection=%s", self.collection_name)
        else:
            LOGGER.info("Using existing Qdrant collection collection=%s", self.collection_name)
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        self._vector_store = QdrantVectorStore(
            client=self._client, collection_name=self.collection_name, embedding=embedding,
            sparse_embedding=sparse_embedding, retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense", sparse_vector_name="sparse", content_payload_key="text",
        )

    # This helper builds a filter that identifies every chunk belonging to one indexed file version.
    def _document_filter(
        self,
        document_key: str,
        document_hash: str | None = None,
        index_version: str | None = None,
    ) -> Any:
        """Return a Qdrant payload filter for a document and optional content/index version."""
        models = self._models()
        conditions = [
            models.FieldCondition(
                key="metadata.document_key",
                match=models.MatchValue(value=document_key),
            )
        ]
        if document_hash is not None:
            conditions.append(
                models.FieldCondition(
                    key="metadata.document_hash",
                    match=models.MatchValue(value=document_hash),
                )
            )
        if index_version is not None:
            conditions.append(
                models.FieldCondition(
                    key="metadata.index_version",
                    match=models.MatchValue(value=index_version),
                )
            )
        return models.Filter(must=conditions)

    # This method checks whether Qdrant already contains precisely the expected chunks for one file version.
    def document_is_current(
        self,
        document_key: str,
        document_hash: str,
        index_version: str,
        expected_chunk_count: int,
    ) -> bool:
        """Return true only when the stored current-version point count matches expectation."""
        result = self._client.count(
            collection_name=self.collection_name,
            count_filter=self._document_filter(
                document_key,
                document_hash,
                index_version,
            ),
            exact=True,
        )
        LOGGER.debug(
            "Checked document index state document_key=%s expected_chunks=%d stored_chunks=%d index_version=%s",
            document_key,
            expected_chunk_count,
            result.count,
            index_version,
        )
        return result.count == expected_chunk_count

    # This method removes stale chunks for one source document before the replacement chunks are written.
    def delete_document(self, document_key: str) -> None:
        """Delete every existing point for one source file across old hashes and index versions."""
        models = self._models()
        LOGGER.info("Deleting existing Qdrant points document_key=%s collection=%s", document_key, self.collection_name)
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=self._document_filter(document_key)
            ),
            wait=True,
        )

    # This method upserts complete chunks in a batch after their vectors and payload have been prepared.
    def upsert_chunks(self, chunks: list[PreparedChunk]) -> None:
        """Use LangChain Documents and QdrantVectorStore to generate and write hybrid vectors."""
        if not chunks:
            return
        if self._vector_store is None:
            raise RuntimeError("Collection must be initialized before LangChain ingestion.")
        LOGGER.info("Writing hybrid Qdrant points collection=%s document_key=%s chunks=%d", self.collection_name, chunks[0].document_key, len(chunks))
        from langchain_core.documents import Document
        documents = [Document(page_content=chunk.text, metadata={
            "source_document": chunk.source_document, "document_key": chunk.document_key,
            "collection": chunk.collection, "access_roles": chunk.access_roles,
            "section_title": chunk.section_title, "chunk_type": chunk.chunk_type,
            "document_hash": chunk.document_hash, "index_version": chunk.index_version,
        }) for chunk in chunks]
        self._vector_store.add_documents(documents, ids=[chunk.id for chunk in chunks])
        LOGGER.info("Wrote hybrid Qdrant points collection=%s document_key=%s chunks=%d", self.collection_name, chunks[0].document_key, len(chunks))
