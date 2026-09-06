"""Data objects passed between parsing, embedding, and Qdrant ingestion stages."""

from dataclasses import dataclass


# This immutable object represents a Docling chunk before embedding vectors are generated.
@dataclass(frozen=True)
class PreparedChunk:
    """A heading-aware document chunk with source and RBAC metadata."""

    id: str
    text: str
    source_document: str
    document_key: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str
    document_hash: str
    index_version: str


# This immutable object represents the complete Qdrant point contract after embedding.
@dataclass(frozen=True)
class IndexedChunk:
    """One Qdrant point with dense and sparse vectors plus required payload fields."""

    id: str
    text: str
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    source_document: str
    document_key: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str
    document_hash: str
    index_version: str

    # This method exposes the non-vector fields stored as Qdrant payload metadata.
    def payload(self) -> dict[str, object]:
        """Return the required chunk metadata in a Qdrant-compatible dictionary."""
        return {
            "text": self.text,
            "source_document": self.source_document,
            "document_key": self.document_key,
            "collection": self.collection,
            "access_roles": self.access_roles,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "document_hash": self.document_hash,
            "index_version": self.index_version,
        }


# This immutable object reports the result of an explicit ingestion command.
@dataclass(frozen=True)
class IngestionSummary:
    """Counts describing one ingestion run for terminal feedback and tests."""

    indexed_documents: int
    skipped_documents: int
    indexed_chunks: int
    pruned_documents: int
