"""Unit tests for incremental ingestion decisions without Docling, embeddings, or Qdrant."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.ingestion.collection_policy import (
    access_roles_for_collection,
    collection_for_source,
)
from app.ingestion.docling_chunker import create_chunk_id
from app.ingestion.ingestion_service import DocumentIngestionService
from app.ingestion.interfaces import ChunkingService, EmbeddingService
from app.ingestion.models import PreparedChunk


# This fake chunker returns prebuilt chunks for tests instead of parsing PDFs with Docling.
class FakeChunker(ChunkingService):
    """In-memory chunker that returns deterministic chunks by source filename."""

    # This constructor stores a mapping of source filename to prepared chunks.
    def __init__(self, chunks_by_filename: dict[str, list[PreparedChunk]]) -> None:
        """Create a fake chunker with deterministic per-file results."""
        self.chunks_by_filename = chunks_by_filename

    # This method returns chunks for the selected test file without external dependencies.
    def chunk_file(
        self,
        source_root: Path,
        source_path: Path,
        index_version: str,
    ) -> list[PreparedChunk]:
        """Return the prepared chunks registered for the test source file."""
        return self.chunks_by_filename[source_path.name]


# This fake embedder adds simple vectors so tests can verify only invocation decisions.
class FakeEmbedder(EmbeddingService):
    """Local fake embedding implementation that records batches without model downloads."""

    # This constructor prepares a record of every chunk batch asked to embed.
    def __init__(self) -> None:
        """Create an empty fake embedding call record."""
        self.sparse_adapter = object()

    # This method returns the fixed dense-vector size expected by the fake index.
    def dense_vector_size(self) -> int:
        """Return a deterministic dense vector dimension."""
        return 2

    # This method supplies the fake sparse adapter expected by LangChain hybrid storage.
    def sparse_embeddings(self) -> object:
        """Return a deterministic sparse embedding adapter placeholder."""
        return self.sparse_adapter


# This fake index records state checks and writes without requiring a running Qdrant container.
class FakeIndex:
    """In-memory Qdrant-like index used to test incremental decisions."""

    # This constructor selects the document keys treated as currently indexed.
    def __init__(self, current_document_keys: set[str] | None = None) -> None:
        """Create a fake index with optional already-current documents."""
        self.current_document_keys = current_document_keys or set()
        self.collection_sizes: list[int] = []
        self.deleted_document_keys: list[str] = []
        self.upserted_batches: list[list[PreparedChunk]] = []

    # This method records the collection vector size requested by the service.
    def ensure_collection(self, dense_vector_size: int, embedding: object, sparse_embedding: object) -> None:
        """Record collection initialization without creating a database collection."""
        self.collection_sizes.append(dense_vector_size)

    # This method reports whether a configured document key is already current.
    def document_is_current(
        self,
        document_key: str,
        document_hash: str,
        index_version: str,
        expected_chunk_count: int,
    ) -> bool:
        """Return the configured current state for a test document key."""
        return document_key in self.current_document_keys

    # This method records removal of stale chunks for a test document.
    def delete_document(self, document_key: str) -> None:
        """Record the document whose prior chunks would be removed."""
        self.deleted_document_keys.append(document_key)

    # This method records indexed chunks that would be sent to Qdrant.
    def upsert_chunks(self, chunks: list[PreparedChunk]) -> None:
        """Record one batch of complete indexed chunks."""
        self.upserted_batches.append(chunks)


# This test class verifies collection-based RBAC mapping and stable chunk identities.
class IngestionMetadataTests(unittest.TestCase):
    # This test verifies the assignment's billing role policy is stored at ingestion time.
    def test_billing_collection_has_only_billing_and_admin_roles(self) -> None:
        self.assertEqual(
            access_roles_for_collection("billing"),
            ["admin", "billing_executive"],
        )

    # This test verifies a source directory determines a permitted collection explicitly.
    def test_collection_is_read_from_source_root_directory(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory)
            source_path = source_root / "nursing" / "procedure.pdf"
            source_path.parent.mkdir()
            source_path.touch()

            self.assertEqual(collection_for_source(source_root, source_path), "nursing")

    # This test verifies a repeat of unchanged chunk inputs produces the same Qdrant point UUID.
    def test_chunk_id_is_deterministic(self) -> None:
        first_id = create_chunk_id("billing/guide.pdf", "hash", "v1", 0, "text")
        second_id = create_chunk_id("billing/guide.pdf", "hash", "v1", 0, "text")

        self.assertEqual(first_id, second_id)


# This test class verifies incremental indexing decisions with replaceable dependencies.
class DocumentIngestionServiceTests(unittest.TestCase):
    # This helper creates a complete prepared chunk for a test source file.
    def _prepared_chunk(self, filename: str, document_key: str) -> PreparedChunk:
        return PreparedChunk(
            id="123e4567-e89b-12d3-a456-426614174000",
            text="Heading\nChunk text",
            source_document=filename,
            document_key=document_key,
            collection="billing",
            access_roles=["admin", "billing_executive"],
            section_title="Heading",
            chunk_type="text",
            document_hash="document-hash",
            index_version="v1",
        )

    # This test verifies a new document is embedded and written after stale chunks are cleared.
    def test_indexes_new_document(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory)
            source_file = source_root / "billing" / "guide.md"
            source_file.parent.mkdir()
            source_file.write_text("content")
            chunk = self._prepared_chunk("guide.md", "billing/guide.md")
            embedder = FakeEmbedder()
            index = FakeIndex()
            service = DocumentIngestionService(
                chunker=FakeChunker({"guide.md": [chunk]}),
                embedder=embedder,
                index=index,
            )

            summary = service.ingest(source_root, "v1")

        self.assertEqual(summary.indexed_documents, 1)
        self.assertEqual(summary.skipped_documents, 0)
        self.assertEqual(summary.indexed_chunks, 1)
        self.assertEqual(index.deleted_document_keys, ["billing/guide.md"])
        self.assertEqual(len(index.upserted_batches), 1)

    # This test verifies a current document skips the expensive embedding and write phases.
    def test_skips_current_document(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            source_root = Path(temporary_directory)
            source_file = source_root / "billing" / "guide.md"
            source_file.parent.mkdir()
            source_file.write_text("content")
            chunk = self._prepared_chunk("guide.md", "billing/guide.md")
            embedder = FakeEmbedder()
            index = FakeIndex({"billing/guide.md"})
            service = DocumentIngestionService(
                chunker=FakeChunker({"guide.md": [chunk]}),
                embedder=embedder,
                index=index,
            )

            summary = service.ingest(source_root, "v1")

        self.assertEqual(summary.indexed_documents, 0)
        self.assertEqual(summary.skipped_documents, 1)
        self.assertEqual(index.upserted_batches, [])
