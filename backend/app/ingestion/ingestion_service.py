"""Incremental document-ingestion orchestration without retrieval or FastAPI."""

import logging
import time
from pathlib import Path

from app.ingestion.docling_chunker import DoclingHybridChunker
from app.ingestion.embedding_service import FastEmbedEmbeddingService
from app.ingestion.interfaces import ChunkingService, EmbeddingService
from app.ingestion.models import IngestionSummary, PreparedChunk
from app.ingestion.qdrant_index import QdrantDocumentIndex


LOGGER = logging.getLogger(__name__)


# This class orchestrates explicit incremental indexing without implementing any retrieval behavior.
class DocumentIngestionService:
    """Parse, compare, embed, and upsert only documents needing an index update."""

    # This constructor accepts replaceable components so unit tests avoid Docling, models, and Qdrant.
    def __init__(
        self,
        chunker: ChunkingService | None = None,
        embedder: EmbeddingService | None = None,
        index: QdrantDocumentIndex | None = None,
    ) -> None:
        """Create an ingestion service with supplied or production dependencies."""
        self._chunker = chunker or DoclingHybridChunker()
        self._embedder = embedder or FastEmbedEmbeddingService()
        if index is None:
            raise ValueError("DocumentIngestionService requires a QdrantDocumentIndex.")
        self._index = index

    # This method finds only supported document types beneath assignment collection directories.
    def _source_files(self, source_root: Path) -> list[Path]:
        """Return PDF and Markdown source files in stable path order."""
        supported_extensions = {".pdf", ".md", ".markdown"}
        return sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        )

    # This method runs one incremental pass and indexes only documents whose stored state differs.
    def ingest(
        self,
        source_root: Path,
        index_version: str,
        force_reindex: bool = False,
    ) -> IngestionSummary:
        """Synchronize supported source documents into the configured Qdrant collection."""
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source_root}")

        source_files = self._source_files(source_root)
        run_started_at = time.monotonic()
        LOGGER.info(
            "Starting ingestion run source_root=%s documents_found=%d index_version=%s force_reindex=%s",
            source_root,
            len(source_files),
            index_version,
            force_reindex,
        )
        self._index.ensure_collection(
            self._embedder.dense_vector_size(), self._embedder, self._embedder.sparse_embeddings()
        )
        indexed_documents = 0
        skipped_documents = 0
        indexed_chunks = 0

        for source_path in source_files:
            document_started_at = time.monotonic()
            LOGGER.info("Processing source_document=%s", source_path.relative_to(source_root))
            try:
                prepared_chunks = self._chunker.chunk_file(
                    source_root,
                    source_path,
                    index_version,
                )
            except Exception:
                LOGGER.exception(
                    "Document parsing failed source_document=%s index_version=%s",
                    source_path.relative_to(source_root),
                    index_version,
                )
                raise
            if not prepared_chunks:
                raise ValueError(f"Docling produced no chunks for: {source_path}")

            first_chunk = prepared_chunks[0]
            if not force_reindex and self._index.document_is_current(
                first_chunk.document_key,
                first_chunk.document_hash,
                index_version,
                len(prepared_chunks),
            ):
                skipped_documents += 1
                LOGGER.info(
                    "Skipped current document document_key=%s chunks=%d duration_ms=%d",
                    first_chunk.document_key,
                    len(prepared_chunks),
                    (time.monotonic() - document_started_at) * 1000,
                )
                continue

            LOGGER.info(
                "Replacing document document_key=%s chunks=%d index_version=%s",
                first_chunk.document_key,
                len(prepared_chunks),
                index_version,
            )
            try:
                self._index.delete_document(first_chunk.document_key)
                self._index.upsert_chunks(prepared_chunks)
            except Exception:
                LOGGER.exception(
                    "Document indexing failed document_key=%s chunks=%d index_version=%s",
                    first_chunk.document_key,
                    len(prepared_chunks),
                    index_version,
                )
                raise
            indexed_documents += 1
            indexed_chunks += len(prepared_chunks)
            LOGGER.info(
                "Indexed document document_key=%s chunks=%d duration_ms=%d",
                first_chunk.document_key,
                len(prepared_chunks),
                (time.monotonic() - document_started_at) * 1000,
            )

        summary = IngestionSummary(
            indexed_documents=indexed_documents,
            skipped_documents=skipped_documents,
            indexed_chunks=indexed_chunks,
            pruned_documents=0,
        )
        LOGGER.info(
            "Finished ingestion run indexed_documents=%d skipped_documents=%d indexed_chunks=%d duration_ms=%d",
            summary.indexed_documents,
            summary.skipped_documents,
            summary.indexed_chunks,
            (time.monotonic() - run_started_at) * 1000,
        )
        return summary
