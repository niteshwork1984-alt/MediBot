"""Explicit terminal command for Docling-to-Qdrant ingestion; not an API endpoint."""

import argparse
import logging
import os
from pathlib import Path

from app.core.config import INDEX_VERSION, QDRANT_COLLECTION, QDRANT_URL
from app.ingestion.embedding_service import FastEmbedEmbeddingService
from app.ingestion.ingestion_service import DocumentIngestionService
from app.ingestion.qdrant_index import QdrantDocumentIndex


LOGGER = logging.getLogger(__name__)


# This function makes Python network clients trust certificates configured in the operating system.
def _use_system_trust_store() -> None:
    """Enable the operating system certificate store before model downloads begin."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


# This function reads explicit terminal settings for an ingestion run.
def _parse_arguments() -> argparse.Namespace:
    """Return source, collection, and index-version values supplied by the operator."""
    parser = argparse.ArgumentParser(
        description="Parse MediBot documents with Docling and index vectors in Qdrant."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing general, clinical, nursing, billing, and equipment folders.",
    )
    parser.add_argument(
        "--index-version",
        default=os.getenv("INDEX_VERSION", INDEX_VERSION),
        help="Index contract version; use a new value when models or chunking change.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION),
        help="Versioned Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", QDRANT_URL),
        help="Qdrant REST endpoint.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Replace every discovered document even when its stored hash and version match.",
    )
    return parser.parse_args()


# This function configures concise terminal logs for this explicit command only.
def _configure_logging() -> None:
    """Configure timestamped operational logs without logging document contents or secrets."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


# This function assembles concrete ingestion dependencies and prints a short result summary.
def main() -> None:
    """Run one explicit incremental or forced Docling-to-Qdrant ingestion command."""
    _configure_logging()
    _use_system_trust_store()
    arguments = _parse_arguments()
    LOGGER.info(
        "Starting ingestion source_root=%s collection=%s index_version=%s force_reindex=%s",
        arguments.source_root,
        arguments.collection,
        arguments.index_version,
        arguments.force_reindex,
    )
    embedder = FastEmbedEmbeddingService()
    index = QdrantDocumentIndex(arguments.qdrant_url, arguments.collection)
    service = DocumentIngestionService(embedder=embedder, index=index)
    summary = service.ingest(
        source_root=arguments.source_root,
        index_version=arguments.index_version,
        force_reindex=arguments.force_reindex,
    )
    LOGGER.info(
        "Ingestion completed indexed_documents=%d skipped_documents=%d indexed_chunks=%d",
        summary.indexed_documents,
        summary.skipped_documents,
        summary.indexed_chunks,
    )
    print(
        "Ingestion complete: "
        f"indexed documents={summary.indexed_documents}, "
        f"skipped documents={summary.skipped_documents}, "
        f"indexed chunks={summary.indexed_chunks}."
    )


# This guard runs the explicit command only when this script is started from a terminal.
if __name__ == "__main__":
    main()
