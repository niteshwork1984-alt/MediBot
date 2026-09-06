"""Docling parsing and heading-aware HybridChunker integration."""

import hashlib
import uuid
from pathlib import Path

from app.ingestion.collection_policy import (
    access_roles_for_collection,
    collection_for_source,
)
from app.ingestion.interfaces import ChunkingService
from app.ingestion.models import PreparedChunk


# This function creates a content fingerprint used to detect a changed source file.
def calculate_document_hash(source_path: Path) -> str:
    """Return the SHA-256 hash of one source file without loading it all at once."""
    digest = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# This function generates a deterministic UUID so repeated unchanged ingestion does not duplicate points.
def create_chunk_id(
    document_key: str,
    document_hash: str,
    index_version: str,
    chunk_position: int,
    text: str,
) -> str:
    """Return a stable Qdrant-compatible UUID for one chunk version."""
    identity = "|".join(
        [document_key, document_hash, index_version, str(chunk_position), text]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


# This function converts Docling item labels into the assignment's chunk-type vocabulary.
def _chunk_type(chunk: object) -> str:
    """Classify a Docling chunk as text, table, heading, or code."""
    metadata = getattr(chunk, "meta", None)
    labels = {
        str(getattr(item, "label", "")).lower()
        for item in getattr(metadata, "doc_items", [])
    }
    if any("table" in label for label in labels):
        return "table"
    if any("code" in label for label in labels):
        return "code"
    if any("heading" in label or "section_header" in label for label in labels):
        return "heading"
    return "text"


# This function extracts the closest available heading context from Docling chunk metadata.
def _section_title(chunk: object, fallback: str) -> str:
    """Return hierarchical headings as one readable section title."""
    metadata = getattr(chunk, "meta", None)
    headings = [str(heading).strip() for heading in getattr(metadata, "headings", [])]
    non_empty_headings = [heading for heading in headings if heading]
    return " > ".join(non_empty_headings) if non_empty_headings else fallback


# This class adapts Docling objects into MediBot's provider-independent PreparedChunk objects.
class DoclingHybridChunker(ChunkingService):
    """Parse PDF and Markdown files, then produce contextualized hierarchical chunks."""

    # This constructor delays importing heavy Docling modules until ingestion is actually run.
    def __init__(self) -> None:
        """Create reusable Docling converter and HybridChunker instances."""
        try:
            from docling.chunking import HybridChunker
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            raise RuntimeError(
                "Docling is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error

        self._converter = DocumentConverter()
        self._chunker = HybridChunker()

    # This method converts one source file into chunks containing heading context and assignment metadata.
    def chunk_file(
        self,
        source_root: Path,
        source_path: Path,
        index_version: str,
    ) -> list[PreparedChunk]:
        """Parse and chunk one approved PDF or Markdown source document."""
        collection = collection_for_source(source_root, source_path)
        document_hash = calculate_document_hash(source_path)
        document_key = source_path.resolve().relative_to(source_root.resolve()).as_posix()
        converted_document = self._converter.convert(source_path).document
        fallback_title = source_path.stem.replace("_", " ").title()
        prepared_chunks: list[PreparedChunk] = []

        for position, chunk in enumerate(self._chunker.chunk(dl_doc=converted_document)):
            text = self._chunker.contextualize(chunk).strip()
            if not text:
                continue
            prepared_chunks.append(
                PreparedChunk(
                    id=create_chunk_id(
                        document_key,
                        document_hash,
                        index_version,
                        position,
                        text,
                    ),
                    text=text,
                    source_document=source_path.name,
                    document_key=document_key,
                    collection=collection,
                    access_roles=access_roles_for_collection(collection),
                    section_title=_section_title(chunk, fallback_title),
                    chunk_type=_chunk_type(chunk),
                    document_hash=document_hash,
                    index_version=index_version,
                )
            )
        return prepared_chunks
