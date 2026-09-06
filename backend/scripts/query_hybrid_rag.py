"""Explicit terminal command for inspecting role-filtered Hybrid RAG retrieval results."""

import argparse
import logging
import os

from app.core.config import HYBRID_RAG_DEFAULT_LIMIT, QDRANT_COLLECTION, QDRANT_URL
from app.services.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever


# This function reads terminal settings for one Hybrid RAG retrieval test.
def _parse_arguments() -> argparse.Namespace:
    """Return a question, role, result limit, and optional Qdrant connection settings."""
    parser = argparse.ArgumentParser(
        description="Retrieve role-authorized MediBot document chunks from Qdrant."
    )
    parser.add_argument("question", help="Natural-language question to retrieve against.")
    parser.add_argument("--role", required=True, help="Authenticated MediBot role to test.")
    parser.add_argument(
        "--limit",
        type=int,
        default=HYBRID_RAG_DEFAULT_LIMIT,
        help="Maximum chunks to return; bounded by HYBRID_RAG_MAX_LIMIT.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", QDRANT_COLLECTION),
        help="Existing Qdrant collection containing MediBot chunks.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", QDRANT_URL),
        help="Qdrant REST endpoint.",
    )
    return parser.parse_args()


# This function configures concise terminal logs without recording question text or retrieved content.
def _configure_logging() -> None:
    """Configure timestamped operational logs for this explicit terminal command."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


# This function runs the retrieval command and prints results for learning before adding an LLM answer layer.
def main() -> None:
    """Run one role-filtered hybrid retrieval and print citation-ready chunks."""
    _configure_logging()
    arguments = _parse_arguments()
    retriever = QdrantHybridRetriever(
        url=arguments.qdrant_url,
        collection_name=arguments.collection,
    )
    results = retriever.search(arguments.question, arguments.role, arguments.limit)
    print(f"Retrieved chunks: {len(results)}")
    for position, result in enumerate(results, start=1):
        print(
            f"\n[{position}] source={result.source_document} "
            f"section={result.section_title} collection={result.collection} "
            f"fusion_score={result.fusion_score:.6f}\n{result.text}"
        )


# This guard runs main only when this module is started from a terminal.
if __name__ == "__main__":
    main()
