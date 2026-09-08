"""Explicit terminal command for inspecting Hybrid RAG candidates before and after cross-encoder reranking."""

import argparse
import logging
import os

from app.core.config import (
    HYBRID_RAG_CANDIDATE_LIMIT,
    HYBRID_RAG_RERANK_LIMIT,
    QDRANT_COLLECTION,
    QDRANT_URL,
)
from app.services.reranking.fastembed_cross_encoder_reranker import FastEmbedCrossEncoderReranker
from app.services.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever


# This function enables operating-system certificate trust before FastEmbed downloads the reranker model.
def _use_system_trust_store() -> None:
    """Enable system certificate trust for a first-time local reranker-model download."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


# This function maps the project's descriptive token name to the Hugging Face client setting without logging it.
def _configure_hugging_face_token() -> None:
    """Expose HF_ACCESS_TOKEN as HF_TOKEN only when the standard setting is absent."""
    access_token = os.getenv("HF_ACCESS_TOKEN")
    if access_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = access_token


# This function reads terminal settings for a complete retrieve-then-rerank learning test.
def _parse_arguments() -> argparse.Namespace:
    """Return question, role, candidate count, rerank count, and Qdrant settings."""
    parser = argparse.ArgumentParser(
        description="Retrieve Qdrant Hybrid RAG candidates, then rerank them with a cross-encoder."
    )
    parser.add_argument("question", help="Natural-language question to retrieve and rerank.")
    parser.add_argument("--role", required=True, help="MediBot role to apply inside Qdrant.")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=HYBRID_RAG_CANDIDATE_LIMIT,
        help="Role-authorized Hybrid RAG candidates retrieved before reranking.",
    )
    parser.add_argument(
        "--rerank-limit",
        type=int,
        default=HYBRID_RAG_RERANK_LIMIT,
        help="Best cross-encoder-ranked chunks to display.",
    )
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    return parser.parse_args()


# This function configures concise logs without recording questions, chunk text, vectors, or scores.
def _configure_logging() -> None:
    """Configure timestamped terminal logs for this explicit reranking command."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


# This function performs one role-filtered retrieval followed by a local cross-encoder reranking pass.
def main() -> None:
    """Print the final reranked chunks and their distinct fusion and reranker scores."""
    _configure_logging()
    _use_system_trust_store()
    _configure_hugging_face_token()
    arguments = _parse_arguments()
    candidates = QdrantHybridRetriever(
        url=arguments.qdrant_url,
        collection_name=arguments.collection,
    ).search(arguments.question, arguments.role, arguments.candidate_limit)
    reranked_chunks = FastEmbedCrossEncoderReranker().rerank(
        arguments.question,
        candidates,
        arguments.rerank_limit,
    )
    print(f"Reranked chunks: {len(reranked_chunks)}")
    for position, result in enumerate(reranked_chunks, start=1):
        chunk = result.chunk
        print(
            f"\n[{position}] source={chunk.source_document} section={chunk.section_title} "
            f"collection={chunk.collection} fusion_score={chunk.fusion_score:.6f} "
            f"reranker_score={result.reranker_score:.6f}\n{chunk.text}"
        )


# This guard starts the explicit reranking command only when Python executes this module directly.
if __name__ == "__main__":
    main()
