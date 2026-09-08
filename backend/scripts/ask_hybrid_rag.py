"""Explicit terminal command for a complete role-filtered Hybrid RAG answer request."""

import argparse
import logging
import os

from app.core.config import QDRANT_COLLECTION, QDRANT_URL
from app.services.hybrid_rag_answer_service import create_hybrid_rag_answer_service
from app.services.reranking.fastembed_cross_encoder_reranker import FastEmbedCrossEncoderReranker
from app.services.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever


# This function enables operating-system certificate trust before local model initialization.
def _use_system_trust_store() -> None:
    """Enable system certificate trust for local model downloads when available."""
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()


# This function maps the project token name to the Hugging Face client setting without logging it.
def _configure_hugging_face_token() -> None:
    """Expose HF_ACCESS_TOKEN as HF_TOKEN only when the standard setting is absent."""
    access_token = os.getenv("HF_ACCESS_TOKEN")
    if access_token and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = access_token


# This function reads one intentional terminal answer request and requires acknowledgement of the provider call.
def _parse_arguments() -> argparse.Namespace:
    """Return the question, authenticated-role simulation, Qdrant settings, and explicit call approval."""
    parser = argparse.ArgumentParser(
        description="Retrieve, rerank, and answer with the configured Hybrid RAG LLM provider."
    )
    parser.add_argument("question", help="Natural-language document question to answer.")
    parser.add_argument("--role", required=True, help="Role to apply inside Qdrant for this learning test.")
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument(
        "--allow-llm-call",
        action="store_true",
        help="Required acknowledgement before this command sends context to the configured provider.",
    )
    return parser.parse_args()


# This function configures concise logs without writing questions, document text, or model responses to logs.
def _configure_logging() -> None:
    """Configure timestamped terminal logs for the explicit answer command."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


# This function assembles concrete Qdrant, reranking, and configured-provider dependencies.
def _create_service(arguments: argparse.Namespace):
    """Create the complete Hybrid RAG service used only by this terminal command."""
    retriever = QdrantHybridRetriever(
        url=arguments.qdrant_url,
        collection_name=arguments.collection,
    )
    reranker = FastEmbedCrossEncoderReranker()
    return create_hybrid_rag_answer_service(retriever, reranker)


# This function runs one explicitly approved provider request and prints the answer with trusted citations.
def main() -> None:
    """Answer one document question after retrieve, rerank, and provider-generation stages."""
    _configure_logging()
    _use_system_trust_store()
    _configure_hugging_face_token()
    arguments = _parse_arguments()
    if not arguments.allow_llm_call:
        raise SystemExit("Add --allow-llm-call to confirm this command may send context to the LLM provider.")

    result = _create_service(arguments).answer_with_details(arguments.question, arguments.role)
    print(f"Answer:\n{result.answer}")
    print("\nSources:")
    if not result.sources:
        print("None")
    for position, source in enumerate(result.sources, start=1):
        print(
            f"[{position}] document={source.source_document}; "
            f"section={source.section_title}; collection={source.collection}"
        )


# This guard starts the explicit answer command only when Python executes this module directly.
if __name__ == "__main__":
    main()
