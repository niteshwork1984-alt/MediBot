"""LangChain embedding adapters backed by local FastEmbed models."""

import logging
import time

from app.ingestion.interfaces import EmbeddingService
from langchain_core.embeddings import Embeddings


LOGGER = logging.getLogger(__name__)


# This class creates local dense and sparse embeddings without sending document text to an LLM provider.
class FastEmbedEmbeddingService(EmbeddingService, Embeddings):
    """Expose local FastEmbed dense and sparse models through LangChain interfaces."""

    # This constructor configures models but defers downloading weights until the first embedding call.
    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        sparse_model_name: str = "Qdrant/bm25",
    ) -> None:
        """Create local FastEmbed model adapters for semantic and keyword vectors."""
        try:
            from fastembed import TextEmbedding
            from langchain_qdrant import FastEmbedSparse
        except ImportError as error:
            raise RuntimeError(
                "FastEmbed is not installed. Run: python3 -m pip install -r requirements.txt"
            ) from error

        self.dense_model_name = dense_model_name
        self.sparse_model_name = sparse_model_name
        self._dense_model = TextEmbedding(model_name=dense_model_name, lazy_load=True)
        self._sparse_model = FastEmbedSparse(model_name=sparse_model_name)
        LOGGER.info(
            "Configured local embedding models dense_model=%s sparse_model=%s",
            dense_model_name,
            sparse_model_name,
        )

    # This method finds the dense-vector dimension required when creating a Qdrant collection.
    def dense_vector_size(self) -> int:
        """Return the configured dense embedding size using one local probe embedding."""
        probe_vector = next(self._dense_model.embed(["MediBot embedding dimension probe"]))
        dimension = len(probe_vector)
        LOGGER.info("Resolved dense embedding dimension dense_model=%s dimension=%d", self.dense_model_name, dimension)
        return dimension

    # This LangChain interface method embeds a batch of document texts as dense vectors.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return local dense embeddings for LangChain document ingestion."""
        embedding_started_at = time.monotonic()
        vectors = [[float(value) for value in vector] for vector in self._dense_model.embed(texts)]
        LOGGER.info(
            "Generated dense embeddings dense_model=%s document_count=%d duration_ms=%d",
            self.dense_model_name,
            len(texts),
            (time.monotonic() - embedding_started_at) * 1000,
        )
        return vectors

    # This LangChain interface method embeds one query text as a dense vector.
    def embed_query(self, text: str) -> list[float]:
        """Return one local dense embedding for future LangChain retrieval."""
        return self.embed_documents([text])[0]

    # This method returns LangChain's FastEmbed sparse/BM25 adapter for hybrid Qdrant storage.
    def sparse_embeddings(self) -> object:
        """Return the configured LangChain sparse embedding provider."""
        return self._sparse_model
