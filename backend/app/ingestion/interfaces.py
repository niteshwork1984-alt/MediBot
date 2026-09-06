"""Java-like abstract interfaces shared by ingestion implementations and orchestration."""

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.models import IndexedChunk, PreparedChunk


# This abstract base class is the Java-like interface for parser/chunker implementations.
class ChunkingService(ABC):
    """Interface for converting one source file into prepared metadata-rich chunks."""

    # This abstract method supplies all chunks for one source document version.
    @abstractmethod
    def chunk_file(
        self,
        source_root: Path,
        source_path: Path,
        index_version: str,
    ) -> list[PreparedChunk]:
        """Parse and chunk one source document."""
        raise NotImplementedError


# This abstract base class is the Java-like interface for local embedding implementations.
class EmbeddingService(ABC):
    """Interface for generating vectors from prepared chunks."""

    # This abstract method reports the dense-vector size required by Qdrant collection creation.
    @abstractmethod
    def dense_vector_size(self) -> int:
        """Return the number of values in each dense vector."""
        raise NotImplementedError

    # This abstract method returns the LangChain sparse embedding implementation for hybrid storage.
    @abstractmethod
    def sparse_embeddings(self) -> object:
        """Return the sparse embeddings implementation used by LangChain Qdrant."""
        raise NotImplementedError
