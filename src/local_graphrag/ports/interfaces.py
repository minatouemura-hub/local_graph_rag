from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from local_graphrag.domain.models import (
    Chunk,
    EmbeddedChunk,
    GeneratedAnswer,
    OCRDocument,
    OCRPage,
    RetrievalContext,
    RetrievalHit,
    SourceDocument,
)


class OCRBackend(ABC):
    @abstractmethod
    def parse_page(self, source: SourceDocument, page_image: str, page_number: int) -> OCRPage: ...


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]: ...


class Generator(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(self, query: str, context: RetrievalContext) -> GeneratedAnswer: ...


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: OCRDocument) -> list[Chunk]: ...


class ChunkStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None: ...

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [chunk for chunk_id in chunk_ids if (chunk := self.get(chunk_id)) is not None]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None: ...

    @abstractmethod
    def search(self, embedding: Sequence[float], limit: int) -> list[RetrievalHit]: ...


class FullTextStore(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, limit: int) -> list[RetrievalHit]: ...


class GraphStore(ABC):
    @abstractmethod
    def upsert_document_structure(self, source: SourceDocument, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def expand(self, chunk_ids: Sequence[str], hops: int = 1, limit: int = 20) -> list[RetrievalHit]: ...
