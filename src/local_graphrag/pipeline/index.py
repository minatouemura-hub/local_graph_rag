from __future__ import annotations

from local_graphrag.adapters.embedding import E5Embedder
from local_graphrag.domain.models import Chunk, EmbeddedChunk, OCRDocument
from local_graphrag.ports.interfaces import ChunkStore, FullTextStore, GraphStore, VectorStore


class IndexPipeline:
    def __init__(self, embedder: E5Embedder, chunks: ChunkStore, vector: VectorStore, full_text: FullTextStore, graph: GraphStore):
        self.embedder, self.chunks, self.vector, self.full_text, self.graph = embedder, chunks, vector, full_text, graph

    def index(self, document: OCRDocument, chunks: list[Chunk]) -> None:
        self.chunks.upsert(chunks)
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.vector.upsert([EmbeddedChunk(chunk, tuple(vector)) for chunk, vector in zip(chunks, vectors)])
        self.full_text.upsert(chunks)
        self.graph.upsert_document_structure(document.source, chunks)
