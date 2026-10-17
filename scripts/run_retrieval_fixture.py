"""Run fixture indexing and hybrid retrieval against an embedded Ladybug DB."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from run_chunker_fixture import load_ocr_document

from local_graphrag.adapters.ladybug import (
    LadybugChunkStore,
    LadybugConnection,
    LadybugFullTextStore,
    LadybugGraphStore,
    LadybugVectorStore,
)
from local_graphrag.domain.models import EmbeddedChunk
from local_graphrag.services.layout_chunker import LayoutAwareChunker
from local_graphrag.services.retrieval import HybridRetriever


class FixtureEmbedder:
    dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        if "電源" in text:
            return [1.0, 0.0, 0.0]
        if "安全" in text or "分解" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="電源を切る必要があるのはなぜですか？")
    parser.add_argument("--db", type=Path, default=None)
    args = parser.parse_args()

    document = load_ocr_document(Path("tests/fixtures/sample_ocr_document.json"))
    chunks = LayoutAwareChunker().chunk(document)
    temp = tempfile.TemporaryDirectory() if args.db is None else None
    db_path = args.db or Path(temp.name) / "retrieval.lbdb"
    try:
        connection = LadybugConnection(db_path)
        connection.initialize(dimension=3)
        chunk_store = LadybugChunkStore(connection)
        vector_store = LadybugVectorStore(connection)
        full_text_store = LadybugFullTextStore(connection)
        graph_store = LadybugGraphStore(connection)
        embedder = FixtureEmbedder()
        chunk_store.upsert(chunks)
        vector_store.upsert(
            [EmbeddedChunk(chunk, tuple(embedder._vector(chunk.text))) for chunk in chunks]
        )
        full_text_store.upsert(chunks)
        graph_store.upsert_document_structure(document.source, chunks)
        retriever = HybridRetriever(embedder, vector_store, full_text_store, graph_store)
        context = retriever.retrieve(args.query, limit=5, candidate_limit=5)
        print(f"Query: {args.query}")
        print(f"Indexed chunks: {len(chunks)}")
        print("Final context:")
        for index, item in enumerate(context.items, start=1):
            print(f"{index}. [{item.reason}] p{item.chunk.page} {item.chunk.text}")
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
