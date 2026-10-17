from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_graphrag.adapters.ladybug import (
    LadybugChunkStore,
    LadybugConnection,
    LadybugFullTextStore,
    LadybugGraphStore,
    LadybugVectorStore,
)
from local_graphrag.domain.models import (
    BoundingBox,
    EmbeddedChunk,
    OCRBlock,
    OCRDocument,
    OCRPage,
    SourceDocument,
)
from local_graphrag.ports.interfaces import Embedder
from local_graphrag.services.layout_chunker import LayoutAwareChunker
from local_graphrag.services.retrieval import HybridRetriever

FIXTURE = Path(__file__).parents[1] / "fixtures" / "sample_ocr_document.json"


def load_fixture() -> OCRDocument:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pages = tuple(
        OCRPage(
            raw["document_id"],
            page["page"],
            None,
            None,
            tuple(
                OCRBlock(
                    block["block_id"],
                    page["page"],
                    block.get("text", ""),
                    BoundingBox(*block["bbox"]) if block.get("bbox") else None,
                    block.get("layout_type", "Text"),
                    block.get("order", index),
                )
                for index, block in enumerate(page["blocks"])
            ),
        )
        for page in raw["pages"]
    )
    return OCRDocument(SourceDocument(raw["document_id"], Path(raw["path"])), pages)


class FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        if "電源" in text:
            return [1.0, 0.0, 0.0]
        if "安全" in text or "分解" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.fixture
def stores(tmp_path: Path):
    connection = LadybugConnection(tmp_path / "retrieval.lbdb")
    connection.initialize(dimension=3)
    chunks = LayoutAwareChunker().chunk(load_fixture())
    chunk_store = LadybugChunkStore(connection)
    vector_store = LadybugVectorStore(connection)
    full_text_store = LadybugFullTextStore(connection)
    graph_store = LadybugGraphStore(connection)
    chunk_store.upsert(chunks)
    vector_store.upsert(
        [EmbeddedChunk(chunk, tuple(FakeEmbedder._vector(chunk.text))) for chunk in chunks]
    )
    full_text_store.upsert(chunks)
    graph_store.upsert_document_structure(load_fixture().source, chunks)
    return connection, chunks, chunk_store, vector_store, full_text_store, graph_store


def test_chunk_round_trip_and_schema(stores) -> None:
    _, chunks, chunk_store, _, _, _ = stores
    restored = chunk_store.get(chunks[2].chunk_id)
    assert restored is not None
    assert restored.chunk_id == chunks[2].chunk_id
    assert restored.text == chunks[2].text
    assert restored.page == chunks[2].page
    assert restored.section_id == chunks[2].section_id
    assert restored.layout_types == chunks[2].layout_types


def test_vector_hnsw_search(stores) -> None:
    _, chunks, _, vector_store, _, _ = stores
    hits = vector_store.search([1.0, 0.0, 0.0], limit=5)
    assert hits
    assert hits[0].source == "vector"
    assert "電源" in hits[0].chunk.text
    assert hits[0].chunk.chunk_id in {chunk.chunk_id for chunk in chunks}


def test_bm25_search(stores) -> None:
    _, _, _, _, full_text_store, _ = stores
    hits = full_text_store.search("電源", limit=5)
    assert hits
    assert all(hit.source == "fulltext" for hit in hits)
    assert any("電源" in hit.chunk.text for hit in hits)


def test_graph_next_and_same_section_expansion(stores) -> None:
    _, chunks, _, _, _, graph_store = stores
    seed = next(chunk for chunk in chunks if "水に濡らさない" in chunk.text)
    hits = graph_store.expand([seed.chunk_id], hops=1, limit=20)
    ids = {hit.chunk.chunk_id for hit in hits}
    assert any("本装置を使用する前に" in hit.chunk.text for hit in hits)
    assert any(hit.source.startswith("graph:") for hit in hits)
    assert seed.chunk_id not in ids


def test_graph_limit_and_hop_zero(stores) -> None:
    _, chunks, _, _, _, graph_store = stores
    hits = graph_store.expand([chunks[2].chunk_id], hops=1, limit=2)
    assert len(hits) <= 2
    assert graph_store.expand([chunks[2].chunk_id], hops=0, limit=20) == []


def test_hybrid_retrieval_deduplicates_and_limits(stores) -> None:
    _, _, _, vector_store, full_text_store, graph_store = stores
    retriever = HybridRetriever(
        FakeEmbedder(),
        vector_store,
        full_text_store,
        graph_store,
        graph_hops=1,
        graph_max_nodes=20,
    )
    context = retriever.retrieve("電源", limit=5, candidate_limit=5)
    ids = [item.chunk.chunk_id for item in context.items]
    assert context.query == "電源"
    assert context.items
    assert len(ids) == len(set(ids))
    assert len(ids) <= 5
    assert any("電源" in item.chunk.text for item in context.items)
    assert any(item.reason for item in context.items)
    assert any("graph:" in item.reason for item in context.items)


def test_empty_query_returns_empty_context(stores) -> None:
    _, _, _, vector_store, full_text_store, graph_store = stores
    retriever = HybridRetriever(FakeEmbedder(), vector_store, full_text_store, graph_store)
    assert retriever.retrieve("   ").items == ()
