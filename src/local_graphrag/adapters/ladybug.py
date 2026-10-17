from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_graphrag.domain.models import (
    BoundingBox,
    Chunk,
    EmbeddedChunk,
    RetrievalHit,
    SourceDocument,
)
from local_graphrag.ports.interfaces import ChunkStore, FullTextStore, GraphStore, VectorStore


@dataclass(frozen=True)
class LadybugConfig:
    vector_index_name: str = "chunk_embedding_index"
    full_text_index_name: str = "chunk_text_index"
    graph_max_nodes: int = 20


def _schema(dimension: int) -> tuple[str, ...]:
    return (
        "CREATE NODE TABLE Document(document_id STRING PRIMARY KEY, path STRING, title STRING)",
        "CREATE NODE TABLE Page(page_id STRING PRIMARY KEY, document_id STRING, page_number INT64)",
        "CREATE NODE TABLE Section(section_id STRING PRIMARY KEY, document_id STRING, title STRING)",
        (
            "CREATE NODE TABLE Chunk("
            f"chunk_id STRING PRIMARY KEY, document_id STRING, section_id STRING, page INT64, "
            f"chunk_order INT64, text STRING, fts_text STRING, layout_type STRING, source_block_ids STRING[], "
            f"bbox_x0 DOUBLE, bbox_y0 DOUBLE, bbox_x1 DOUBLE, bbox_y1 DOUBLE, embedding FLOAT[{dimension}])"
        ),
        "CREATE REL TABLE HAS_PAGE(FROM Document TO Page)",
        "CREATE REL TABLE HAS_SECTION(FROM Page TO Section)",
        "CREATE REL TABLE HAS_CHUNK(FROM Section TO Chunk)",
        "CREATE REL TABLE NEXT(FROM Chunk TO Chunk)",
    )


class LadybugConnection:
    """Own one embedded Ladybug database and its shared connection."""

    def __init__(self, path: str | Path, *, config: LadybugConfig | None = None):
        import ladybug as lb

        self.path = Path(path)
        self.config = config or LadybugConfig()
        self.db = lb.Database(str(self.path))
        self.conn = lb.Connection(self.db)
        self.dimension: int | None = None

    def initialize(self, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.conn.execute("INSTALL VECTOR;")
        self.conn.execute("LOAD VECTOR;")
        self.conn.execute("INSTALL FTS;")
        self.conn.execute("LOAD FTS;")
        for statement in _schema(dimension):
            self.conn.execute(statement)
        self.conn.execute(
            f"CALL CREATE_VECTOR_INDEX('Chunk', '{self.config.vector_index_name}', 'embedding', metric := 'cosine', cache_embeddings := false)"
        )


class LadybugChunkStore(ChunkStore):
    def __init__(self, connection: LadybugConnection):
        self.connection = connection

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self.connection.conn.execute(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.document_id = $document_id,
                    c.section_id = $section_id,
                    c.page = $page,
                    c.chunk_order = $chunk_order,
                    c.text = $text,
                    c.fts_text = $fts_text,
                    c.layout_type = $layout_type,
                    c.source_block_ids = $source_block_ids,
                    c.bbox_x0 = $bbox_x0,
                    c.bbox_y0 = $bbox_y0,
                    c.bbox_x1 = $bbox_x1,
                    c.bbox_y1 = $bbox_y1
                """,
                _chunk_params(chunk),
            )

    def get(self, chunk_id: str) -> Chunk | None:
        if not chunk_id:
            return None
        result = self.connection.conn.execute(
            "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c.chunk_id, c.document_id, c.section_id, c.page, c.chunk_order, c.text, c.layout_type, c.source_block_ids, c.bbox_x0, c.bbox_y0, c.bbox_x1, c.bbox_y1",
            {"chunk_id": chunk_id},
        )
        rows = _rows(result)
        return _chunk_from_values(rows[0]) if rows else None

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [chunk for chunk_id in chunk_ids if (chunk := self.get(chunk_id)) is not None]


class LadybugVectorStore(VectorStore):
    def __init__(self, connection: LadybugConnection):
        self.connection = connection

    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        dimension = self.connection.dimension
        for item in chunks:
            if dimension is not None and len(item.embedding) != dimension:
                raise ValueError(f"embedding dimension mismatch: expected {dimension}, got {len(item.embedding)}")
            self.connection.conn.execute(
                "MATCH (c:Chunk {chunk_id: $chunk_id}) SET c.embedding = $embedding",
                {"chunk_id": item.chunk.chunk_id, "embedding": list(item.embedding)},
            )

    def search(self, embedding: Sequence[float], limit: int) -> list[RetrievalHit]:
        if limit <= 0 or not embedding:
            return []
        if self.connection.dimension is not None and len(embedding) != self.connection.dimension:
            raise ValueError(f"embedding dimension mismatch: expected {self.connection.dimension}, got {len(embedding)}")
        result = self.connection.conn.execute(
            f"""
            CALL QUERY_VECTOR_INDEX('Chunk', '{self.connection.config.vector_index_name}', $embedding, $limit)
            RETURN node.chunk_id, node.document_id, node.section_id, node.page,
                   node.chunk_order, node.text, node.layout_type, node.source_block_ids,
                   node.bbox_x0, node.bbox_y0, node.bbox_x1, node.bbox_y1, distance
            ORDER BY distance
            """,
            {
                "embedding": list(embedding),
                "limit": limit,
            },
        )
        return [
            RetrievalHit(_chunk_from_values(row[:-1]), 1.0 - float(row[-1]), "vector", rank)
            for rank, row in enumerate(_rows(result), start=1)
        ]


class LadybugFullTextStore(FullTextStore):
    def __init__(self, connection: LadybugConnection):
        self.connection = connection

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        try:
            self.connection.conn.execute(
                f"CALL DROP_FTS_INDEX('Chunk', '{self.connection.config.full_text_index_name}')"
            )
        except RuntimeError:
            # The first upsert may run before an index has been created.
            pass
        self.connection.conn.execute(
            f"CALL CREATE_FTS_INDEX('Chunk', '{self.connection.config.full_text_index_name}', ['fts_text'], stemmer := 'none')"
        )

    def search(self, query: str, limit: int) -> list[RetrievalHit]:
        if not query.strip() or limit <= 0:
            return []
        result = self.connection.conn.execute(
            f"""
            CALL QUERY_FTS_INDEX('Chunk', '{self.connection.config.full_text_index_name}', $query, top := $limit)
            RETURN node.chunk_id, node.document_id, node.section_id, node.page,
                   node.chunk_order, node.text, node.layout_type, node.source_block_ids,
                   node.bbox_x0, node.bbox_y0, node.bbox_x1, node.bbox_y1, score
            ORDER BY score DESC, node.chunk_id
            """,
            {
                "query": _fts_text(query),
                "limit": limit,
            },
        )
        return [
            RetrievalHit(_chunk_from_values(row[:-1]), float(row[-1]), "fulltext", rank)
            for rank, row in enumerate(_rows(result), start=1)
        ]


class LadybugGraphStore(GraphStore):
    def __init__(self, connection: LadybugConnection):
        self.connection = connection

    def upsert_document_structure(self, source: SourceDocument, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        self.connection.conn.execute(
            "MERGE (d:Document {document_id: $document_id}) SET d.path = $path, d.title = $title",
            {
                "document_id": source.document_id,
                "path": str(source.path),
                "title": source.title or "",
            },
        )
        pages = sorted({chunk.page for chunk in chunks})
        for page in pages:
            page_id = f"{source.document_id}:page:{page}"
            self.connection.conn.execute(
                "MERGE (p:Page {page_id: $page_id}) SET p.document_id = $document_id, p.page_number = $page_number",
                {"page_id": page_id, "document_id": source.document_id, "page_number": page},
            )
            self.connection.conn.execute(
                "MATCH (d:Document {document_id: $document_id}), (p:Page {page_id: $page_id}) MERGE (d)-[:HAS_PAGE]->(p)",
                {"document_id": source.document_id, "page_id": page_id},
            )

        sections: dict[str, str] = {}
        for chunk in chunks:
            if chunk.section_id is not None:
                sections[chunk.section_id] = str(chunk.metadata.get("section_title") or "")
        for section_id, title in sections.items():
            self.connection.conn.execute(
                "MERGE (s:Section {section_id: $section_id}) SET s.document_id = $document_id, s.title = $title",
                {"section_id": section_id, "document_id": source.document_id, "title": title},
            )
        for chunk in chunks:
            if chunk.section_id:
                self.connection.conn.execute(
                    "MATCH (s:Section {section_id: $section_id}), (c:Chunk {chunk_id: $chunk_id}) MERGE (s)-[:HAS_CHUNK]->(c)",
                    {"section_id": chunk.section_id, "chunk_id": chunk.chunk_id},
                )
                self.connection.conn.execute(
                    "MATCH (p:Page {document_id: $document_id, page_number: $page}), (s:Section {section_id: $section_id}) MERGE (p)-[:HAS_SECTION]->(s)",
                    {"document_id": source.document_id, "page": chunk.page, "section_id": chunk.section_id},
                )
        ordered = sorted(chunks, key=lambda chunk: (chunk.page, chunk.order, chunk.chunk_id))
        for previous, current in itertools.pairwise(ordered):
            if previous.page == current.page:
                self.connection.conn.execute(
                    "MATCH (a:Chunk {chunk_id: $previous}), (b:Chunk {chunk_id: $current}) MERGE (a)-[:NEXT]->(b)",
                    {"previous": previous.chunk_id, "current": current.chunk_id},
                )

    def expand(self, chunk_ids: Sequence[str], hops: int = 1, limit: int | None = None) -> list[RetrievalHit]:
        if hops <= 0 or not chunk_ids:
            return []
        max_nodes = limit or self.connection.config.graph_max_nodes
        candidates: dict[str, RetrievalHit] = {}
        for seed_id in chunk_ids:
            for query, params, score, source in self._expansion_queries(seed_id):
                result = self.connection.conn.execute(query, params)
                for row in _rows(result):
                    chunk = _chunk_from_values(row)
                    if chunk.chunk_id in chunk_ids:
                        continue
                    hit = RetrievalHit(chunk, score, source, 0)
                    previous = candidates.get(chunk.chunk_id)
                    if previous is None or hit.score > previous.score:
                        candidates[chunk.chunk_id] = hit
        ordered = sorted(candidates.values(), key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:max_nodes]
        return [RetrievalHit(hit.chunk, hit.score, hit.source, rank) for rank, hit in enumerate(ordered, 1)]

    @staticmethod
    def _expansion_queries(seed_id: str) -> tuple[tuple[str, dict[str, str], float, str], ...]:
        common = (
            "n.chunk_id, n.document_id, n.section_id, n.page, n.chunk_order, n.text, "
            "n.layout_type, n.source_block_ids, n.bbox_x0, n.bbox_y0, n.bbox_x1, n.bbox_y1"
        )
        return (
            (
                f"MATCH (seed:Chunk {{chunk_id: $seed_id}})-[:NEXT]->(n:Chunk) RETURN {common}",
                {"seed_id": seed_id},
                1.0,
                "graph:next",
            ),
            (
                f"MATCH (n:Chunk)-[:NEXT]->(seed:Chunk {{chunk_id: $seed_id}}) RETURN {common}",
                {"seed_id": seed_id},
                1.0,
                "graph:next",
            ),
            (
                f"MATCH (s:Section)-[:HAS_CHUNK]->(seed:Chunk {{chunk_id: $seed_id}}) MATCH (s)-[:HAS_CHUNK]->(n:Chunk) WHERE n.chunk_id <> $seed_id RETURN {common}",
                {"seed_id": seed_id},
                0.8,
                "graph:same_section",
            ),
        )


def _chunk_params(chunk: Chunk) -> dict[str, Any]:
    bbox = chunk.bbox
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "section_id": chunk.section_id or "",
        "page": chunk.page,
        "chunk_order": chunk.order,
        "text": chunk.text,
        "fts_text": _fts_text(chunk.text),
        "layout_type": ",".join(chunk.layout_types) or "Text",
        "source_block_ids": list(chunk.metadata.get("source_block_ids", ())),
        "bbox_x0": bbox.x0 if bbox else None,
        "bbox_y0": bbox.y0 if bbox else None,
        "bbox_x1": bbox.x1 if bbox else None,
        "bbox_y1": bbox.y1 if bbox else None,
    }


def _rows(result: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    while result.has_next():
        rows.append(result.get_next())
    return rows


def _chunk_from_values(values: Sequence[Any]) -> Chunk:
    bbox = None
    if all(value is not None for value in values[8:12]):
        bbox = BoundingBox(float(values[8]), float(values[9]), float(values[10]), float(values[11]))
    return Chunk(
        chunk_id=str(values[0]),
        document_id=str(values[1]),
        section_id=str(values[2]) or None,
        page=int(values[3]),
        text=str(values[5]),
        order=int(values[4]),
        bbox=bbox,
        layout_types=tuple(str(values[6]).split(",")) if values[6] else (),
        metadata={"source_block_ids": tuple(values[7] or ())},
    )


def _fts_text(text: str) -> str:
    """Create a small CJK-friendly token stream for Ladybug's BM25 index."""
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]|[^\s]", text)
    return " ".join(tokens)
