from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    path: Path
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRBlock:
    block_id: str
    page: int
    text: str
    bbox: BoundingBox | None
    layout_type: str
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRPage:
    document_id: str
    page: int
    width: int | None
    height: int | None
    blocks: tuple[OCRBlock, ...]


@dataclass(frozen=True)
class OCRDocument:
    source: SourceDocument
    pages: tuple[OCRPage, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    page: int
    text: str
    section_id: str | None
    order: int
    bbox: BoundingBox | None = None
    layout_types: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class Citation:
    document_id: str
    page: int
    chunk_id: str
    source_path: str
    bbox: BoundingBox | None = None


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    source: str
    rank: int


@dataclass(frozen=True)
class ContextItem:
    chunk: Chunk
    score: float
    citations: tuple[Citation, ...]
    reason: str = ""


@dataclass(frozen=True)
class RetrievalContext:
    query: str
    items: tuple[ContextItem, ...]


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    citations: tuple[Citation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
