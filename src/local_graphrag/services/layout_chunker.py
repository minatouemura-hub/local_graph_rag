from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import ceil
from typing import Final

from local_graphrag.domain.models import BoundingBox, Chunk, OCRBlock, OCRDocument
from local_graphrag.ports.interfaces import Chunker

_TEXT_LIKE: Final = {"Text"}
_LIST_LIKE: Final = {"List-item", "List"}
_ATOMIC: Final = {
    "Table",
    "Formula",
    "Caption",
    "Figure",
    "Picture",
    "Footnote",
}


@dataclass
class _Group:
    blocks: list[OCRBlock]
    section_id: str
    section_title: str | None


class LayoutAwareChunker(Chunker):
    """Convert OCR blocks into deterministic, page-local retrieval chunks.

    The chunker only depends on normalized domain DTOs. Model-specific OCR parsing
    and tokenizers remain outside this service.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        *,
        token_counter: Callable[[str], int] | None = None,
        chars_per_token: int = 4,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.max_tokens = max_tokens
        self.token_counter = token_counter
        self.chars_per_token = chars_per_token

    def chunk(self, document: OCRDocument) -> list[Chunk]:
        blocks = self._ordered_blocks(document)
        if not blocks:
            return []

        chunks: list[Chunk] = []
        current_section_id = f"{document.source.document_id}:section:preamble"
        current_section_title: str | None = None
        section_index = 0
        title_text: str | None = None
        pending: _Group | None = None

        for block in blocks:
            layout = self._normalize_layout_type(block.layout_type)

            if layout == "Title":
                self._flush_group(pending, chunks, document, title_text=title_text)
                pending = None
                if block.text.strip():
                    title_text = block.text.strip()
                    chunks.extend(
                        self._build_chunks(
                            [block],
                            f"{document.source.document_id}:section:title",
                            "Document title",
                            document,
                        )
                    )
                continue

            if layout == "Section-header":
                self._flush_group(pending, chunks, document, title_text=title_text)
                pending = None
                section_index += 1
                current_section_id = f"{document.source.document_id}:section:{section_index}"
                current_section_title = block.text.strip() or None
                continue

            if self._is_empty_text_block(block, layout):
                # Empty structural blocks (Figure/Picture) are retained below;
                # empty ordinary text does not create retrieval noise.
                continue

            if layout in _ATOMIC:
                self._flush_group(pending, chunks, document, title_text=title_text)
                pending = None
                chunks.extend(
                    self._build_chunks(
                        [block], current_section_id, current_section_title, document,
                        title_text=title_text,
                    )
                )
                continue

            if pending is None or not self._can_merge(pending, block, layout):
                self._flush_group(pending, chunks, document, title_text=title_text)
                pending = _Group([block], current_section_id, current_section_title)
            else:
                pending.blocks.append(block)

            if self._would_exceed(pending.blocks):
                overflow = pending.blocks.pop()
                self._flush_group(pending, chunks, document, title_text=title_text)
                pending = _Group([overflow], current_section_id, current_section_title)

        self._flush_group(pending, chunks, document, title_text=title_text)
        return self._renumber(chunks)

    def _ordered_blocks(self, document: OCRDocument) -> list[OCRBlock]:
        return [
            block
            for page in sorted(document.pages, key=lambda item: item.page)
            for block in sorted(page.blocks, key=lambda item: (item.order, item.block_id))
        ]

    def _flush_group(
        self,
        group: _Group | None,
        chunks: list[Chunk],
        document: OCRDocument,
        *,
        title_text: str | None,
    ) -> None:
        if group is not None and group.blocks:
            chunks.extend(
                self._build_chunks(
                    group.blocks,
                    group.section_id,
                    group.section_title,
                    document,
                    title_text=title_text,
                )
            )

    def _build_chunks(
        self,
        blocks: list[OCRBlock],
        section_id: str,
        section_title: str | None,
        document: OCRDocument,
        *,
        title_text: str | None = None,
    ) -> list[Chunk]:
        result: list[Chunk] = []
        for part_index, part in enumerate(self._split_blocks(blocks)):
            text = self._join_text(part)
            layout_types = tuple(dict.fromkeys(self._normalize_layout_type(b.layout_type) for b in part))
            metadata = {
                "source_block_ids": tuple(block.block_id for block in part),
                "section_title": section_title,
            }
            if title_text:
                metadata["document_title"] = title_text
            result.append(
                Chunk(
                    chunk_id="",  # Assigned deterministically by _renumber.
                    document_id=document.source.document_id,
                    page=part[0].page,
                    text=text,
                    section_id=section_id,
                    order=0,
                    bbox=self._union_bbox(part),
                    layout_types=layout_types,
                    metadata=metadata,
                )
            )
        return result

    def _split_blocks(self, blocks: list[OCRBlock]) -> list[list[OCRBlock]]:
        if not blocks:
            return []
        if len(blocks) == 1 and self._estimate_tokens(self._join_text(blocks)) <= self.max_tokens:
            return [blocks]

        parts: list[list[OCRBlock]] = []
        current: list[OCRBlock] = []
        for block in blocks:
            candidate = current + [block]
            if current and self._estimate_tokens(self._join_text(candidate)) > self.max_tokens:
                parts.append(current)
                current = [block]
            else:
                current = candidate
        if current:
            # A single oversized block is split exactly by character offsets so
            # its normalized text is not lost or duplicated.
            if len(current) == 1 and self._estimate_tokens(self._join_text(current)) > self.max_tokens:
                text = self._join_text(current)
                width = self.max_tokens * self.chars_per_token
                for start in range(0, len(text), width):
                    end = min(start + width, len(text))
                    parts.append([self._copy_block_with_text(current[0], text[start:end])])
            else:
                parts.append(current)
        return parts

    def _can_merge(self, group: _Group, block: OCRBlock, layout: str) -> bool:
        previous = group.blocks[-1]
        previous_layout = self._normalize_layout_type(previous.layout_type)
        return (
            block.page == previous.page
            and layout == previous_layout
            and layout in _TEXT_LIKE | _LIST_LIKE
        )

    def _would_exceed(self, blocks: list[OCRBlock]) -> bool:
        return len(blocks) > 1 and self._estimate_tokens(self._join_text(blocks)) > self.max_tokens

    def _estimate_tokens(self, text: str) -> int:
        if self.token_counter is not None:
            return self.token_counter(text)
        return max(1, ceil(len(text) / self.chars_per_token))

    def _renumber(self, chunks: list[Chunk]) -> list[Chunk]:
        counters: dict[int, int] = {}
        result: list[Chunk] = []
        for order, chunk in enumerate(chunks):
            index = counters.get(chunk.page, 0)
            counters[chunk.page] = index + 1
            result.append(
                Chunk(
                    chunk_id=f"{chunk.document_id}:chunk:{chunk.page}:{index}",
                    document_id=chunk.document_id,
                    page=chunk.page,
                    text=chunk.text,
                    section_id=chunk.section_id,
                    order=order,
                    bbox=chunk.bbox,
                    layout_types=chunk.layout_types,
                    metadata=chunk.metadata,
                )
            )
        return result

    @staticmethod
    def _normalize_layout_type(layout_type: str) -> str:
        normalized = layout_type.strip().lower().replace("_", "-")
        aliases = {
            "sectionheader": "Section-header",
            "section-header": "Section-header",
            "listitem": "List-item",
            "list-item": "List-item",
            "text": "Text",
            "title": "Title",
            "table": "Table",
            "formula": "Formula",
            "caption": "Caption",
            "figure": "Figure",
            "picture": "Picture",
            "footnote": "Footnote",
            "list": "List",
        }
        return aliases.get(normalized, layout_type.strip() or "Text")

    @classmethod
    def _is_empty_text_block(cls, block: OCRBlock, layout: str) -> bool:
        return not block.text.strip() and layout not in {"Figure", "Picture"}

    @staticmethod
    def _join_text(blocks: Iterable[OCRBlock]) -> str:
        return "\n".join(block.text.strip() for block in blocks)

    @staticmethod
    def _union_bbox(blocks: Iterable[OCRBlock]) -> BoundingBox | None:
        boxes = [block.bbox for block in blocks if block.bbox is not None]
        if not boxes:
            return None
        return BoundingBox(
            min(box.x0 for box in boxes),
            min(box.y0 for box in boxes),
            max(box.x1 for box in boxes),
            max(box.y1 for box in boxes),
        )

    @staticmethod
    def _copy_block_with_text(block: OCRBlock, text: str) -> OCRBlock:
        return OCRBlock(
            block_id=block.block_id,
            page=block.page,
            text=text,
            bbox=block.bbox,
            layout_type=block.layout_type,
            order=block.order,
            metadata=block.metadata,
        )
