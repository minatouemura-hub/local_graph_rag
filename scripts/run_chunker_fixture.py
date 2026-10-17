"""Run the implemented OCRDocument -> Chunk path without model or DB dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_graphrag.domain.models import BoundingBox, OCRBlock, OCRDocument, OCRPage, SourceDocument
from local_graphrag.services.layout_chunker import LayoutAwareChunker


def load_ocr_document(path: Path) -> OCRDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    document_id = payload["document_id"]
    pages = tuple(
        OCRPage(
            document_id=document_id,
            page=page_payload["page"],
            width=page_payload.get("width"),
            height=page_payload.get("height"),
            blocks=tuple(
                OCRBlock(
                    block_id=block["block_id"],
                    page=page_payload["page"],
                    text=block.get("text", ""),
                    bbox=BoundingBox(*block["bbox"]) if block.get("bbox") else None,
                    layout_type=block.get("layout_type", "Text"),
                    order=block.get("order", index),
                    metadata=block.get("metadata", {}),
                )
                for index, block in enumerate(page_payload.get("blocks", []))
            ),
        )
        for page_payload in payload.get("pages", [])
    )
    return OCRDocument(
        source=SourceDocument(
            document_id=document_id,
            path=Path(payload.get("path", f"{document_id}.pdf")),
            title=payload.get("title"),
            metadata=payload.get("metadata", {}),
        ),
        pages=pages,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=Path("tests/fixtures/sample_ocr_document.json"),
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    document = load_ocr_document(args.fixture)
    chunks = LayoutAwareChunker(max_tokens=args.max_tokens).chunk(document)
    print(json.dumps([
        {
            "chunk_id": chunk.chunk_id,
            "page": chunk.page,
            "order": chunk.order,
            "section_id": chunk.section_id,
            "text": chunk.text,
            "layout_types": chunk.layout_types,
            "bbox": chunk.bbox.__dict__ if chunk.bbox else None,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
