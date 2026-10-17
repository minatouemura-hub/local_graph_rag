"""Run one local PDF through OCR, chunking, Ladybug indexing, and retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import pymupdf

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
from local_graphrag.services.layout_chunker import LayoutAwareChunker
from local_graphrag.services.retrieval import HybridRetriever


class HashEmbedder:
    """Small deterministic local embedder for infrastructure-only validation."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.replace("\n", " ").split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


def ocr_pdf(pdf_path: Path, ocr_json: Path, tesseract: str) -> OCRDocument:
    source = SourceDocument(pdf_path.stem, pdf_path)
    pages: list[OCRPage] = []
    with (
        pymupdf.open(pdf_path) as pdf,
        tempfile.TemporaryDirectory(prefix="local-graphrag-ocr-") as temp_dir,
    ):
        for page_number, page in enumerate(pdf, start=1):
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image_path = Path(temp_dir) / f"page-{page_number:04d}.png"
            pixmap.save(str(image_path))
            result = subprocess.run(
                [tesseract, str(image_path), "stdout", "-l", "jpn+eng", "--psm", "3"],
                check=True,
                capture_output=True,
                text=True,
            )
            text = result.stdout.strip() or page.get_text().strip()
            block = OCRBlock(
                block_id=f"{source.document_id}:page:{page_number}:ocr:0",
                page=page_number,
                text=text,
                bbox=BoundingBox(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                layout_type="Text",
                order=0,
                metadata={"backend": "tesseract", "language": "jpn+eng"},
            )
            pages.append(
                OCRPage(
                    document_id=source.document_id,
                    page=page_number,
                    width=pixmap.width,
                    height=pixmap.height,
                    blocks=(block,),
                )
            )
    document = OCRDocument(source, tuple(pages))
    ocr_json.parent.mkdir(parents=True, exist_ok=True)
    ocr_json.write_text(
        json.dumps(_document_json(document), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return document


def _document_json(document: OCRDocument) -> dict:
    return {
        "document_id": document.source.document_id,
        "path": str(document.source.path),
        "pages": [
            {
                "page": page.page,
                "width": page.width,
                "height": page.height,
                "blocks": [
                    {
                        "block_id": block.block_id,
                        "page": block.page,
                        "text": block.text,
                        "bbox": [block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1]
                        if block.bbox
                        else None,
                        "layout_type": block.layout_type,
                        "order": block.order,
                        "metadata": block.metadata,
                    }
                    for block in page.blocks
                ],
            }
            for page in document.pages
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("data/raw/tbio.pdf"))
    parser.add_argument("--query", default="線維束性収縮の検出方法について")
    parser.add_argument("--tesseract", default="/opt/homebrew/bin/tesseract")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ocr/tbio-e2e"))
    args = parser.parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    document = ocr_pdf(args.pdf, args.output_dir / "ocr_document.json", args.tesseract)
    chunks = LayoutAwareChunker(max_tokens=512).chunk(document)
    embedder = HashEmbedder(64)
    db_dir = args.output_dir / "ladybug"
    connection = LadybugConnection(db_dir)
    connection.initialize(dimension=embedder.dimension)
    chunk_store = LadybugChunkStore(connection)
    vector_store = LadybugVectorStore(connection)
    full_text_store = LadybugFullTextStore(connection)
    graph_store = LadybugGraphStore(connection)
    chunk_store.upsert(chunks)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    vector_store.upsert(
        [EmbeddedChunk(chunk, tuple(vector)) for chunk, vector in zip(chunks, vectors)]
    )
    full_text_store.upsert(chunks)
    graph_store.upsert_document_structure(document.source, chunks)
    retriever = HybridRetriever(embedder, vector_store, full_text_store, graph_store)
    context = retriever.retrieve(args.query, limit=5, candidate_limit=10)

    print(f"PDF: {args.pdf}")
    print(f"Pages: {len(document.pages)}")
    print(f"OCR blocks: {sum(len(page.blocks) for page in document.pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Query: {args.query}")
    print("RetrievalContext:")
    for index, item in enumerate(context.items, start=1):
        preview = item.chunk.text.replace("\n", " ")[:180]
        print(f"{index}. page={item.chunk.page} reason={item.reason} text={preview}")


if __name__ == "__main__":
    main()
