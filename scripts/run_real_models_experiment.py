"""Run the real local E5, dots.ocr, and Qwen components on one PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pymupdf
from run_pdf_e2e import ocr_pdf

from local_graphrag.adapters.embedding import E5Embedder
from local_graphrag.adapters.generator import LlamaCppGenerator
from local_graphrag.adapters.ladybug import (
    LadybugChunkStore,
    LadybugConnection,
    LadybugFullTextStore,
    LadybugGraphStore,
    LadybugVectorStore,
)
from local_graphrag.adapters.ocr import DotsOCRBackend
from local_graphrag.domain.models import EmbeddedChunk, SourceDocument
from local_graphrag.services.context_packer import TokenBudgetContextPacker, estimate_token_count
from local_graphrag.services.layout_chunker import LayoutAwareChunker
from local_graphrag.services.retrieval import HybridRetriever


def run(pdf: Path, output_dir: Path, query: str, tesseract: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = SourceDocument(pdf.stem, pdf)

    # dots.ocr connectivity/quality experiment on page 1.
    page_image = output_dir / "page-1.png"
    with pymupdf.open(pdf) as document:
        page = document[0]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        pixmap.save(str(page_image))
    dots_page = DotsOCRBackend().parse_page(source, str(page_image), 1)
    dots_text = dots_page.blocks[0].text if dots_page.blocks else ""
    (output_dir / "dots_ocr_page1.txt").write_text(dots_text, encoding="utf-8")

    # Full-PDF retrieval uses local Tesseract blocks so the experiment remains
    # bounded; E5 and Qwen are the actual production model components here.
    document = ocr_pdf(pdf, output_dir / "ocr_document.json", tesseract)
    chunks = LayoutAwareChunker(max_tokens=512).chunk(document)
    embedder = E5Embedder(device="cpu")
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    connection = LadybugConnection(output_dir / "ladybug-e5")
    connection.initialize(dimension=embedder.dimension)
    LadybugChunkStore(connection).upsert(chunks)
    LadybugVectorStore(connection).upsert(
        [EmbeddedChunk(chunk, tuple(vector)) for chunk, vector in zip(chunks, vectors)]
    )
    LadybugFullTextStore(connection).upsert(chunks)
    LadybugGraphStore(connection).upsert_document_structure(document.source, chunks)
    retriever = HybridRetriever(
        embedder,
        LadybugVectorStore(connection),
        LadybugFullTextStore(connection),
        LadybugGraphStore(connection),
    )
    context = retriever.retrieve(query, limit=5, candidate_limit=10)

    generator = LlamaCppGenerator(context_size=4096, max_tokens=384, max_context_chars=100000)
    packer = TokenBudgetContextPacker()
    packing_budget = 4096 - 384 - 1024 - 128 - estimate_token_count(query)
    packed_context = packer.pack(query, context, packing_budget)
    generator.load()
    answer = generator.generate(query, packed_context)
    generator.unload()
    diagnostics = answer.metadata.get("diagnostics", {})
    (output_dir / "generator_raw_stdout.txt").write_text(
        str(diagnostics.get("raw_stdout", "")), encoding="utf-8"
    )
    (output_dir / "generator_raw_stderr.txt").write_text(
        str(diagnostics.get("raw_stderr", "")), encoding="utf-8"
    )
    report = {
        "pdf": str(pdf),
        "pages": len(document.pages),
        "chunks": len(chunks),
        "embedding": {"model": embedder.model_name, "dimension": embedder.dimension},
        "dots_ocr": {"model": "ggml-org/dots.ocr-GGUF:Q8_0", "page": 1, "text": dots_text},
        "query": query,
        "context_packing": {
            "budget": packing_budget,
            "before_chunks": len(context.items),
            "after_chunks": len(packed_context.items),
            "before_tokens_estimated": sum(estimate_token_count(i.chunk.text) for i in context.items),
            "after_tokens_estimated": sum(estimate_token_count(i.chunk.text) for i in packed_context.items),
        },
        "context": [
            {"page": item.chunk.page, "reason": item.reason, "text": item.chunk.text}
            for item in context.items
        ],
        "packed_context": [
            {"page": item.chunk.page, "reason": item.reason, "text": item.chunk.text}
            for item in packed_context.items
        ],
        "answer": answer.text,
        "empty_answer": not bool(answer.text.strip()),
        "generator": answer.metadata,
    }
    (output_dir / "real_models_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=Path("data/raw/tbio.pdf"))
    parser.add_argument("--query", default="線維束性収縮の検出方法について")
    parser.add_argument("--tesseract", default="/opt/homebrew/bin/tesseract")
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/real-models"))
    args = parser.parse_args()
    report = run(args.pdf, args.output_dir, args.query, args.tesseract)
    print(f"PDF: {report['pdf']}")
    print(f"Pages: {report['pages']}  Chunks: {report['chunks']}")
    print("dots.ocr page 1:", str(report["dots_ocr"]["text"])[:500].replace("\n", " "))
    print("RetrievalContext:")
    for index, item in enumerate(report["context"], start=1):
        print(f"{index}. page={item['page']} reason={item['reason']} text={item['text'][:180]}")
    print("Qwen answer:", report["answer"])
    print("Report:", args.output_dir / "real_models_report.json")


if __name__ == "__main__":
    main()
