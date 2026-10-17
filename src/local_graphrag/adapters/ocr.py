from __future__ import annotations

import subprocess

from local_graphrag.domain.models import OCRBlock, OCRPage, SourceDocument
from local_graphrag.ports.interfaces import OCRBackend


class DotsOCRBackend(OCRBackend):
    """Local dots.ocr GGUF adapter using llama.cpp's multimodal CLI.

    The current dots.ocr GGUF emits Markdown-like text rather than the normalized
    JSON schema. This adapter keeps that output as one page-level Text block; a
    later parser can split headings, tables, and bounding boxes without changing
    the runtime boundary.
    """

    def __init__(
        self,
        command: str = "/opt/homebrew/bin/llama-mtmd-cli",
        model: str = "ggml-org/dots.ocr-GGUF:Q8_0",
        prompt: str = "Read this document image and transcribe all visible text in reading order. Preserve tables and headings in Markdown.",
        max_tokens: int = 2048,
    ):
        self.command, self.model = command, model
        self.prompt, self.max_tokens = prompt, max_tokens

    def parse_page(self, source: SourceDocument, page_image: str, page_number: int) -> OCRPage:
        args = [self.command, "-hf", self.model, "--image", page_image, "-p", self.prompt, "-n", str(self.max_tokens), "-ngl", "all"]
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        text = result.stdout.strip()
        block = OCRBlock(
            block_id=f"{source.document_id}-p{page_number}-dotsocr-0",
            page=page_number,
            text=text,
            bbox=None,
            layout_type="Text",
            order=0,
            metadata={"backend": "dots.ocr", "model": self.model},
        )
        return OCRPage(source.document_id, page_number, None, None, (block,))
