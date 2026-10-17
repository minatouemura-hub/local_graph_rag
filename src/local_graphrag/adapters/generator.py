from __future__ import annotations

import subprocess
import time
from typing import Any

from local_graphrag.domain.models import GeneratedAnswer, RetrievalContext
from local_graphrag.local_graphrag_error import AdapterError
from local_graphrag.ports.interfaces import Generator
from local_graphrag.services.context_packer import estimate_token_count


class LlamaCppGenerator(Generator):
    def __init__(
        self,
        command: str = "/opt/homebrew/bin/llama-cli",
        model: str = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M",
        thinking: bool = False,
        context_size: int = 4096,
        max_tokens: int = 256,
        max_context_chars: int = 2500,
    ):
        self.command, self.model, self.thinking = command, model, thinking
        self.context_size, self.max_tokens = context_size, max_tokens
        self.max_context_chars = max_context_chars
        self._loaded = False

    def load(self) -> None: self._loaded = True
    def unload(self) -> None: self._loaded = False

    def generate(self, query: str, context: RetrievalContext) -> GeneratedAnswer:
        if not self._loaded:
            raise AdapterError("generator is not loaded")
        evidence = "\n\n".join(
            f"[page {item.chunk.page}]\n{item.chunk.text}" for item in context.items
        )
        evidence = evidence[: self.max_context_chars]
        mode = "" if self.thinking else "/no_think"
        prompt = (
            "<|im_start|>system\n"
            "あなたは文書検索アシスタントです。根拠にないことは推測せず、回答にはページ番号を付けてください。\n"
            f"{mode}<|im_end|>\n"
            f"<|im_start|>user\n根拠:\n{evidence}\n\n質問: {query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        args = [self.command]
        if "/" in self.model and not self.model.endswith(".gguf"):
            args.extend(["-hf", self.model])
        else:
            args.extend(["-m", self.model])
        args.extend(
            [
                "-ngl",
                "all",
                "-c",
                str(self.context_size),
                "-n",
                str(self.max_tokens),
                "--no-display-prompt",
                "--no-show-timings",
                "--simple-io",
                "--color",
                "off",
                "-p",
                prompt,
            ]
        )
        started = time.perf_counter()
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.perf_counter() - started
        raw_stdout, raw_stderr = result.stdout, result.stderr
        output = f"{raw_stdout}\n{raw_stderr}"
        marker = "<|im_start|>assistant\n"
        if marker in raw_stdout:
            output = raw_stdout.rsplit(marker, 1)[1]
        elif marker in raw_stderr:
            output = raw_stderr.rsplit(marker, 1)[1]
        elif "... (truncated)" in raw_stdout:
            output = raw_stdout.rsplit("... (truncated)", 1)[1]
        elif "<|im_end|>" in raw_stdout:
            output = raw_stdout.rsplit("<|im_end|>", 1)[1]
        elif "available commands:" in output:
            output = ""
        output = output.split("\n> ", 1)[0].replace("Exiting...", "").strip()
        diagnostics: dict[str, Any] = {
            "prompt_chars": len(prompt),
            "prompt_tokens": estimate_token_count(prompt),
            "token_count_method": "conservative_approximation",
            "context_window": self.context_size,
            "max_output_tokens": self.max_tokens,
            "return_code": result.returncode,
            "elapsed_seconds": elapsed,
            "generated_tokens": estimate_token_count(output),
            "stdout_length": len(raw_stdout),
            "stderr_length": len(raw_stderr),
            "raw_stdout": raw_stdout,
            "raw_stderr": raw_stderr,
            "final_text": output,
            "thinking": self.thinking,
            "no_think_applied": not self.thinking,
            "command": args[:3],
        }
        return GeneratedAnswer(
            output,
            tuple(c for item in context.items for c in item.citations),
            {"model": self.model, "thinking": self.thinking, "diagnostics": diagnostics},
        )
