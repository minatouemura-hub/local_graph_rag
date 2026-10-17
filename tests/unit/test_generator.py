from __future__ import annotations

from types import SimpleNamespace

from local_graphrag.adapters import generator as generator_module
from local_graphrag.adapters.generator import LlamaCppGenerator
from local_graphrag.domain.models import Chunk, ContextItem, RetrievalContext


def context() -> RetrievalContext:
    chunk = Chunk("c1", "doc", 1, "PIVで線維束性収縮を検出する。", None, 0)
    return RetrievalContext("検出方法は？", (ContextItem(chunk, 1.0, ()),))


def test_generator_records_raw_streams_and_no_think(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            stdout="<|im_start|>assistant\n回答です。",
            stderr="prompt log",
            returncode=0,
        )

    monkeypatch.setattr(generator_module.subprocess, "run", fake_run)
    generator = LlamaCppGenerator(max_tokens=32)
    generator.load()
    answer = generator.generate("検出方法は？", context())
    diagnostics = answer.metadata["diagnostics"]
    assert answer.text == "回答です。"
    assert diagnostics["return_code"] == 0
    assert diagnostics["raw_stdout"].startswith("<|im_start|>")
    assert diagnostics["raw_stderr"] == "prompt log"
    assert diagnostics["no_think_applied"] is True
    assert diagnostics["prompt_tokens"] > 0


def test_generator_keeps_empty_output_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        generator_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="EOS", returncode=0),
    )
    generator = LlamaCppGenerator(max_tokens=32)
    generator.load()
    answer = generator.generate("質問", context())
    assert answer.text == "EOS"
    assert answer.metadata["diagnostics"]["stdout_length"] == 0
    assert answer.metadata["diagnostics"]["stderr_length"] == 3
