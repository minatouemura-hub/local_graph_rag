"""Sweep real Qwen generation over 1..5 retrieved chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_graphrag.adapters.generator import LlamaCppGenerator
from local_graphrag.domain.models import Chunk, ContextItem, RetrievalContext
from local_graphrag.services.context_packer import estimate_token_count


def load_context(report_path: Path) -> RetrievalContext:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = tuple(
        ContextItem(
            Chunk(f"real-{index}", Path(report["pdf"]).stem, item["page"], item["text"], None, index),
            1.0 / index,
            (),
            item["reason"],
        )
        for index, item in enumerate(report["context"], start=1)
    )
    return RetrievalContext(report["query"], items)


def run(report_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = load_context(report_path)
    results: list[dict[str, object]] = []
    for count in range(1, min(5, len(context.items)) + 1):
        current = RetrievalContext(context.query, context.items[:count])
        generator = LlamaCppGenerator(context_size=4096, max_tokens=384, max_context_chars=100000)
        generator.load()
        answer = generator.generate(context.query, current)
        generator.unload()
        diagnostics = answer.metadata["diagnostics"]
        (output_dir / f"chunks-{count}-stdout.txt").write_text(
            diagnostics["raw_stdout"], encoding="utf-8"
        )
        (output_dir / f"chunks-{count}-stderr.txt").write_text(
            diagnostics["raw_stderr"], encoding="utf-8"
        )
        results.append(
            {
                "chunk_count": count,
                "prompt_chars": diagnostics["prompt_chars"],
                "prompt_tokens": diagnostics["prompt_tokens"],
                "retrieval_tokens": sum(estimate_token_count(i.chunk.text) for i in current.items),
                "context_window": diagnostics["context_window"],
                "max_output_tokens": diagnostics["max_output_tokens"],
                "return_code": diagnostics["return_code"],
                "generated_tokens": diagnostics["generated_tokens"],
                "empty_answer": not bool(answer.text.strip()),
                "elapsed_seconds": diagnostics["elapsed_seconds"],
                "stdout_length": diagnostics["stdout_length"],
                "stderr_length": diagnostics["stderr_length"],
                "answer": answer.text,
            }
        )
    report = {
        "model": "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M",
        "context_window": 4096,
        "max_output_tokens": 384,
        "thinking": False,
        "source_report": str(report_path),
        "query": context.query,
        "sweep": results,
    }
    (output_dir / "context_sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("data/experiments/real-models-run7/real_models_report.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/qwen-context-sweep"))
    args = parser.parse_args()
    report = run(args.source_report, args.output_dir)
    for result in report["sweep"]:
        print(
            f"chunks={result['chunk_count']} prompt_tokens={result['prompt_tokens']} "
            f"return={result['return_code']} generated={result['generated_tokens']} "
            f"empty={result['empty_answer']} elapsed={result['elapsed_seconds']:.2f}s"
        )
    print("Report:", args.output_dir / "context_sweep_report.json")


if __name__ == "__main__":
    main()
