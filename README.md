# Local GraphRAG

Fully local, layout-aware OCR GraphRAG for Apple Silicon edge devices. The project preserves OCR
document structure—pages, sections, reading order, layout types, bounding boxes, and source block
IDs—and combines dense retrieval, BM25 full-text retrieval, and structural graph traversal.

No hosted OCR, embedding, vector database, LLM API, or cloud graph database is required at runtime.

## Verified pipeline

```text
PDF -> local OCR -> OCRDocument -> layout-aware chunks
    -> multilingual-e5-small -> embedded LadybugDB
    -> HNSW + BM25 + RRF + graph expansion
    -> TokenBudgetContextPacker -> Qwen3 via llama.cpp
```

The current real-PDF experiment uses `data/raw/tbio.pdf`: 13 pages, 23 chunks, local E5
embeddings, LadybugDB retrieval, and local Qwen3 generation.

## Components

| Area | Current implementation |
|---|---|
| OCR | `ggml-org/dots.ocr-GGUF:Q8_0` with `llama-mtmd-cli` |
| Embedding | `intfloat/multilingual-e5-small`, 384 dimensions |
| Database | LadybugDB embedded Python package |
| Dense retrieval | HNSW cosine similarity |
| Sparse retrieval | LadybugDB FTS / BM25 |
| Graph retrieval | `HAS_PAGE`, `HAS_SECTION`, `HAS_CHUNK`, `NEXT` |
| Generator | `ggml-org/Qwen3-1.7B-GGUF:Q4_K_M` with llama.cpp |
| Generation | Qwen3 non-thinking mode (`/no_think`) |

MVP graph nodes are `Document`, `Page`, `Section`, and `Chunk`. OCR layout output is normalized
through domain DTOs before chunking or persistence.

## Setup

Python 3.11 or newer is required. A dedicated conda environment is recommended on macOS:

```bash
conda create -n local-graphrag -c conda-forge -y python=3.11 numpy=1.26
conda run -n local-graphrag python -m pip install -e ".[embedding,graph,dev]"
brew install llama.cpp
```

For the Apple Silicon PyTorch environment used by the real-model experiment:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="$PWD/data/models/huggingface"
```

Internet access is needed only to install packages and download models. Runtime OCR, indexing,
retrieval, and generation are local.

## Fixture retrieval demo

This validates chunking, LadybugDB, HNSW, BM25, RRF, and graph expansion without model downloads:

```bash
PYTHONPATH=src:scripts \
conda run -n local-graphrag python scripts/run_retrieval_fixture.py
```

## Real PDF experiment

The script runs dots.ocr on page 1, uses local OCR text for the full PDF indexing pass, embeds with
E5, retrieves with LadybugDB, packs context by token budget, and generates with Qwen3:

```bash
PYTHONPATH=src:scripts \
conda run -n local-graphrag python scripts/run_real_models_experiment.py \
  --output-dir data/experiments/real-models-packed2
```

Important artifacts are:

```text
data/experiments/real-models-packed2/real_models_report.json
data/experiments/real-models-packed2/dots_ocr_page1.txt
data/experiments/real-models-packed2/generator_raw_stdout.txt
data/experiments/real-models-packed2/generator_raw_stderr.txt
```

## Qwen context sweep

The sweep tests one through five retrieved chunks and records prompt estimates, return codes,
generated token estimates, raw streams, and elapsed time:

```bash
PYTHONPATH=src:scripts \
conda run -n local-graphrag python scripts/run_qwen_context_sweep.py \
  --source-report data/experiments/real-models-run7/real_models_report.json \
  --output-dir data/experiments/qwen-context-sweep-fixed
```

The current experiment found that one or two chunks are more stable than three or more chunks for
Qwen3-1.7B. The 8 GB profile therefore uses a conservative safety margin and whole-chunk packing.

## Context packing and diagnostics

`TokenBudgetContextPacker` removes lower-ranked chunks when the retrieval budget is full. If the
first chunk alone is too large, it is truncated and marked with `packed_truncated` metadata.

`LlamaCppGenerator` records prompt characters/tokens, context window, output limit, return code,
elapsed time, generated token estimate, raw stdout/stderr, final text, and `/no_think` status.
Without a model tokenizer, token counts are explicitly labeled as conservative estimates.

## Tests

```bash
conda run -n base python -m pytest -q
conda run -n base python -m ruff check src tests scripts
conda run -n base python -m compileall -q src tests scripts
conda run -n base python -m mypy src
```

Latest verification: 23 tests passed, ruff passed, compileall passed, and mypy passed.

## Repository layout

```text
src/local_graphrag/  domain DTOs, ports, adapters, services, pipelines
scripts/             fixture, PDF, model, and benchmark runners
tests/               unit and LadybugDB integration tests
configs/             edge profiles and chunking settings
data/raw/            input PDFs and images
data/experiments/    generated experiment artifacts
logs/                implementation and experiment logs
```

## Limitations

- The dots.ocr adapter currently keeps its Markdown-like page output as a normalized page-level
  text block; fine-grained dots.ocr layout/bbox parsing remains future work.
- The full-PDF experiment currently uses local Tesseract for all pages and dots.ocr for a page-1
  model experiment.
- Japanese OCR errors can affect retrieval and answer wording.
- Token counts are estimates unless a compatible tokenizer is supplied.
- Entity extraction, entity relations, PDF highlighting UI, and answer-quality evaluation are not
  implemented yet.
- Downloaded models and generated databases are local artifacts and should not be committed.

This is an MVP / research prototype. Check each downloaded model and runtime license before
redistribution or commercial use.
