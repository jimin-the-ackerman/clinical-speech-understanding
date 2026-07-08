---
type: Component
title: CLI (run.py)
description: The stt-eval command — prepare, transcribe, score, entity-build, entity-score, entity-bakeoff.
tags: [component, cli, run]
timestamp: 2026-07-08
---

# CLI — `stt-eval` (`run.py`)

`build_parser()` defines the subcommands; every backend import is lazy inside `main()` so
unused extras never load. Top-level `--results-dir` (default `results/`).

- **`prepare --datasets`** — download/preprocess [datasets](../datasets/record-model.md).
- **`transcribe --models --datasets [--workers 8] [--limit N]`** — run models; a model whose
  API key is missing is **skipped with a warning** (`MissingKeyError`), not an error. Drives
  [orchestration](orchestration.md).
- **`score`** — [WER](../metrics/wer.md) tables from the cache.
- **`entity-build --method X [--model ID] [--workers N] [--limit N] [--datasets ...]`** — freeze
  reference entities to a manifest (LLM methods get a resumable per-ref `cache_dir`); see
  [medical-term recall](../metrics/medical-term-recall.md).
- **`entity-score --manifest P`** — offline recall table.
- **`entity-bakeoff --specs method[:model],... [--limit 15]`** — side-by-side entity sets for a
  human pick.

Tests: `uv run pytest` — no network, GPU, or keys.
