---
type: Concept
title: Architecture
description: The prepare → transcribe → score pipeline and how the modules fit together.
tags: [project, architecture, pipeline]
timestamp: 2026-07-08
---

# Architecture

Three-stage pipeline, one command each ([CLI](../components/cli.md), `stt-eval`):

1. **prepare** — [dataset loaders](../datasets/record-model.md) download and preprocess audio
   to 16 kHz mono under `data/` (gitignored).
2. **transcribe** — each [model backend](../models/transcriber-protocol.md) runs over the
   records; [orchestration](../components/orchestration.md) writes one JSON per file to the
   committed [transcript cache](../components/transcript-cache.md). Cache-first, so
   interrupted runs resume and APIs are never double-billed.
3. **score** — [WER](../metrics/wer.md) and [medical-term recall](../metrics/medical-term-recall.md)
   read the cache offline and emit tables under `results/`.

Design threads that recur across the codebase:

- **Lazy registries** for [models](../models/transcriber-protocol.md) and
  [datasets](../datasets/record-model.md) — an unused backend's heavy SDK never loads
  (imports live inside factory closures).
- **Failure as data** — a failed file is recorded `{failed: true, ...}`, never crashes a run.
- **The reference text is embedded in each cached transcript**, so scoring needs no dataset
  access and re-scoring under new normalization never re-runs a model.

Source map: `transcribers/`, `datasets/`, `normalize.py` + vendored `_whisper_norm/`,
`metrics.py` + `score.py`, `entity_score.py` + `entity_llm.py`, `store.py`, `runner.py`,
`run.py`. Managed with uv — see [environment](environment.md).
