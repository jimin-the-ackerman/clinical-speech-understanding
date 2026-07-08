---
type: Concept
title: Environment & packaging
description: uv-managed env, the data/local/entities extras, and the CUDA-12.6 torch pin.
tags: [project, environment, uv, cuda, packaging]
timestamp: 2026-07-08
---

# Environment & packaging

Managed with **uv** (one venv per project; `uv run` syncs then runs — you never activate).
Core deps stay light so the test suite is offline; capabilities are gated behind optional
extras:

- **`data`** — `datasets[audio]`; needed for [MedDialog-Audio](../datasets/meddialog-audio.md)
  `prepare()`.
- **`local`** — `faster-whisper, transformers>=5.13, torch, accelerate, torchcodec,
  bitsandbytes`; the local GPU models + 4-bit [MedGemma](../entity-methods/medgemma.md).
  `bitsandbytes` lives HERE, not in a `uv run --with` overlay — an overlay re-resolves torch
  off default PyPI and breaks the cu126 pin below.
- **`entities`** — `scispacy`; the [bc5cdr](../entity-methods/bc5cdr.md) method (its NER model
  wheel is installed separately). [med7](../entity-methods/med7.md) and
  [stanza-i2b2](../entity-methods/stanza-i2b2.md) run via ephemeral `uv run --with` overlays.

**The one non-obvious build detail:** `[tool.uv.sources]` + `[[tool.uv.index]]` pin
`torch`/`torchcodec` to the **CUDA-12.6** wheel index (`pytorch-cu126`), not default PyPI.
Default PyPI torch is a CUDA-13 build needing driver ≥580; this machine's driver (~555) and
CTranslate2/faster-whisper both need CUDA-12 libs. See the
[GPU benchmark runbook](../runbooks/gpu-benchmark.md).
