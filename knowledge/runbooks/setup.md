---
type: Runbook
title: Setup
description: Install with uv and configure API keys before running the benchmark.
tags: [runbook, setup, uv]
timestamp: 2026-07-08
---

# Setup

Requires Python ≥3.10, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and `git`/`git-lfs` on PATH.

```bash
uv sync                    # core (scoring + API transcribers)
uv sync --extra data       # + HF datasets (MedDialog-Audio prepare)
uv sync --extra local      # + faster-whisper / transformers / torch / bitsandbytes (GPU + MedGemma)
uv sync --extra entities   # + scispaCy (bc5cdr entity method)
uv run pytest              # offline sanity — no network/GPU/keys
```

API keys (a missing key skips that model with a warning): `OPENAI_API_KEY`, `SONIOX_API_KEY`,
`DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY`. Pass them with `uv run --env-file .env <cmd>`.

See [environment](../project/environment.md) for the extras and the CUDA-12.6 torch pin, and the
[GPU benchmark runbook](gpu-benchmark.md) for the full run. Full uv primer: `README.md`.
