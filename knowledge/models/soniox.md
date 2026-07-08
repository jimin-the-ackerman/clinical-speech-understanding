---
type: Model
title: Soniox stt-async-v5 (API)
description: Soniox async STT; the noise-robustness champion and #1 on medical-term recall.
tags: [model, api, soniox, korean-ready]
timestamp: 2026-07-08
---

# Soniox `stt-async-v5` — API

`transcribers/soniox_api.py`, `parallel_safe = True`, needs `SONIOX_API_KEY`. Async 4-endpoint
flow: upload file → create transcription → `poll_until` done → GET transcript.

- **Self-cleaning**: a `finally` block DELETEs the stored transcription + file (best-effort) so
  uploads don't accumulate against the account's file quota — the local
  [cache](../components/transcript-cache.md) is the retained copy.
- **Result**: most white-noise-robust of the six models, and **#1 on
  [medical-term recall](../metrics/medical-term-recall.md)** on PriMock57 across all four entity
  methods (WER ranks it #2). Slowest wall-clock (async API). Korean-capable → reusable in phase 2.
