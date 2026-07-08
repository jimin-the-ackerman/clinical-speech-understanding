---
type: Component
title: Orchestration (runner.py)
description: Cache-first, failure-as-data transcription driver; parallel for API models, serial for GPU.
tags: [component, runner, orchestration, concurrency]
timestamp: 2026-07-08
---

# Orchestration — `runner.py`

Drives one model over a dataset's [records](../datasets/record-model.md).

- `transcribe_record(t, rec, ...)` — **cache-first**: returns `"cached"` if the JSON already
  exists (resumable, never double-bills). Else times `with_retries(t.transcribe)`, records
  `audio_seconds`/`seconds` and a UTC `created_at`, and writes via
  [store](transcript-cache.md). **Failure is data**: any exception → `{failed: true,
  error: repr(e)}` with empty text, so one bad file never crashes a run.
- `transcribe_dataset(t, records, ..., workers=8)` — a `ThreadPoolExecutor` sized
  `workers if t.parallel_safe else 1`, so [API models](../models/transcriber-protocol.md) run
  in parallel while local GPU models run serially. Prints a per-model status histogram.
