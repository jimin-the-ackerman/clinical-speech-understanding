---
type: Component
title: Transcript cache (store.py)
description: Per-file JSON transcript cache with atomic writes and collision-safe ids; the resume/no-rebill seam.
tags: [component, cache, store, resumability]
timestamp: 2026-07-08
---

# Transcript cache — `store.py`

The committed cache at `results/transcripts/{dataset}/{model}/{file_id}.json`. One JSON per
transcription; the **reference text is embedded**, so [scoring](../metrics/wer.md) needs no
dataset access and re-scoring never re-runs a model.

- `cache_path(root, dataset, model, file_id)` — the path layout above.
- `safe_id(file_id)` — sanitize non-`[A-Za-z0-9._-]` to `_`; if changed, append an 8-char sha1
  so distinct ids never collide on disk.
- `write_result(path, payload)` — **atomic** (`.tmp` + `os.replace`), pretty UTF-8 JSON.
- `read_results(root)` — sorted `rglob("*.json")`, yields parsed dicts. The single read seam
  for [WER](../metrics/wer.md) and [medical-term recall](../metrics/medical-term-recall.md).

"File exists = done" is what makes runs resumable and APIs never double-billed (see
[orchestration](orchestration.md)). The same atomic-cache pattern backs the per-reference LLM
entity cache used by [medgemma](../entity-methods/medgemma.md)/[openrouter](../entity-methods/openrouter.md).
