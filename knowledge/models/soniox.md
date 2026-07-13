---
type: Model
title: Soniox stt-async-v5 (API)
description: Soniox async STT; the noise-robustness champion, #1 on medical-term recall on both clinical corpora, and the benchmark's only diarizing backend.
tags: [model, api, soniox, korean-ready, diarization]
timestamp: 2026-07-13
---

# Soniox `stt-async-v5` — API

`transcribers/soniox_api.py`, `parallel_safe = True`, needs `SONIOX_API_KEY`. Async 4-endpoint
flow: upload file → create transcription → `poll_until` done → GET transcript.

- **Self-cleaning**: a `finally` block DELETEs the stored transcription + file (best-effort) so
  uploads don't accumulate against the account's file quota — the local
  [cache](../components/transcript-cache.md) is the retained copy.
- **Diarization** (opt-in): `stt-eval transcribe --diarize` sets the `diarize` class flag →
  request carries `enable_speaker_diarization`, each token carries a `speaker`, and
  `transcribe` returns `(text, {"by_speaker": ...})` — the buckets are cached beside the flat
  text and scored offline by `scripts/score_cpwer.py`. The only diarizing backend in the lineup;
  attribution ≈ free (see the [attribution finding](../findings/speaker-attribution-cost.md)).
  Clustering is nondeterministic per file; recognition is stable.
- **Result**: most white-noise-robust of the six models, and **#1 on
  [medical-term recall](../metrics/medical-term-recall.md) on both clinical corpora**
  (PriMock57 and OSCE/Fareez) across all four entity methods; on WER it's #2 on PriMock57 and
  co-#1 on OSCE. Slowest wall-clock (async API). Korean-capable → reusable in phase 2.
