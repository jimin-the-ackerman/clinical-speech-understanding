---
type: Model
title: gpt-4o-transcribe (OpenAI API)
description: OpenAI gpt-4o-transcribe; strong on clean audio, collapses under white noise.
tags: [model, api, openai]
timestamp: 2026-07-08
---

# gpt-4o-transcribe — OpenAI API

`transcribers/openai_api.py`, `parallel_safe = True`, needs `OPENAI_API_KEY`.

- POSTs multipart to `/v1/audio/transcriptions`. Files > 25 MB are re-encoded to FLAC in memory
  (`_to_flac`); still-oversize errors (chunking not implemented).
- **Behavior**: best-in-class on clean/easy audio, but **hallucinates fluent nonsense under
  white noise** (WER 41 → 96 → 97%, medical-term recall → ~0) and is *last* on real PriMock57
  consultations — a real caution for a scribe. See the [finding](../findings/medical-term-recall.md).
- Note: gpt-4o-transcribe is a **scored ASR system**, which is why no OpenAI model is used as an
  [entity-extraction](../entity-methods/openrouter.md) judge.
