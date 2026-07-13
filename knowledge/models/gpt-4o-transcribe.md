---
type: Model
title: gpt-4o-transcribe (OpenAI API)
description: OpenAI gpt-4o-transcribe (direct or via OpenRouter); strong on clean audio, collapses under white noise, drops mid-audio passages on long files.
tags: [model, api, openai, openrouter]
timestamp: 2026-07-13
---

# gpt-4o-transcribe — OpenAI API

`transcribers/openai_api.py`, `parallel_safe = True`, needs `OPENAI_API_KEY`.

- POSTs multipart to `/v1/audio/transcriptions`. Files > 25 MB are re-encoded to FLAC in memory
  (`_to_flac`); still-oversize errors (chunking not implemented).
- **Second route** (`transcribers/openrouter_api.py`, registry name
  `gpt-4o-transcribe-openrouter`, needs `OPENROUTER_API_KEY`): same model through OpenRouter's
  OpenAI-compatible multipart `/api/v1/audio/transcriptions` (their chat-completions
  `input_audio` route 400s for this model). Token-level billing on OpenRouter credits ran
  cheaper than OpenAI's $0.006/min convention. Separate registry entry for provenance — the two
  routes give near-but-not-identical transcripts (nondeterministic decode). Used for the OSCE
  pass; the *diarize* variant is OpenAI-direct only.
- **Behavior**: best-in-class on clean/easy audio, but **hallucinates fluent nonsense under
  white noise** (WER 41 → 96 → 97%, medical-term recall → ~0) and is *last* on both real
  clinical corpora. Its clinical WER is **deletion-shaped**: on long-form audio it silently
  drops mid-audio passages (24/272 OSCE and 12/57 PriMock57 files lose >20% of transcript
  length, on both routes) — for a scribe, whole-passage loss is the worst failure mode. See the
  [finding](../findings/medical-term-recall.md).
- Note: gpt-4o-transcribe is a **scored ASR system**, which is why no OpenAI model is used as an
  [entity-extraction](../entity-methods/openrouter.md) judge.
