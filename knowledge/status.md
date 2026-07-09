---
type: Status
title: Project status
description: Live progress and open todos for the STT benchmark and the medical-term-recall exercise.
tags: [status, todos, live]
timestamp: 2026-07-08
---

# Project status (2026-07-09)

## Done
- **English WER round** — 6 models × 3 datasets (LibriSpeech, MedDialog-Audio, PriMock57), zero
  failures, sanity anchors pass. Tables in `results/wer_summary.{csv,md}`.
- **Medical-term recall** — four entity methods built and agreeing (Soniox #1 on PriMock57):
  [bc5cdr](entity-methods/bc5cdr.md), [med7](entity-methods/med7.md),
  [stanza-i2b2](entity-methods/stanza-i2b2.md), [medgemma](entity-methods/medgemma.md). See the
  [finding](findings/medical-term-recall.md).
- **OSCE / Fareez (local models)** — the four local models transcribed and scored on 272 OSCE
  consultations; WER + all four entity manifests now include OSCE, and the recall rerank
  [reproduces](findings/medical-term-recall.md) (qwen3-asr-1.7b #1 on every method). A UTF-16
  loader bug (2 mis-decoded transcripts, `RES0002`/`RES0054`) was found and fixed along the way.

## Open todos
1. **[openrouter](entity-methods/openrouter.md) general-LLM foil** — add `OPENROUTER_API_KEY`,
   then bake-off + build. The last method (MedGemma already answered the specialized side).
2. **[Fareez/OSCE](datasets/fareez-interviews.md) — metered APIs** — local models done and scored.
   Still pending: Soniox + gpt-4o on OSCE (then rebuild the manifests once more) to turn the
   local-only replication into a full cross-family comparison.
3. **Fuzzy entity matching** — current match is exact contiguous tokens (`ponytail:` note in
   `entity_score.py`); would recover reference-spelling/abbreviation misses that hit all models
   equally ("flem"→phlegm, "a and e"→A&E).
4. **Deepgram + AssemblyAI** ASR models — configured, not run (no keys).

## Phase 2 (Korean, forward look)
The harness, caching, CLI, and this metric carry over; swap WER → CER; add Korean datasets
(AI Hub, KsponSpeech) and Korean-capable APIs. Reuse Whisper / Qwen3-ASR / Soniox.
