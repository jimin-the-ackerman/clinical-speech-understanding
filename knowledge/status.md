---
type: Status
title: Project status
description: Live progress and open todos for the STT benchmark and the medical-term-recall exercise.
tags: [status, todos, live]
timestamp: 2026-07-11
---

# Project status (2026-07-11)

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
- **Speaker-attribution (cpWER) probe** — Soniox × PriMock57, full n=57: flat WER 0.1224 vs
  **cpWER 0.1164** — attribution is roughly free on top of transcription (gate: **GO**). Two
  independent runs reproduce the corpus numbers; per-file diarization is nondeterministic
  (`day1_consultation11`: cpWER 0.233 vs 0.149 across draws). Throwaway probe
  (`scripts/diarize_probe.py`, spec in `docs/superpowers/specs/`), but raw transcript dumps are
  committed under `results/diarize-probe/` for sharing. Feeds the undecided scribe-architecture
  choice (flat-transcript-in vs speaker-tagged-turns-in).

- **Soniox on OSCE, with diarization (Gate B)** — 272/272 in one pass (`stt-eval transcribe
  --diarize` caches `by_speaker` beside the flat text). Flat WER **0.0971**, tied with
  qwen3-asr-0.6b (0.0968) for #1. cpWER **0.1016** — attribution costs **+0.45 pt** here, the
  *opposite* sign of PriMock57 (−0.6 pt): clean turn-taking audio leaves no interleaving for
  cpWER to forgive, so occasional clustering confusion is all that's left (cpWER ≤ flat on
  148/272; the corpus delta sits in a small outlier tail, e.g. RES0086 0.091→0.182). Verdict:
  soft GO — attribution stays cheap on both datasets.

## Open todos
1. **[openrouter](entity-methods/openrouter.md) general-LLM foil** — add `OPENROUTER_API_KEY`,
   then bake-off + build. The last method (MedGemma already answered the specialized side).
2. **[Fareez/OSCE](datasets/fareez-interviews.md) — fold the cross-family result into the
   [finding](findings/medical-term-recall.md)** — no manifest rebuild was needed (manifests
   freeze *reference* entities; Soniox only added hypotheses): `entity-score` on all four
   manifests puts **Soniox #1 on OSCE on every method**, with WER co-winner qwen3-asr-0.6b
   last — the rerank now reproduces cross-family. The finding's "qwen3-asr-1.7b #1 on OSCE
   (local models)" is superseded. gpt-4o on OSCE **skipped for now on cost** (~$19: 51.9 h ×
   $0.006/min); revisit if a second API family becomes worth the spend.
3. **Fuzzy entity matching** — current match is exact contiguous tokens (`ponytail:` note in
   `entity_score.py`); would recover reference-spelling/abbreviation misses that hit all models
   equally ("flem"→phlegm, "a and e"→A&E).
4. **Deepgram + AssemblyAI** ASR models — configured, not run (no keys).

## Phase 2 (Korean, forward look)
The harness, caching, CLI, and this metric carry over; swap WER → CER; add Korean datasets
(AI Hub, KsponSpeech) and Korean-capable APIs. Reuse Whisper / Qwen3-ASR / Soniox.
