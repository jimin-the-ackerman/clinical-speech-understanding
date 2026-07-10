---
type: Plan
title: Research plan — clinical STT benchmark, English proving ground then Korean
description: The questions this benchmark answers, the design that answers them, and the phased roadmap from the English round to the Korean scribe.
tags: [plan, research-agenda, roadmap, asr, evaluation]
timestamp: 2026-07-10
---

# Research plan

## Why this project

The end goal is a Korean transcription system for healthcare conversations (doctor–nurse–patient,
an "AI scribe"). **Phase 1 benchmarks existing STT models on English medical audio** to settle the
harness, the scoring conventions, and a model shortlist before the Korean phase — where paid data
and API costs make it expensive to still be figuring out method. English is the proving ground.

In scope: transcription accuracy. Out of scope (for now): speaker diarization as a benchmark
metric, clinical-note generation, streaming latency — though a throwaway **cpWER probe** has
answered the attribution-cost question on PriMock57 (Soniox: attribution ≈ free; see
[status](../status.md) and the spec `docs/superpowers/specs/2026-07-09-diarization-cpwer-probe.md`).
See [overview](overview.md) for scope detail.

## Questions we're answering

**Phase 1 (English):**

1. **Which models transcribe medical conversations most accurately?** Scored by [WER](../metrics/wer.md),
   using the Open ASR Leaderboard convention so numbers compare to published ones.
2. **Does word-level accuracy capture what matters for a scribe?** WER weights every word equally,
   but a dropped drug or diagnosis costs far more than a dropped "um". So we added a
   [medical-term recall](../metrics/medical-term-recall.md) metric — a research extension beyond the
   original accuracy-only benchmark — and asked whether it **reranks** the models versus WER.
3. **Is that rerank robust to how you define "a medical term"?** Or is it an artifact of one tool?
   Checked with four independent extractors (three clinical NER models + a medical LLM).

**Carry-over question for phase 2 (Korean):** do the phase-1 shortlist, harness, and metric
transfer — swapping WER for CER — to Korean audio?

Questions 2 and 3 are answered: see the [finding](../findings/medical-term-recall.md) (recall
reranks the winner, robustly). Question 1's table lives in `results/wer_summary.md`.

## How we answer them (design in brief)

- **Metrics.** WER as primary (jiwer corpus WER + a vendored Whisper English normalizer); medical-term
  recall as a *complement*, not a replacement. See [WER](../metrics/wer.md) and
  [medical-term recall](../metrics/medical-term-recall.md).
- **Datasets, chosen for distinct roles.** [PriMock57](../datasets/primock57.md) — 57 real
  primary-care consultations, the clinical signal; [MedDialog-Audio](../datasets/meddialog-audio.md)
  — synthetic, tests noise robustness; [LibriSpeech test-other](../datasets/librispeech-test-other.md)
  — a WER sanity check, excluded from the entity metric (no clinical content).
- **Models.** Eight in the lineup (open GPU models + cloud APIs), picked with a **Korean-ready bias**
  so phase 2 can reuse them; six benchmarked so far (Deepgram + AssemblyAI pending keys). See the
  [transcriber protocol](../models/transcriber-protocol.md).
- **Harness principles.** Runs headless on a remote GPU server; caches every transcript per file so
  runs resume and paid APIs are never double-billed; the test suite stays fully offline. See
  [architecture](architecture.md) and the [transcript cache](../components/transcript-cache.md).
- **Two-stage entity metric.** `entity-build` freezes the reference entities to a manifest (the
  heavy, method-specific step); `entity-score` aggregates recall offline. This is what lets four
  definitions of "a medical term" be compared cheaply.

The full design rationale — every decision, the models considered and excluded — lives in the
design spec, `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md` (the deep reference, kept
outside this bundle).

## Roadmap

**Done.** English WER round (6 models × 3 datasets, zero failures). Medical-term recall built four
ways, all agreeing. See [status](../status.md) and the [finding](../findings/medical-term-recall.md).

**Next (English round).**
- General-LLM foil via [openrouter](../entity-methods/openrouter.md) — does a *general* frontier
  model also rank Soniox #1, as the medical-specialized MedGemma did? Blocked on `OPENROUTER_API_KEY`.
- [OSCE / Fareez](../datasets/fareez-interviews.md) — local models transcribed and scored, and the
  rerank reproduces (see the [finding](../findings/medical-term-recall.md)); the metered APIs
  (Soniox, gpt-4o) on OSCE are the remaining step for a full cross-family comparison. The Soniox
  pass should run with diarization enabled: the cpWER probe passed its PriMock gate, and the
  `D:`/`P:` tags make the Fareez extension near-free (decision input for the scribe architecture,
  flat-transcript-in vs speaker-tagged-turns-in).
- Fuzzy entity matching — to recover spelling/abbreviation variants exact matching undercounts.
- Deepgram + AssemblyAI runs, once keys are available.

**Phase 2 (Korean).** The harness, caching, CLI, and the recall metric carry over; swap WER → CER,
add Korean datasets (AI Hub, KsponSpeech) and Korean-capable APIs. Reuse Whisper / Qwen3-ASR / Soniox.

Live progress and the current todo list live in [status](../status.md).
