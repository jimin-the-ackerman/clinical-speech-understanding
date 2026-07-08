---
type: Research Finding
title: Medical-term recall for clinical ASR — how you define "a medical term" barely moves the ranking
description: On real clinical consultations, four independent ways of scoring medical-term fidelity all rank the same STT model first — a result plain WER misranks.
resource: https://github.com/jimin-the-ackerman/clinical-speech-understanding/blob/stt-dev/docs/reports/medical-term-recall.md
tags: [asr, stt, medical-nlp, evaluation, primock57, entity-recall]
timestamp: 2026-07-08
---

# Medical-term recall for clinical ASR

**TL;DR** — We're benchmarking speech-to-text (STT) models for a clinical "AI scribe". The
standard metric, Word Error Rate (WER), weights every word equally — but for a scribe, a
dropped drug or diagnosis matters far more than a dropped "um". We added a **medical-term
recall** metric and found it reranks the models on real consultations: **Soniox is #1, not the
WER winner.** Crucially, that rerank is **robust** — four independent ways of defining "a medical
term" (three clinical NER models + a medical LLM) all put Soniox first with the same top-3. So
the finding isn't an artifact of one entity extractor.

## The question

This is phase 1 (English) of an STT benchmark for a Korean healthcare scribe. WER counts every
word edit equally, so a model can win on WER while mangling exactly the words a clinician cares
about. Two questions:

1. Does a **medical-term recall** metric rerank models versus WER?
2. Is that ranking **robust to how you define "a medical term"**, or an artifact of one tool?

## Setup

**Metric.** Medical-term recall = the fraction of the reference transcript's clinical entities
whose surface form survives into the model's hypothesis, reported per (model, dataset). It
complements WER; it does not replace it.

**Datasets.**
- **PriMock57** — 57 real primary-care consultations (mock but naturally spoken). This is the
  clinical signal.
- **MedDialog-Audio** — synthetic (TTS) dialogues across white/background-noise levels; used for
  noise robustness.
- **LibriSpeech** — read-speech audiobooks; used only as a WER sanity check and **excluded from
  the entity metric** (no clinical content).

**Models (6 STT systems).** whisper-large-v3, whisper-large-v3-turbo, qwen3-asr-0.6b,
qwen3-asr-1.7b, gpt-4o-transcribe, soniox-stt-async-v5.

The metric is a two-stage pipeline (freeze reference entities to a manifest, then score offline);
see the [working doc](../entity-metric-comparison.md) for how to reproduce it.

## Result

Medical-term recall on PriMock57 (higher is better), next to WER (lower is better). The four
recall columns are four different definitions of "a medical term":

| model | WER ↓ | bc5cdr | med7 | stanza-i2b2 | medgemma |
|---|---|---|---|---|---|
| **soniox** | .123 | **.936** | **.916** | **.940** | **.947** |
| qwen3-asr-1.7b | **.119** | .927 | .874 | .927 | .938 |
| whisper-large-v3 | .168 | .923 | .862 | .918 | .930 |
| whisper-large-v3-turbo | .127 | .916 | .824 | .912 | .925 |
| qwen3-asr-0.6b | .125 | .907 | .828 | .913 | .921 |
| gpt-4o-transcribe | .217 | .892 | .839 | .876 | .879 |

The four methods:

- **bc5cdr** — scispaCy NER, diseases + chemicals (narrow).
- **med7** — spaCy NER for drugs / dosage / route / frequency (sparse: ~1 term/file).
- **stanza-i2b2** — Stanza NER for problem / test / treatment (dense: ~6 terms/file).
- **medgemma** — MedGemma-27B, a medical LLM prompted to extract clinical terms open-endedly.

## Three findings

**(a) Recall reranks the winner.** WER ranks `qwen3-asr-1.7b` #1 and Soniox #2. Medical-term
recall **flips them — Soniox is #1** on every method. And `whisper-large-v3` **rises from 5th on
WER to 3rd on recall**: its errors land on function words, not clinical content — something plain
WER cannot show.

**(b) The rerank is robust to the definition.** **All four methods rank Soniox #1, with an
identical top-3** (soniox > qwen-1.7b > whisper-v3). We even split a combined NER method into its
two schemas to check — the drug-focused (`med7`) and problem/test/treatment-focused
(`stanza-i2b2`) extractors *independently* agree, and neither drove the earlier combined result
(i2b2 contributes ~13,100 spans, med7 only ~1,973). A medical LLM concurs. So the ranking is a
property of the transcripts, not of one entity tool.

**(c) It restates gpt-4o's noise collapse in clinical terms.** Under white noise, gpt-4o
hallucinates fluent nonsense. On MedDialog its WER climbs 41% → 96% → 97% across noise levels
2/6/10, and its medical-term recall falls **0.59 → 0.001 → 0.000** — literally zero clinical
terms survive at the higher levels. For a scribe, "no medical terms survived" is a sharper,
more actionable statement than "97% WER".

## Caveats (honest limits)

- **It largely tracks WER** (Pearson ≈ −0.97 across all groups). It is a *complement*, earning
  its keep on the PriMock57 rerank and the clinical readability of the noise rows — not an
  independent signal.
- **Exact-match undercounts.** Matching is exact contiguous tokens, so a model is penalized when
  it transcribes a *valid* variant the reference happens to spell differently — e.g. reference
  "flem" vs the model's correct "phlegm", or "a and e" vs "A&E". This hits all models roughly
  equally (so it doesn't bias the ranking) but makes the absolute recall a slight underestimate.
  Fuzzy matching is on the todo list below.

## Open todos / next steps

- **General-LLM foil (OpenRouter).** MedGemma is medically *specialized*. Does a *general*
  frontier model also rank Soniox #1? One extractor away — blocked only on an `OPENROUTER_API_KEY`.
- **OSCE / Fareez dataset.** ~51 h of patient–physician interviews are loaded and verified; a
  paid transcription checkpoint (~$30–50 API + GPU time) before it joins the comparison.
- **Fuzzy entity matching.** Would recover the spelling/abbreviation misses noted above.
- **Phase 2 (Korean).** The harness, caching, and this metric carry over (swapping WER → CER).

## Pointers

- [`docs/entity-metric-comparison.md`](../entity-metric-comparison.md) — full method table,
  reproduce commands, and implementation notes.
- [`docs/research/2026-07-06-medical-entity-asr-metrics.md`](../research/2026-07-06-medical-entity-asr-metrics.md)
  — literature survey and metric design.
- Raw numbers: `results/entity_recall_{bc5cdr,med7,stanza-i2b2,medgemma}.{csv,md}` and
  `results/wer_summary.md`.
