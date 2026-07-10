---
type: Research Finding
title: Medical-term recall for clinical ASR — how you define "a medical term" barely moves the ranking
description: On real clinical consultations, four independent ways of scoring medical-term fidelity all rank the same STT model first — a result plain WER misranks.
resource: https://github.com/jimin-the-ackerman/clinical-speech-understanding/blob/stt-dev/knowledge/findings/medical-term-recall.md
tags: [asr, stt, medical-nlp, evaluation, primock57, entity-recall]
timestamp: 2026-07-08
---

# Medical-term recall for clinical ASR

**TL;DR** — We benchmark speech-to-text (STT) models for a clinical "AI scribe". The standard
metric, Word Error Rate (WER), weights every word equally — but for a scribe, a dropped drug or
diagnosis matters far more than a dropped "um". We added a **medical-term recall** metric and
found it reranks the models on real consultations: **Soniox is #1, not the WER winner.** Crucially,
that rerank is **robust** — four independent ways of defining "a medical term" (three clinical NER
models + a medical LLM) all put Soniox first with the same top-3.

## The question

Phase 1 (English) of an STT benchmark for a Korean healthcare scribe. WER counts every word edit
equally, so a model can win on WER while mangling exactly the words a clinician cares about. Two
questions: (1) does a **medical-term recall** metric rerank models vs WER? (2) is that ranking
**robust to how you define "a medical term"**, or an artifact of one tool?

## Setup

Metric = the fraction of the reference transcript's clinical entities whose surface form survives
into the model's hypothesis, per (model, dataset). It complements WER; it does not replace it.
Datasets: **PriMock57** (57 real primary-care consultations — the clinical signal),
MedDialog-Audio (synthetic, noise robustness), LibriSpeech (WER sanity only, excluded from the
entity metric). Six STT systems: whisper-large-v3, -v3-turbo, qwen3-asr-0.6b, -1.7b,
gpt-4o-transcribe, soniox-stt-async-v5. See [medical-term recall](../metrics/medical-term-recall.md)
for how the two-stage pipeline works.

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

The four methods: [bc5cdr](../entity-methods/bc5cdr.md) (disease+chemical NER),
[med7](../entity-methods/med7.md) (drug/dosage NER),
[stanza-i2b2](../entity-methods/stanza-i2b2.md) (problem/test/treatment NER),
[medgemma](../entity-methods/medgemma.md) (a medical LLM, open-ended).

## Three findings

**(a) Recall reranks the winner.** WER ranks `qwen3-asr-1.7b` #1 and Soniox #2. Medical-term
recall **flips them — Soniox is #1** on every method. And `whisper-large-v3` **rises from 5th on
WER to 3rd on recall**: its errors land on function words, not clinical content.

**(b) The rerank is robust to the definition.** **All four methods rank Soniox #1, with an
identical top-3** (soniox > qwen-1.7b > whisper-v3). We split a combined NER method into its two
schemas to check — the drug-focused (med7) and problem/test/treatment-focused (stanza-i2b2)
extractors *independently* agree, and neither drove the earlier combined result (i2b2 ~13,100
spans, med7 only ~1,973). A medical LLM concurs. So the ranking is a property of the transcripts,
not of one entity tool.

**(c) It restates gpt-4o's noise collapse in clinical terms.** Under white noise, gpt-4o
hallucinates fluent nonsense: on MedDialog its WER climbs 41% → 96% → 97% across noise levels
2/6/10, and its medical-term recall falls **0.59 → 0.001 → 0.000** — literally zero clinical terms
survive. For a scribe, "no medical terms survived" is sharper than "97% WER".

## Replication on OSCE / Fareez (local models)

We re-ran the four local models on a second, very different clinical corpus:
[OSCE / Fareez](../datasets/fareez-interviews.md) — 272 simulated patient–physician consultations
(~51 h), clean Teams-recorded long-form dialogue, ~5× PriMock57's reference mass. The cloud APIs
(Soniox, gpt-4o) sat out this round, so this is a **local-models-only** check, not a cross-family
ranking — the "Soniox #1" headline above is a PriMock57 result and is not retested here.

| model | WER ↓ | bc5cdr | med7 | stanza-i2b2 | medgemma |
|---|---|---|---|---|---|
| **qwen3-asr-1.7b** | .112 | **.957** | **.867** | **.957** | **.958** |
| whisper-large-v3-turbo | .098 | .953 | .851 | .950 | .956 |
| whisper-large-v3 | .143 | .952 | .862 | .947 | .952 |
| qwen3-asr-0.6b | **.097** | .946 | .844 | .939 | .942 |

The finding **reproduces**: `qwen3-asr-1.7b` is #1 on every one of the four methods, and the WER
winner is again the recall loser — `qwen3-asr-0.6b` has the best WER (.097) but ranks **last on all
four** recall metrics. The same rerank appearing on a corpus with different speakers, acoustics, and
length is evidence the effect is a property of clinical transcription, not of PriMock57. (Data
notes: med7 leaves 24/272 OSCE refs empty — legitimate, since med7 is sparse; the other three
methods cover all 272. Two references shipped as UTF-16 and were mis-decoded until the loader
was fixed — see [Fareez](../datasets/fareez-interviews.md).)

## Caveats (honest limits)

- **It largely tracks WER** (Pearson ≈ −0.97 across all groups). It is a *complement*, earning
  its keep on the PriMock57 rerank and the clinical readability of the noise rows — not an
  independent signal.
- **Exact-match undercounts.** Matching is exact contiguous tokens, so a model is penalized when
  it transcribes a *valid* variant the reference spells differently — e.g. reference "flem" vs
  "phlegm", or "a and e" vs "A&E". This hits all models roughly equally (so it doesn't bias the
  ranking) but makes the absolute recall a slight underestimate. Fuzzy matching is a todo.

## Open todos / next steps

- **General-LLM foil ([openrouter](../entity-methods/openrouter.md)).** MedGemma is medically
  *specialized*. Does a *general* frontier model also rank Soniox #1? One extractor away — blocked
  only on an `OPENROUTER_API_KEY`.
- **[OSCE / Fareez](../datasets/fareez-interviews.md) dataset.** Local models transcribed and
  scored (see replication above); the rerank reproduces. Still pending: the metered APIs (Soniox,
  gpt-4o) on OSCE, to make it a full cross-family comparison rather than local-only.
- **Fuzzy entity matching.** Would recover the spelling/abbreviation misses above.
- **Phase 2 (Korean).** The harness, caching, and this metric carry over (swapping WER → CER).

## Pointers

- [Medical-term recall metric](../metrics/medical-term-recall.md) — mechanism, build/score.
- Entity methods: [bc5cdr](../entity-methods/bc5cdr.md) · [med7](../entity-methods/med7.md) ·
  [stanza-i2b2](../entity-methods/stanza-i2b2.md) · [medgemma](../entity-methods/medgemma.md) ·
  [openrouter](../entity-methods/openrouter.md).
- `docs/research/2026-07-06-medical-entity-asr-metrics.md` — literature survey and metric design.
- Raw numbers: `results/entity_recall_{bc5cdr,med7,stanza-i2b2,medgemma}.{csv,md}` and
  `results/wer_summary.md`.
