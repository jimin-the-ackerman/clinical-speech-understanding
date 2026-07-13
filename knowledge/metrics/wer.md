---
type: Metric
title: Word Error Rate (WER)
description: Corpus WER via jiwer after the Whisper normalizer, grouped by (model, dataset, condition).
tags: [metric, wer, jiwer, scoring]
timestamp: 2026-07-10
---

# Word Error Rate (WER)

The primary accuracy metric. `stt-eval score` (`score.py` + `metrics.py`) reads the
[transcript cache](../components/transcript-cache.md) — pure text math, no audio/GPU/keys, so
anyone can reproduce the table from a bare checkout.

## How it's computed

1. **Normalize** both reference and hypothesis with [`normalize_en`](text-normalization.md).
2. **Align + count** with jiwer: minimum-edit alignment labels each position match /
   **S**ubstitution / **D**eletion / **I**nsertion. `WER = (S + D + I) / (reference words)`.

   ```
   ref:  ... three times  daily
   hyp:  ... three times  a  day
                          S   I     (daily→a, insert day)  ≈ 2 errors / 10 words = 0.20
   ```
3. **Pool, don't average** — group files by **(model, dataset, condition)** and compute
   **corpus WER**: `sum(errors over group) / sum(reference words over group)`, so a 1,500-word
   consult isn't outweighed by a 3-word utterance. (`condition` is why MedDialog reports 7 rows
   per model and PriMock57/LibriSpeech one each.)
4. **Failures excluded, loudly** — a `failed: true` file is skipped, counted in `n_failed`, with
   a visible `[warn]`; a divergent-reference warning flags stale caches. Check `n_failed ≈ 0`.

## Output

`results/wer_summary.{csv,md}` + `results/wer_per_file.csv`. One row per (model, dataset,
condition): `n_scored, n_failed, n_empty_ref, wer` (corpus), `rtf` (compute-sec / audio-sec).

## WER's blind spot

Every reference word counts the same — `the` weighs as much as `amoxicillin`. For a clinical
scribe that's the wrong weighting, which is what [medical-term recall](medical-term-recall.md)
addresses. The Korean phase swaps WER → CER (a contained change in `score.py`).

## cpWER (speaker attribution)

`metrics.py` also implements **cpWER** (concatenated minimum-permutation WER): `cpwer_align`
pairs hypothesis speakers to reference speakers by whichever permutation minimizes WER over the
per-speaker concatenated text, and `cpwer` scores one file; pooling the aligned strings through
`corpus_wer` micro-averages it across files, matching the CHiME-6/MeetEval convention. The
permutation search is O(n!) over speakers — fine for 2-party clinical audio. With one speaker it
collapses to plain corpus WER. Speaker buckets come from `stt-eval transcribe --diarize`
(Soniox-only; cached as `by_speaker` beside the flat text) and are scored offline by
`scripts/score_cpwer.py` — **not** wired into `stt-eval score`. Origin: the retired PriMock57
probe (see the [spec][spec]; result in the
[attribution finding](../findings/speaker-attribution-cost.md)). Caveat: scoring normalizes with
`normalize_en` so cpWER is comparable to *our* flat WER, while published cpWER applies no
linguistic normalization.

[spec]: ../../docs/superpowers/specs/2026-07-09-diarization-cpwer-probe.md
