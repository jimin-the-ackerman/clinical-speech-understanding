# Tutorial: how scoring works

How `stt-eval score` turns cached transcripts into a WER table. Read this if
you want to understand what the numbers mean, or before you add a new metric.

Code referenced: [`score.py`](../../src/stt_eval/score.py),
[`normalize.py`](../../src/stt_eval/normalize.py),
[`metrics.py`](../../src/stt_eval/metrics.py).

## First, a common misconception

The normalizer does **not** swap synonyms. It never turns "hi" into "hello".
That would change the *meaning*, and no scoring step is allowed to do that.

Keep two ideas separate:

- **Normalization** canonicalizes *surface form only* — the same words written
  differently: case, punctuation, contractions (`I'll` ↔ `I will`), number and
  currency formatting (`500mg` ↔ `500 mg`), filler markers, whitespace. The
  reference and the hypothesis go through the **identical** transformation, so a
  pure spelling/formatting difference cancels out and is not counted as an error.
- **Substitution** is one word genuinely replaced by a *different* word
  (`hi` vs `hello`, `fifty` vs `fifteen`). It survives normalization and **is**
  counted as an error.

Rule of thumb: normalization only ever merges strings that represent the *same
spoken words*. It will never merge `hi` and `hello`.

## The big picture

Evaluation is the third stage of the pipeline: `prepare` → `transcribe` →
**`score`**. By the time you score, all audio and model work is already done and
frozen in the transcript cache. Scoring is pure text math over JSON files — no
audio, no GPU, no network, no API keys. That is why anyone can re-run `score`
from a bare checkout and get the same table.

## Step 0 — what a cached transcript holds

Each `results/transcripts/{dataset}/{model}/{file_id}.json` stores the two
strings scoring needs, both **raw** (original casing and punctuation):

- `reference` — the ground-truth transcript, embedded from the dataset at
  transcribe time.
- `text` — what the model produced.

Storing the reference *raw* (not pre-normalized) is deliberate: a new metric can
re-derive whatever form it needs. WER wants normalized text; clinical NER wants
the raw cased text (see the last section).

## Step 1 — normalize both sides

`score()` runs `normalize_en()` (the vendored Whisper `EnglishTextNormalizer`)
on **both** the reference and the hypothesis.

```
reference (raw):  "I'll prescribe 500mg of Amoxicillin, three times daily."
hypothesis (raw): "i will prescribe 500 mg of amoxicillin three times a day"

after normalize_en:
ref:  i will prescribe 500 mg of amoxicillin three times daily
hyp:  i will prescribe 500 mg of amoxicillin three times a day
```

Look at what normalization **erased**: the apostrophe in `I'll`, the casing of
`Amoxicillin`, the comma, `500mg` vs `500 mg`. None of those are recognition
errors, and after normalizing both sides agree on those tokens. What it **kept**
is the one real difference: the model said `a day` where the reference says
`daily`. That difference should count — and it will.

This is the Open ASR Leaderboard convention. Using it is what makes our WER
numbers comparable to published ones.

## Step 2 — align and count (this is WER)

WER hands the two normalized token streams to jiwer, which finds the
minimum-edit alignment and labels each position: match, **S**ubstitution,
**D**eletion, or **I**nsertion.

```
ref:  i will prescribe 500 mg of amoxicillin three times  daily
hyp:  i will prescribe 500 mg of amoxicillin three times  a  day
                                                           S   I
                                                    (daily->a, insert day)
```

```
WER = (S + D + I) / (number of reference words)
```

In the example, ~2 errors over 10 reference words ≈ 0.20.

So WER is literally "what fraction of the reference did the model get wrong,"
and **every reference word counts the same** — `the` weighs as much as
`amoxicillin`. That equal weighting is WER's known blind spot for clinical use
(see [the medical-entity research note](../research/2026-07-06-medical-entity-asr-metrics.md)).

## Step 3 — pool, don't average

We do **not** compute one WER per file and average the per-file numbers. That
would let a 3-word utterance move the score as much as a 1,500-word
consultation. Instead `score()` groups files by **(model, dataset, condition)**
and computes **corpus WER**: sum all errors across the group, divide by all
reference words across the group. Long files contribute in proportion to their
length.

```
corpus WER = sum(errors over all files in group) / sum(reference words in group)
```

The `condition` key is why MedDialog reports 7 rows per model (clean plus six
noise levels) while LibriSpeech and PriMock57, which have no condition, report
one row each.

`score()` also emits a per-file WER (`file_wer`) for drill-down, but the
headline table is always corpus WER.

## Step 4 — failures are excluded, loudly

A file marked `failed: true` (e.g. an API error) is skipped, counted in
`n_failed`, and the scorer prints a visible `[warn]`. This guarantees a model's
WER is never quietly computed over a different subset than its peers. Before
trusting any row, check `n_failed ≈ 0`.

## Reading the output

One row per (model, dataset, condition):

| field | meaning |
|---|---|
| `n_scored` | files that contributed to the WER |
| `n_failed` | files excluded because transcription failed |
| `n_empty_ref` | files excluded because the reference had no scorable words |
| `wer` | corpus WER for the group (lower is better) |
| `rtf` | real-time factor = compute-seconds / audio-seconds (lower is faster) |

Written to `results/wer_summary.{csv,md}` and `results/wer_per_file.csv`.

## Where a new metric plugs in

The medical-entity metric maps onto this same machinery as a parallel branch at
Steps 1-2:

1. Run clinical NER on the **raw** `reference` (before normalization) — NER
   models need the casing and punctuation that Step 1 destroys, which is exactly
   why the reference is cached raw.
2. Normalize each extracted entity's surface form the same way, then test whether
   it survived into the normalized hypothesis.
3. Aggregate at the same (model, dataset, condition) grain and report it *next
   to* WER.

Same cache, same normalizer, same grouping — it just measures "did the
*clinically important* words survive" instead of "did *all* words survive." See
the [research note](../research/2026-07-06-medical-entity-asr-metrics.md) for the
design details and the normalizer gotcha.
