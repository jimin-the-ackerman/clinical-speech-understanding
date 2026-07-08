---
type: Concept
title: Text normalization
description: The Whisper English normalizer canonicalizes surface form only — it never swaps synonyms.
tags: [metric, normalization, wer]
timestamp: 2026-07-08
---

# Text normalization

`normalize_en()` (`normalize.py`) wraps the vendored Whisper `EnglishTextNormalizer` (in
`_whisper_norm/`, MIT) — the **Open ASR Leaderboard convention**, which is what makes our WER
numbers comparable to published ones. Applied identically to reference and hypothesis before
[WER](wer.md) and [entity matching](medical-term-recall.md).

## It canonicalizes surface form only — it never swaps synonyms

Two ideas to keep separate:

- **Normalization** canonicalizes the *same words written differently*: case, punctuation,
  contractions (`I'll` ↔ `I will`), number/currency formatting (`500mg` ↔ `500 mg`), filler
  markers, whitespace. Both sides get the identical transform, so a pure spelling/formatting
  difference cancels and is not counted as an error.
- **Substitution** is one word genuinely replaced by a *different* word (`hi` vs `hello`,
  `fifty` vs `fifteen`). It survives normalization and **is** counted.

It will never merge `hi` and `hello` — that would change meaning.

## Worked example

```
reference (raw):  "I'll prescribe 500mg of Amoxicillin, three times daily."
hypothesis (raw): "i will prescribe 500 mg of amoxicillin three times a day"

after normalize_en:
ref:  i will prescribe 500 mg of amoxicillin three times daily
hyp:  i will prescribe 500 mg of amoxicillin three times a day
```

Erased: the apostrophe in `I'll`, the casing of `Amoxicillin`, the comma, `500mg` vs `500 mg`
— none are recognition errors. Kept: the one real difference, `daily` vs `a day`, which counts.

Transcripts are cached **raw** (see [transcript cache](../components/transcript-cache.md)) so
each metric derives the form it needs — WER wants normalized text; clinical NER wants the raw
cased text (see [medical-term recall](medical-term-recall.md)).
