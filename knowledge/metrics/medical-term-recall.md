---
type: Metric
title: Medical-term recall
description: Fraction of reference clinical entities whose surface form survives into the hypothesis.
tags: [metric, entity-recall, clinical, nlp]
timestamp: 2026-07-08
---

# Medical-term recall

A complement to [WER](wer.md) that weights only the clinically important words: the fraction of
a reference's clinical entities whose surface form survives into the hypothesis, reported per
(model, dataset, condition). `entity_score.py`. Background + literature survey:
`docs/research/2026-07-06-medical-entity-asr-metrics.md`.

## Two-stage design

1. **build** — `entity-build --method X` runs an entity extractor over each unique **raw**
   reference (clinical NER needs the casing/punctuation the normalizer strips) and freezes the
   result to `results/entity_manifests/X.json`. The method-specific cost/nondeterminism lives
   here; LLM methods get a resumable per-reference cache.
2. **score** — `entity-score --manifest P` aggregates recall offline (deterministic, no
   NER/keys) to `results/entity_recall_<X>.{csv,md}`. Comparing methods = scoring different
   manifests.

## Matching

`entity_hit` is a **token-level contiguous-run** match on normalized tokens (so "art" doesn't
match inside "heart"); `file_recall` = (hits, total). Per-occurrence for the NER methods,
deduped per file for the LLM. Matching is exact — a known limitation (it undercounts valid
spelling/abbreviation variants like "flem"→phlegm, "a and e"→A&E); fuzzy matching is future work.

## Methods (five definitions of "a medical term")

NER: [bc5cdr](../entity-methods/bc5cdr.md) · [med7](../entity-methods/med7.md) ·
[stanza-i2b2](../entity-methods/stanza-i2b2.md). LLM: [medgemma](../entity-methods/medgemma.md) ·
[openrouter](../entity-methods/openrouter.md). The headline result — all built methods agree
Soniox #1 on PriMock57 — is the [finding](../findings/medical-term-recall.md).
