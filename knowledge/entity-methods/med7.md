---
type: Entity Method
title: med7 (spaCy NER)
description: Med7 en_core_med7_lg — drug/dosage/route/frequency NER; sparse (~0.9 terms/file).
tags: [entity-method, ner, med7, drugs]
timestamp: 2026-07-08
---

# med7 — Med7 spaCy NER

Med7 (`en_core_med7_lg`) tags medication concepts: **drug / dosage / route / frequency /
strength / form**. Sparse on these consults (~0.9 terms/file — drugs are rare), so its
lower-rank ordering rests on a small denominator. `_med7_extractor` in `entity_score.py`.

- **Env**: needs spacy≥3.8 (incompatible with [bc5cdr](bc5cdr.md)'s spacy<3.8); no Stanza; its
  own overlay.
- **Build**:
  ```
  uv run --with "en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-1.1.0-py3-none-any.whl" \
    stt-eval entity-build --method med7 --datasets primock57,meddialog-audio
  ```
  The wheel must be the versioned `-1.1.0-py3-none-any.whl` (the `-any-` URL is not PEP 440).
- **Result**: still ranks Soniox #1 despite sparsity — see the [finding](../findings/medical-term-recall.md).
