---
type: Entity Method
title: bc5cdr (scispaCy NER)
description: scispaCy en_ner_bc5cdr_md — disease + chemical NER; the narrow baseline.
tags: [entity-method, ner, scispacy, bc5cdr]
timestamp: 2026-07-08
---

# bc5cdr — scispaCy NER

A supervised NER model (scispaCy `en_ner_bc5cdr_md`, trained on the BC5CDR biomedical corpus)
that tags exactly two types: **disease** and **chemical** — the narrow baseline for
[medical-term recall](../metrics/medical-term-recall.md). `_scispacy_extractor` in
`entity_score.py` (excludes parser/lemmatizer).

- **Env**: pins spacy<3.8 — cannot share an env with [med7](med7.md) (spacy≥3.8); runs in its
  own overlay. See [environment](../project/environment.md).
- **Build**:
  ```
  uv run --with scispacy \
    --with "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz" \
    stt-eval entity-build --method bc5cdr --datasets primock57,meddialog-audio
  ```
- **Result**: ranks Soniox #1 on PriMock57 — see the [finding](../findings/medical-term-recall.md).
