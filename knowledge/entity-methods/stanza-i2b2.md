---
type: Entity Method
title: stanza-i2b2 (Stanza NER)
description: Stanza i2b2 NER — problem/test/treatment; dense (~6 terms/file), most of the old union's mass.
tags: [entity-method, ner, stanza, i2b2]
timestamp: 2026-07-08
---

# stanza-i2b2 — Stanza i2b2 NER

Stanza's i2b2 clinical NER tags **problem / test / treatment**. It supersets bc5cdr's
disease+chemical and is the only method that catches procedures and tests (ECG, thyroid profile,
X-ray). Dense (~6 terms/file — most of the old Med7+Stanza union's mass). `_stanza_i2b2_extractor`
in `entity_score.py`; strips leading determiners ("your electrocardiogram" → "electrocardiogram").

- **Env**: pure Stanza (no spaCy); `stanza.download` fetches the i2b2 model at runtime.
- **Build**:
  ```
  uv run --with stanza stt-eval entity-build --method stanza-i2b2 --datasets primock57,meddialog-audio
  ```
- **History**: split out (with [med7](med7.md)) from a former `ner-union` method — med7 ∪
  stanza-i2b2 (deduped per file) reproduces the old union exactly. Ranks Soniox #1 — see the
  [finding](../findings/medical-term-recall.md).
