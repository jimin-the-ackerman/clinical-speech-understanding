---
type: Dataset
title: PriMock57
description: 57 mock primary-care consultations with real conversational acoustics — the clinical signal.
tags: [dataset, primock57, clinical, conversational]
timestamp: 2026-07-08
---

# PriMock57

57 mock primary-care consultations (Babylon Health), ~10 min each. Doctor and patient are
recorded on separate channels with TextGrid transcripts. **The primary clinical signal** in
this benchmark — real spontaneous speech, the honest test for a scribe.

- **Prep** (`datasets/primock57.py`): `git clone --depth=1` + `git lfs pull`, then per
  consultation `ffmpeg amix` the two channels to one 16 kHz mono wav (the room-mic condition).
- **Reference**: time-ordered merge of both speakers' TextGrid utterances (`merge_reference`
  by `minTime`); `_clean` strips `<...>` non-speech markers. `condition = None`.
- **Caveat**: overlapping speech makes word order slightly ambiguous → adds WER noise, but is
  the realistic condition.

Ranked #1 by [medical-term recall](../metrics/medical-term-recall.md): Soniox — see the
[finding](../findings/medical-term-recall.md).
