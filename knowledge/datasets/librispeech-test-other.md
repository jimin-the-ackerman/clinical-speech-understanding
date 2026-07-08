---
type: Dataset
title: LibriSpeech test-other
description: Standard read-speech benchmark used only as a WER sanity check; excluded from the entity metric.
tags: [dataset, librispeech, sanity-check]
timestamp: 2026-07-08
---

# LibriSpeech test-other

Standard per-utterance read-speech benchmark (audiobooks, OpenSLR). **Sole purpose:** validate
the harness by comparing our WER against published Open-ASR-Leaderboard numbers (~4–5%).

- **Prep** (`datasets/librispeech.py`, no HF dep): streamed download, extract to a temp dir
  then atomic rename; audio kept as original 16 kHz mono FLAC. `parse_trans_file` splits a
  `.trans.txt` into `(file_id, reference)`. `condition = None`.
- **Excluded from [medical-term recall](../metrics/medical-term-recall.md)**: read-speech
  literature has no clinical content, and its all-caps references break clinical NER — the
  entity manifests are built with `--datasets primock57,meddialog-audio`.
