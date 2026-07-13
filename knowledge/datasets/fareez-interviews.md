---
type: Dataset
title: Fareez OSCE interviews
description: 272 verbatim patient-physician OSCE interviews (~51 h); all six models transcribed and scored, Soniox pass diarized.
tags: [dataset, fareez, osce, long-form]
timestamp: 2026-07-13
---

# Fareez OSCE interviews (`fareez-interviews`)

272 simulated patient–physician OSCE consultations (Fareez et al. 2022), ~51 h, one 16 kHz
mono MP3 each with a verbatim `D:`/`P:`-tagged transcript. Long-form clinical dialogue, ~5×
PriMock57's reference mass; Teams-recorded clean audio (complements PriMock57's room-mic).
CC0; Figshare DOI `10.6084/m9.figshare.c.5545842.v1` (paper `10.1038/s41597-022-01423-1`).

- **Prep** (`datasets/fareez.py`): download `Data.zip` (retry/backoff — figshare rate-limits),
  **MD5-verify**, extract via temp dir + atomic rename. `parse_transcript` strips the `D:`/`P:`
  tags to one reference; `speaker_reference` keeps the same turns split per speaker
  (`{"doctor": …, "patient": …}` — the cpWER oracle, mirroring PriMock57's);
  `condition = None` (pooled like PriMock57). Transcripts are decoded
  BOM-aware (`_read_transcript`): 2 of the 272 (`RES0002`, `RES0054`) ship as **UTF-16**, which the
  original UTF-8-only read turned into null-byte garbage — poisoning their WER and entity extraction
  until fixed (2026-07-09).
- **Status**: transcribed and scored by **all six models** — the four local models
  (2026-07-09), Soniox with diarization (2026-07-11, `by_speaker` in the cache), and gpt-4o via
  OpenRouter (2026-07-13; only its *diarize* variant remains open). The medical-term-recall
  rerank [reproduces cross-family](../findings/medical-term-recall.md): Soniox #1 on every
  method, and attribution costs +0.45 pt here (see the
  [attribution finding](../findings/speaker-attribution-cost.md)). See [status](../status.md).
