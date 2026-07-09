---
type: Dataset
title: Fareez OSCE interviews
description: 272 verbatim patient-physician OSCE interviews (~51 h); local models transcribed, metered APIs pending.
tags: [dataset, fareez, osce, long-form, local-transcribed]
timestamp: 2026-07-09
---

# Fareez OSCE interviews (`fareez-interviews`)

272 simulated patient–physician OSCE consultations (Fareez et al. 2022), ~51 h, one 16 kHz
mono MP3 each with a verbatim `D:`/`P:`-tagged transcript. Long-form clinical dialogue, ~5×
PriMock57's reference mass; Teams-recorded clean audio (complements PriMock57's room-mic).
CC0; Figshare DOI `10.6084/m9.figshare.c.5545842.v1` (paper `10.1038/s41597-022-01423-1`).

- **Prep** (`datasets/fareez.py`): download `Data.zip` (retry/backoff — figshare rate-limits),
  **MD5-verify**, extract via temp dir + atomic rename. `parse_transcript` strips the `D:`/`P:`
  tags to one reference; `condition = None` (pooled like PriMock57). Transcripts are decoded
  BOM-aware (`_read_transcript`): 2 of the 272 (`RES0002`, `RES0054`) ship as **UTF-16**, which the
  original UTF-8-only read turned into null-byte garbage — poisoning their WER and entity extraction
  until fixed (2026-07-09).
- **Status**: transcribed by the four **local** models (whisper-large-v3 / -turbo,
  qwen3-asr-0.6b / -1.7b) and scored — WER + all four entity manifests now include OSCE, and the
  medical-term-recall rerank [reproduces](../findings/medical-term-recall.md) (qwen3-asr-1.7b #1 on
  every method). The metered APIs (Soniox, gpt-4o) are still pending for a full cross-family
  comparison. See [status](../status.md).
