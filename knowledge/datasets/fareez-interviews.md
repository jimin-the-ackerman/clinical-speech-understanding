---
type: Dataset
title: Fareez OSCE interviews
description: 272 verbatim patient-physician OSCE interviews (~51 h); loaded, transcription pending.
tags: [dataset, fareez, osce, long-form, pending]
timestamp: 2026-07-08
---

# Fareez OSCE interviews (`fareez-interviews`)

272 simulated patient–physician OSCE consultations (Fareez et al. 2022), ~51 h, one 16 kHz
mono MP3 each with a verbatim `D:`/`P:`-tagged transcript. Long-form clinical dialogue, ~5×
PriMock57's reference mass; Teams-recorded clean audio (complements PriMock57's room-mic).
CC0; Figshare DOI `10.6084/m9.figshare.c.5545842.v1` (paper `10.1038/s41597-022-01423-1`).

- **Prep** (`datasets/fareez.py`): download `Data.zip` (retry/backoff — figshare rate-limits),
  **MD5-verify**, extract via temp dir + atomic rename. `parse_transcript` strips the `D:`/`P:`
  tags to one reference; `condition = None` (pooled like PriMock57).
- **Status**: loader done, data downloaded and verified loadable (272 records; disfluencies
  present → verbatim, WER-trustworthy). **Not transcribed** — a paid checkpoint (~$30–50 API +
  GPU). See [status](../status.md).
