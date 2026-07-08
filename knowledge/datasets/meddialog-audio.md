---
type: Dataset
title: MedDialog-Audio
description: Synthetic TTS medical dialogues at multiple noise levels — vocabulary + noise robustness.
tags: [dataset, meddialog, synthetic, noise-robustness]
timestamp: 2026-07-08
---

# MedDialog-Audio

Synthetic corpus: MedDialog-EN dialogue text synthesized with Orpheus TTS, with white noise
and hospital ambient noise added at multiple SNR levels (HF `aline-gassenn/MedDialog-Audio`).
Measures medical vocabulary and **noise robustness** — it complements PriMock57, it doesn't
replace it (TTS is acoustically easier than real speech).

- **Subsample** (`datasets/meddialog_audio.py`): a fixed-seed (`SEED=42`) sample of
  `PER_CONDITION=300` per condition, **frozen in a committed manifest**
  (`results/manifests/meddialog_audio.json`) so runs are reproducible and API cost is bounded.
  `prepare()` re-downloads exactly the manifest's files and never rewrites it once it exists.
- **7 conditions** (the `condition` field): `clean`, `background_noise_20/40/60`,
  `white_noise_2/6/10` — encoded in a 3-char filename suffix → a per-model noise-robustness
  curve.
- **Finding**: under white noise, gpt-4o-transcribe collapses (WER 96–97%, medical-term recall
  ≈ 0 — literally zero clinical terms survive). See the [finding](../findings/medical-term-recall.md).
