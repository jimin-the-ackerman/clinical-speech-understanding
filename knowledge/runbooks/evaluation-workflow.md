---
type: Runbook
title: Evaluation workflow
description: Run scoring over cached transcripts and read the WER + medical-term-recall tables.
tags: [runbook, scoring, evaluation]
timestamp: 2026-07-13
---

# Evaluation workflow

Scoring is the third stage (`prepare → transcribe → score`), pure text math over the
[transcript cache](../components/transcript-cache.md) — no audio/GPU/keys, reproducible from a
bare checkout.

```bash
# WER
uv run stt-eval score                      # -> results/wer_summary.{csv,md}, wer_per_file.csv
uv run python scripts/score_cpwer.py       # cpWER vs flat WER from cached by_speaker (needs a --diarize run)

# medical-term recall (per method): build a manifest (in a --with overlay), then score it
uv run stt-eval entity-build --method bc5cdr --datasets primock57,meddialog-audio
uv run stt-eval entity-score --manifest results/entity_manifests/bc5cdr.json   # -> entity_recall_bc5cdr.{csv,md}
```

Reading the WER table (one row per model × dataset × condition): check **`n_failed ≈ 0`** first
(failed files are excluded, loudly), then `wer` (corpus WER, lower is better) and `rtf` (speed).
For *why* the numbers mean what they do — normalization vs substitution, pool-don't-average — see
[WER](../metrics/wer.md) and [text normalization](../metrics/text-normalization.md). For the
entity metric see [medical-term recall](../metrics/medical-term-recall.md).
