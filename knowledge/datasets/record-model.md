---
type: Concept
title: Record model & dataset registry
description: The frozen Record loaders yield, the lazy dataset registry, and the condition grouping field.
tags: [datasets, record, registry, condition]
timestamp: 2026-07-08
---

# Record model & dataset registry

Every dataset loader yields a list of **`Record`** (`records.py`, the package root — not under
`datasets/`; `@dataclass(frozen=True)`):

- `file_id: str` — stable id (also the [cache](../components/transcript-cache.md) key)
- `audio_path: Path` — 16 kHz mono audio on disk
- `reference: str` — the ground-truth transcript
- `condition: str | None` — an experimental bucket (e.g. noise level) that
  [scoring](../metrics/wer.md) groups by; `None` means "pooled" (PriMock57, LibriSpeech, Fareez)

**Registry** (`datasets/__init__.py`): `DATASETS` names → lazily imported modules. Each module
exposes `prepare(data_dir)` (network/preprocess — idempotent and resumable via an exists-check
and temp-dir-then-rename) and `load(data_dir) -> list[Record]` (offline). `prepare(name, ...)`
and `load(name, ...)` dispatch by name.

Datasets: [PriMock57](primock57.md), [MedDialog-Audio](meddialog-audio.md),
[LibriSpeech test-other](librispeech-test-other.md), [Fareez OSCE](fareez-interviews.md).
