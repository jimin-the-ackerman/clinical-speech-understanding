---
type: Runbook
title: GPU benchmark run
description: Step-by-step to run the full English STT benchmark on a CUDA machine, with sanity anchors.
tags: [runbook, gpu, benchmark, cuda]
timestamp: 2026-07-08
---

# GPU benchmark run

Run the full English round on a CUDA machine. Prereqs: NVIDIA GPU + driver, `ffmpeg`, `git-lfs`,
uv. See [setup](setup.md) and [environment](../project/environment.md) (the CUDA-12.6 torch pin).

## Steps

```bash
# 1. setup
git checkout stt-dev
uv sync --group dev --extra local --extra data
uv run pytest                                   # offline; verify env before big downloads
# keys via --env-file .env (OPENAI/SONIOX/DEEPGRAM/ASSEMBLYAI); a missing key just [skip]s

# 2. GPU sanity check (free, ~minutes)
uv run stt-eval prepare --datasets librispeech-test-other
uv run stt-eval transcribe --models whisper-large-v3-turbo --datasets librispeech-test-other --limit 40
uv run stt-eval score

# 3. data prep (once; idempotent/resumable)
uv run stt-eval prepare --datasets primock57,meddialog-audio

# 4. local models (free, a few hours; single-worker by design)
uv run stt-eval transcribe \
  --models whisper-large-v3,whisper-large-v3-turbo,qwen3-asr-0.6b,qwen3-asr-1.7b \
  --datasets librispeech-test-other,primock57,meddialog-audio

# 5. API models (a few dollars; cache-per-file means no re-billing on resume)
uv run --env-file .env stt-eval transcribe \
  --models gpt-4o-transcribe,soniox-stt-async-v5 \
  --datasets librispeech-test-other,primock57,meddialog-audio --workers 8

# 6. score + commit
uv run stt-eval score            # -> results/wer_summary.{csv,md}, results/wer_per_file.csv
git add results/ && git commit -m "results: English round WER benchmark"
```

## Sanity anchors — check before trusting the table

- `whisper-large-v3` on LibriSpeech test-other within a couple points of **~4–5% WER**
  (published). Far above = harness bug, not a bad model.
- `n_failed ≈ 0` in every summary row (the scorer warns otherwise).
- PriMock57 WER much higher than LibriSpeech for every model — expected (real room-mic audio is
  the hard condition).
- MedDialog WER worsens as noise gets harder (`background_noise` 20→60, `white_noise` 2→10).
- A reference-drift warning in `score` = stale caches from a changed reference build →
  re-transcribe or delete the named cache files.

## Known quirks

- `--results-dir` is a top-level flag: `stt-eval --results-dir X score`.
- `--limit N` takes the first N records alphabetically — on MedDialog that's all one noise
  condition; fine for smoke tests, not real numbers.
- Don't run two `transcribe` invocations against the same results dir concurrently.
