# GPU Machine Runbook — English Benchmark Round

Step-by-step for running the full English STT benchmark on a CUDA machine.
Design: `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md` · Usage details: `README.md`

## uv in 30 seconds

`uv run <cmd>` = create/sync `.venv` from `uv.lock` (exact pinned versions), then run
`<cmd>` inside it. No venv activation, ever. `uv sync --extra local --extra data`
installs the optional heavy deps (torch/transformers/faster-whisper, HF datasets).
`stt-eval` is a console script defined in `pyproject.toml` (`stt_eval.run:main`) —
the packaged equivalent of `python -m`.

## 0. Prerequisites

- NVIDIA GPU + driver (faster-whisper uses CUDA via CTranslate2; Qwen3-ASR via torch)
- `ffmpeg` and `git-lfs` on PATH (PriMock57 prep needs both)
- uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 1. Setup

```bash
git clone <repo-url> && cd clinical-speech-understanding
git checkout stt-dev
uv sync --group dev --extra local --extra data
uv run pytest          # 59 offline tests; verifies the env before any big downloads
export OPENAI_API_KEY=...
export DEEPGRAM_API_KEY=...
export ASSEMBLYAI_API_KEY=...
export SONIOX_API_KEY=...
```

A missing key skips that model with a `[skip]` warning — it never aborts the run.

## 2. GPU sanity check (free, ~minutes)

```bash
uv run stt-eval prepare --datasets librispeech-test-other       # ~330 MB
uv run stt-eval transcribe --models whisper-large-v3-turbo \
  --datasets librispeech-test-other --limit 40
uv run stt-eval score
```

Confirms CUDA + faster-whisper work. The first 20 files are already cached from the
Mac smoke run (committed under `results/transcripts/`) and will be skipped — that's
the resume behavior working, not a bug.

## 3. Data prep (once; idempotent and resumable)

```bash
uv run stt-eval prepare --datasets primock57,meddialog-audio
```

- PriMock57: git-lfs clone + ffmpeg-mixes 57 consultations to mono 16 kHz.
- MedDialog: downloads only the ~2,100 wavs listed in the committed manifest
  (`results/manifests/meddialog_audio.json`) — same subsample on every machine.
- Interrupted prep is safe to just re-run.

## 4. Local models (free, a few hours)

```bash
uv run stt-eval transcribe \
  --models whisper-large-v3,whisper-large-v3-turbo,qwen3-asr-0.6b,qwen3-asr-1.7b \
  --datasets librispeech-test-other,primock57,meddialog-audio
```

Local models run single-worker by design (one GPU model instance). First use of each
model downloads weights into `~/.cache/huggingface`.

## 5. API models (a few dollars)

```bash
uv run stt-eval transcribe \
  --models gpt-4o-transcribe,deepgram-nova-3-medical,assemblyai-universal-3-5-pro,soniox-stt-async-v5 \
  --datasets librispeech-test-other,primock57,meddialog-audio --workers 8
```

Every transcript is cached per file before the next request — interrupting and
re-running never re-bills completed files.

## 6. Score and commit

```bash
uv run stt-eval score    # -> results/wer_summary.{csv,md}, results/wer_per_file.csv
git add results/ && git commit -m "results: English round WER benchmark" && git push
```

## Sanity anchors — check before trusting the table

- `whisper-large-v3` on LibriSpeech test-other: within a couple points of ~4–5% WER
  (published number). Far above that = harness bug, not a bad model.
- `n_failed` ≈ 0 in every summary row (the scorer warns visibly otherwise).
- PriMock57 WER much higher than LibriSpeech for every model — expected; real
  conversational room-mic audio is the hard condition we care about.
- MedDialog WER should worsen as the noise condition gets harder (b20→b60, w02→w10).
- Reference-drift warning in `score` output means stale caches from a changed
  reference build — re-transcribe or delete the named cache files.

## Known quirks

- `--results-dir` is a top-level flag: `stt-eval --results-dir X score`.
- `--limit N` takes the first N records alphabetically — on MedDialog that's all one
  noise condition; fine for smoke tests, not for real numbers.
- Don't run two `transcribe` invocations against the same results dir concurrently.
