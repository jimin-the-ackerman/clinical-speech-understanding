# clinical-speech-understanding

Benchmarking STT models for a Korean healthcare "AI scribe". Phase 1: English
medical audio. Design: `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`.

## Setup

Requires Python ≥3.10, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and `git` on PATH.

```bash
uv sync                      # core (scoring, API transcribers)
uv sync --extra data         # + HF datasets (MedDialog-Audio prepare)
uv sync --extra local        # + faster-whisper, transformers, torch (GPU models)
```

API keys (a missing key skips that model with a warning):
`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY`, `SONIOX_API_KEY`.

## Usage

```bash
uv run stt-eval prepare --datasets librispeech-test-other,primock57,meddialog-audio
uv run stt-eval transcribe \
  --models whisper-large-v3-turbo,gpt-4o-transcribe \
  --datasets primock57 --workers 8
uv run stt-eval score       # -> results/wer_summary.{csv,md}, results/wer_per_file.csv
```

`--results-dir` is a top-level flag (default `results/`), e.g.
`uv run stt-eval --results-dir X score`; normal usage needs nothing.

Models: `whisper-large-v3`, `whisper-large-v3-turbo`, `qwen3-asr-0.6b`,
`qwen3-asr-1.7b`, `gpt-4o-transcribe`, `deepgram-nova-3-medical`,
`assemblyai-universal-3-5-pro`, `soniox-stt-async-v5`.

Transcripts are cached per file under `results/transcripts/` (committed);
re-runs skip cached files, so interrupted runs resume and APIs are never
double-billed. `--limit N` transcribes only the first N records (smoke tests).

## Tests

```bash
uv run pytest        # no network, no GPU, no API keys needed
```
