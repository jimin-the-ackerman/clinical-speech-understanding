# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

STT benchmarking for a Korean healthcare "AI scribe" (phase 1: English).
Spec: `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`.
Plan: `docs/superpowers/plans/2026-07-05-stt-benchmark-harness.md`.

**Current work (2026-07-07): `docs/entity-metric-comparison.md`** — live status of the
medical-term-recall exercise. Three entity methods done and agreeing (Soniox #1 on
PriMock57): `bc5cdr`, `ner-union`, `medgemma`. Optional remaining: an OpenRouter
general-LLM foil (awaits `OPENROUTER_API_KEY`). Also the OSCE/Fareez dataset (loader
done, data downloaded, transcription is a pending paid checkpoint). Read it before
resuming that work.

## Commands

- `uv sync --group dev` — install (extras: `--extra data`, `--extra local`)
- `uv run pytest` — test suite (no network/GPU/keys; keep it that way)
- `uv run pytest tests/test_score.py::test_score_groups_and_pools -v` — single test
- `uv run stt-eval prepare|transcribe|score` — the benchmark pipeline (see README)

## Architecture

- `src/stt_eval/transcribers/` — one module per STT backend behind a lazy
  registry (`REGISTRY` in `__init__.py`); protocol: `transcribe(Path) -> str`.
  API backends use plain httpx (tests inject `httpx.MockTransport`).
- `src/stt_eval/datasets/` — loaders yielding `Record(file_id, audio_path,
  reference, condition)`; `prepare()` downloads, `load()` is offline.
- `results/transcripts/{dataset}/{model}/{file_id}.json` — committed cache;
  the reference text is embedded, so `stt-eval score` needs no dataset access.
- WER convention: vendored Whisper English normalizer + jiwer corpus WER
  (Open ASR Leaderboard convention — numbers comparable to published ones).
- `data/` is gitignored; `results/` is committed.
