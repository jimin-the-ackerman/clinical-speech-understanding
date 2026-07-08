# AGENTS.md

Guidance for coding agents (Claude Code, Codex, and others) working in this repository. This is
the shared, agent-neutral guide; `CLAUDE.md` just points here.

## Project

STT benchmarking for a Korean healthcare "AI scribe" (phase 1: English).
Spec: `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`.
Plan: `docs/superpowers/plans/2026-07-05-stt-benchmark-harness.md`.

**Knowledge bundle: `knowledge/index.md`** — this repo's knowledge is cataloged as an OKF v0.1
bundle (datasets, models, metrics, entity methods, components, runbooks, findings). Start there;
live status + open todos live in `knowledge/status.md`. New knowledge → add a concept file under
`knowledge/` (markdown + YAML frontmatter, relative links) and update `index.md` / `log.md`.

**Current work (2026-07-08):** medical-term-recall exercise — four entity methods done and
agreeing (Soniox #1 on PriMock57): `bc5cdr`, `med7`, `stanza-i2b2`, `medgemma`. Optional
remaining: an OpenRouter general-LLM foil (awaits `OPENROUTER_API_KEY`). Also the OSCE/Fareez
dataset (loader done, transcription a pending paid checkpoint). See `knowledge/status.md`.

## Commands

- `uv sync --group dev` — install (extras: `--extra data`, `--extra local`, `--extra entities`)
- `uv run pytest` — test suite (no network/GPU/keys; keep it that way)
- `uv run pytest tests/test_score.py::test_score_groups_and_pools -v` — single test
- `uv run stt-eval prepare|transcribe|score` — the benchmark pipeline (see README)
- `uv run stt-eval entity-build --method X` + `entity-score --manifest P` — medical-term
  recall (bc5cdr/med7/stanza-i2b2/medgemma; see `knowledge/metrics/medical-term-recall.md`)

## Architecture

- `src/stt_eval/transcribers/` — one module per STT backend behind a lazy
  registry (`REGISTRY` in `__init__.py`); protocol: `transcribe(Path) -> str`.
  API backends use plain httpx (tests inject `httpx.MockTransport`).
- `src/stt_eval/datasets/` — loaders yielding `Record(file_id, audio_path,
  reference, condition)`; `prepare()` downloads, `load()` is offline.
- `src/stt_eval/score.py` + `metrics.py` — WER. `entity_score.py` + `entity_llm.py` —
  medical-term recall (two-stage: `entity-build` freezes a manifest, `entity-score` aggregates
  offline).
- `results/transcripts/{dataset}/{model}/{file_id}.json` — committed cache;
  the reference text is embedded, so `stt-eval score` needs no dataset access.
- WER convention: vendored Whisper English normalizer + jiwer corpus WER
  (Open ASR Leaderboard convention — numbers comparable to published ones).
- `data/` is gitignored; `results/` is committed.
- Deeper, per-component detail lives in the `knowledge/` bundle (start at
  `knowledge/project/architecture.md`).
