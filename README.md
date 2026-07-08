# clinical-speech-understanding

Benchmarking STT models for a Korean healthcare "AI scribe". Phase 1: English
medical audio. **Knowledge bundle (start here): `knowledge/index.md`.** Coding agents: see
`AGENTS.md`. Design:
`docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`. Full benchmark on a GPU machine:
`knowledge/runbooks/gpu-benchmark.md`.

## Setup

Requires Python ≥3.10, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and `git` on PATH.

```bash
uv sync                      # core (scoring, API transcribers)
uv sync --extra data         # + HF datasets (MedDialog-Audio prepare)
uv sync --extra local        # + faster-whisper, transformers, torch, bitsandbytes (GPU + 4-bit MedGemma)
uv sync --extra entities     # + scispaCy for the bc5cdr entity method (see entity-build)
```

New to uv? See [Working with uv](#working-with-uv) below.

## Get a plain-language brief

To brief a colleague or get re-oriented fast, point your coding agent at the
`okf-summary` skill (`.agents/skills/okf-summary/`). It reads the `knowledge/` bundle and
writes a one-page, plain-language story of the research (the question, what we found, what's left),
generated fresh each time, so it never goes stale. Any AGENTS.md-aware agent (Claude Code, Codex, …)
can run it.

For the cleanest, least-AI-sounding output, also install the de-slop pass:
`npx skills add conorbronsdon/avoid-ai-writing`
([MIT](https://github.com/conorbronsdon/avoid-ai-writing)). `okf-summary` writes the friendly draft,
then hands it to that skill for a deterministic AI-tell sweep; without it, the skill falls back to a
lighter self-review.

**Claude Code** (it's a slash command):

```
/okf-summary
```

**Codex** (it uses `$` instead of `/`):

```
$okf-summary
```

Or, for any agent without a skill-command syntax, just point it at the file:

```
Read .agents/skills/okf-summary/SKILL.md and follow it to give me a
plain-language summary of this project's research.
```

### API keys

Transcriber and entity-method credentials live in a gitignored `.env`. Copy the template and fill
in whichever you have. A missing key just skips the model or step that needs it, with a warning:

```bash
cp .env.example .env      # then edit .env
```

Runs load it explicitly with `--env-file` (e.g. `uv run --env-file .env stt-eval transcribe …`):

| Variable | Unlocks |
|---|---|
| `OPENAI_API_KEY` | `gpt-4o-transcribe` |
| `SONIOX_API_KEY` | `soniox-stt-async-v5` |
| `HF_TOKEN` | `prepare` (MedDialog-Audio download) + the gated `medgemma` entity method |
| `OPENROUTER_API_KEY` | `entity-build --method openrouter` |

`DEEPGRAM_API_KEY` and `ASSEMBLYAI_API_KEY` are omitted above: those two models aren't part of the
current round, so leave them blank until you add the models back. Local Whisper / Qwen3-ASR models
and offline scoring (`score`, `entity-score`) need no keys.

## Usage

```bash
uv run --env-file .env stt-eval prepare --datasets librispeech-test-other,primock57,meddialog-audio
uv run --env-file .env stt-eval transcribe \
  --models whisper-large-v3-turbo,gpt-4o-transcribe \
  --datasets primock57 --workers 8
uv run stt-eval score       # -> results/wer_summary.{csv,md}, results/wer_per_file.csv  (offline, no keys)
```

Medical-term recall (a clinical-entity metric alongside WER; see
`knowledge/metrics/medical-term-recall.md`) is two-stage: `entity-build --method X` freezes the
reference entities to a manifest (the heavy, method-specific step; run it in a `uv run --with`
overlay so nothing permanent is installed), then `entity-score --manifest P` computes the
recall table offline. Methods: `bc5cdr`, `med7`, `stanza-i2b2` (NER) and `medgemma` (local LLM).

```bash
# build one method's manifest (bc5cdr shown; see knowledge/entity-methods/ for med7/stanza-i2b2/medgemma)
uv run --with scispacy \
  --with "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz" \
  stt-eval entity-build --method bc5cdr --datasets primock57,meddialog-audio
# score any manifest (offline, no NER/keys) -> results/entity_recall_bc5cdr.{csv,md}
uv run stt-eval entity-score --manifest results/entity_manifests/bc5cdr.json
```

`--results-dir` is a top-level flag (default `results/`), e.g.
`uv run stt-eval --results-dir X score`; normal usage needs nothing.

The current round runs six models; Deepgram and AssemblyAI are wired into the harness but not
run yet (no keys):

- **Run:** `whisper-large-v3`, `whisper-large-v3-turbo`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`,
  `gpt-4o-transcribe`, `soniox-stt-async-v5`
- **Configured, not yet run:** `deepgram-nova-3-medical`, `assemblyai-universal-3-5-pro`

Transcripts are cached per file under `results/transcripts/` (committed);
re-runs skip cached files, so interrupted runs resume and APIs are never
double-billed. `--limit N` transcribes only the first N records (smoke tests).

## Tests

```bash
uv run pytest        # no network, no GPU, no API keys needed
```

## Working with uv

[uv](https://docs.astral.sh/uv/) manages one virtual environment per project. You
never activate it; every `uv run` syncs the env first, then runs. Three files
define it:

- **`pyproject.toml`**: what you want (dependencies, optional `[extra]` groups,
  the `stt-eval` command). Hand-edited.
- **`uv.lock`**: exact resolved versions of everything, transitive deps included.
  Generated by uv; committed so every machine installs identically.
- **`.venv/`**: the installed environment. Disposable; rebuilt from the two above.

Everyday commands:

| Command | What it does |
|---|---|
| `uv sync [--extra X]` | Make `.venv` match `uv.lock`. Extras (`data`, `local`, `entities`) are optional and off by default. That keeps the core install light and the test suite offline. |
| `uv run <cmd>` | Run inside the env, syncing first. `uv run pytest`, `uv run stt-eval score`, `uv run python -c ...`. Why you never `activate`. |
| `uv run --env-file .env <cmd>` | Load env vars (API keys) from `.env` first, so the transcribe runs get their keys without exporting them to your shell. |
| `uv run --with <pkg> <cmd>` | Run in a throwaway overlay with extra packages, without touching `.venv` or the lockfile (see the `entity-build` example above). |
| `uv pip show <pkg>` / `uv pip tree` | Inspect what's installed (escape hatch; use sparingly). |

You edit `pyproject.toml`; uv regenerates `uv.lock` on the next sync, so you rarely touch
the lock by hand. **This repo's one non-obvious bit:** the `[tool.uv.sources]` and
`[[tool.uv.index]]` blocks in `pyproject.toml` pin `torch`/`torchcodec` to the
CUDA 12.6 wheel index instead of default PyPI. That is what makes the local GPU
models work on a CUDA-12 driver machine (see `knowledge/runbooks/gpu-benchmark.md`).
