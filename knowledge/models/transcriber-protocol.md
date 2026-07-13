---
type: Concept
title: Transcriber protocol & registry
description: The transcribe(Path)->str contract (plus the optional diarized tuple return), the lazy model registry, and shared API-call helpers.
tags: [models, protocol, registry]
timestamp: 2026-07-13
---

# Transcriber protocol & registry

Every ASR backend satisfies one structural contract (`transcribers/base.py`, a
`@runtime_checkable` Protocol):

- `name: str`, `parallel_safe: bool`, and `transcribe(audio_path: Path) -> str`.
- `parallel_safe` = `True` for API backends (safe to run concurrently) and `False` for local
  GPU backends (single instance) — [orchestration](../components/orchestration.md) uses it to
  size the thread pool.
- Optional extension: `transcribe` may return `(text, extras)` instead of a plain string; the
  runner merges `extras` into the cached payload. Only [Soniox](soniox.md) uses it (with
  `--diarize`: `extras = {"by_speaker": ...}`).

**Registry** (`transcribers/__init__.py`): `REGISTRY` maps a model name → a zero-arg factory
that imports its backend module lazily inside the closure, so an uninstalled SDK or missing key
only affects the model that needs it. `create(name)` instantiates and stamps the registry name.

**Shared helpers** (`base.py`): `MissingKeyError` (unset key → skip, don't crash),
`require_env(var)`, `with_retries(fn)` (exponential backoff; never retries a missing key),
`poll_until(...)` (async-job poller). API backends take an injectable `httpx.Client` — the test
seam for `MockTransport`.

Backends — local: [Whisper](whisper-local.md), [Qwen3-ASR](qwen3-asr.md); API:
[gpt-4o-transcribe](gpt-4o-transcribe.md) (two routes: OpenAI-direct and
`gpt-4o-transcribe-openrouter`), [Soniox](soniox.md), [Deepgram](deepgram.md),
[AssemblyAI](assemblyai.md).
