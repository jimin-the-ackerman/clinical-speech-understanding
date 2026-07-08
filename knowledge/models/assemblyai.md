---
type: Model
title: AssemblyAI universal-3-5-pro (API)
description: AssemblyAI universal-3-5-pro (async/polling); configured but not run this round (no key).
tags: [model, api, assemblyai, not-run]
timestamp: 2026-07-08
---

# AssemblyAI `universal-3-5-pro` — API

`transcribers/assemblyai_api.py`, `parallel_safe = True`, needs `ASSEMBLYAI_API_KEY`. Async:
upload bytes → create transcript job → `poll_until(status in {completed, error})` → return
text. Uses the `speech_models` list param (the singular `speech_model` is deprecated upstream).

**Status**: registered and implemented, but **not run this round** (no key configured). A
Korean-capable candidate for phase 2.
