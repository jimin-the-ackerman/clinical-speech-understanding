---
type: Entity Method
title: openrouter (general LLM, pending)
description: A general frontier LLM via OpenRouter — the specialized-vs-general foil; pending an API key.
tags: [entity-method, llm, openrouter, pending]
timestamp: 2026-07-08
---

# openrouter — general LLM (pending)

The one remaining method: a **general** frontier model via OpenRouter, the specialized-vs-general
foil to [medgemma](medgemma.md). Does a general LLM also rank Soniox #1? `openrouter_extractor`
in `entity_llm.py` (parallel-safe, `temperature=0`, resumable per-ref cache). **Blocked on
`OPENROUTER_API_KEY`.**

- **Run** (once keyed): bake off, then build with the winner.
  ```
  uv run --env-file .env stt-eval entity-bakeoff --specs "openrouter:anthropic/claude-opus-4.8,openrouter:google/gemini-2.5-flash" --limit 15
  uv run --env-file .env stt-eval entity-build --method openrouter --model <winner> --workers 8 --datasets primock57
  ```
- Candidate IDs + input $/M: `anthropic/claude-opus-4.8` ($5), `google/gemini-3.1-pro-preview`
  ($2), `google/gemini-2.5-flash` ($0.30). Full extraction ~$2–5. Deliberately **no OpenAI
  model** (gpt-4o-transcribe is a scored ASR system). No medical-specialized model exists on
  OpenRouter — MedGemma is the specialized route. See [status](../status.md).
