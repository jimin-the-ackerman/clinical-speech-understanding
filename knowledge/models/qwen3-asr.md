---
type: Model
title: Qwen3-ASR (local)
description: Qwen3-ASR-0.6B and -1.7B open ASR via transformers-native -hf checkpoints; Korean-capable.
tags: [model, local, qwen, gpu, korean-ready]
timestamp: 2026-07-08
---

# Qwen3-ASR — local

`qwen3-asr-0.6b` and `qwen3-asr-1.7b` (`Qwen/Qwen3-ASR-*-hf`), a dedicated open ASR family
(30 languages incl. Korean → reusable in phase 2). `transcribers/qwen3_asr.py`,
`parallel_safe = False`.

- Loads `AutoProcessor` + `AutoModelForMultimodalLM` (bfloat16, `device_map="auto"`).
- `transcribe()` scales `max_new_tokens = max(512, duration*12)` (from `soundfile` duration) so
  ~10-min consults aren't truncated, drives `processor.apply_transcription_request`, slices new
  tokens, decodes transcription-only.
- **Provenance**: native support landed in transformers#43838 / **v5.13.0** (the `local` extra
  floor). `qwen3-asr-1.7b` wins WER on PriMock57 and is #2 on
  [medical-term recall](../metrics/medical-term-recall.md) after Soniox.
