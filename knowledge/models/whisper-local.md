---
type: Model
title: Whisper (local, faster-whisper)
description: whisper-large-v3 and -v3-turbo via faster-whisper / CTranslate2 on GPU.
tags: [model, local, whisper, gpu]
timestamp: 2026-07-08
---

# Whisper — local (faster-whisper)

`whisper-large-v3` and `whisper-large-v3-turbo` via **faster-whisper** (CTranslate2).
`transcribers/whisper_local.py`, `parallel_safe = False` (single GPU instance).

- Loads `WhisperModel(device="auto", compute_type="auto")`; `transcribe()` joins stripped
  segment texts. `-v3-turbo` is the fastest model in the lineup (RTF ~0.01).
- **`_preload_cuda_libs()`** (a `ponytail:` shim): `ctypes.CDLL(..., RTLD_GLOBAL)` the CUDA
  `.so`s from torch's `nvidia-*-cu12` wheels so CT2 finds libcublas/libcudnn without
  `LD_LIBRARY_PATH` — and **deliberately skips `libnvblas`**, which would hijack CPU BLAS and
  segfault Qwen in the same run. No-op off Linux/CUDA-12. See [environment](../project/environment.md).
