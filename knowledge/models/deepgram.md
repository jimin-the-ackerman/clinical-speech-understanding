---
type: Model
title: Deepgram nova-3-medical (API)
description: Deepgram nova-3-medical; configured but not run this round (no key).
tags: [model, api, deepgram, not-run]
timestamp: 2026-07-08
---

# Deepgram `nova-3-medical` — API

`transcribers/deepgram_api.py`, `parallel_safe = True`, needs `DEEPGRAM_API_KEY`. POSTs raw
audio bytes to `/v1/listen` (`smart_format=true, language=en`), reads
`results.channels[0].alternatives[0].transcript`.

**Status**: registered and implemented, but **not run this round** (no key configured). A
Korean-capable candidate for phase 2.
