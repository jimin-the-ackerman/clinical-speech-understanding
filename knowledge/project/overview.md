---
type: Concept
title: Project overview
description: An STT benchmark harness for a Korean healthcare AI scribe; phase 1 benchmarks English medical ASR.
tags: [project, overview, asr, scribe]
timestamp: 2026-07-08
---

# Project overview

The end goal is a Korean transcription system for healthcare conversations
(doctor–nurse–patient, an "AI scribe"). This first phase benchmarks existing STT models on
**English** medical audio to establish the harness, conventions, and a model shortlist that
carry into the Korean phase.

**In scope (phase 1):** transcription accuracy — [WER](../metrics/wer.md), plus a
complementary [medical-term recall](../metrics/medical-term-recall.md) metric.
**Out of scope:** speaker diarization, clinical-note generation, streaming latency.

The harness runs headless (remote GPU server), caches every transcript so runs resume and
APIs are never double-billed, and keeps its test suite fully offline.

See [architecture](architecture.md) for the pipeline and [status](../status.md) for current
progress and open todos.
