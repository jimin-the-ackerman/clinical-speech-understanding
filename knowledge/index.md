# clinical-speech-understanding — knowledge bundle

OKF v0.1 knowledge catalog for this repo: an STT benchmark harness for a Korean
healthcare "AI scribe" (phase 1: English). Each concept is one markdown file with YAML
frontmatter; links are relative so they resolve on GitHub. Start here, then follow links.

## Project
- [Overview](project/overview.md) — goal, scope, phase 1
- [Research plan](project/research-plan.md) — the questions, the design, the phased roadmap
- [Architecture](project/architecture.md) — the prepare → transcribe → score pipeline
- [Environment & packaging](project/environment.md) — uv, extras, the CUDA-12.6 torch pin

## Datasets
- [Record model & registry](datasets/record-model.md) — the loader contract + `condition`
- [PriMock57](datasets/primock57.md) · [MedDialog-Audio](datasets/meddialog-audio.md) · [LibriSpeech test-other](datasets/librispeech-test-other.md) · [Fareez OSCE interviews](datasets/fareez-interviews.md)

## Models (ASR backends)
- [Transcriber protocol & registry](models/transcriber-protocol.md)
- Local (GPU): [Whisper](models/whisper-local.md) · [Qwen3-ASR](models/qwen3-asr.md)
- API: [gpt-4o-transcribe](models/gpt-4o-transcribe.md) · [Soniox](models/soniox.md) · [Deepgram](models/deepgram.md) · [AssemblyAI](models/assemblyai.md)

## Metrics
- [WER](metrics/wer.md) · [Text normalization](metrics/text-normalization.md) · [Medical-term recall](metrics/medical-term-recall.md)

## Entity-identification methods
- [bc5cdr](entity-methods/bc5cdr.md) · [med7](entity-methods/med7.md) · [stanza-i2b2](entity-methods/stanza-i2b2.md) · [medgemma](entity-methods/medgemma.md) · [openrouter](entity-methods/openrouter.md)

## Components
- [Transcript cache](components/transcript-cache.md) · [Orchestration](components/orchestration.md) · [CLI](components/cli.md)

## Findings & status
- [Finding: medical-term recall](findings/medical-term-recall.md) — the headline result
- [Project status](status.md) — live progress and open todos

## Runbooks
- [Setup](runbooks/setup.md) · [GPU benchmark](runbooks/gpu-benchmark.md) · [Evaluation workflow](runbooks/evaluation-workflow.md)

## Elsewhere in the repo (not part of this bundle)
- `docs/research/2026-07-06-medical-entity-asr-metrics.md` — literature survey with citations
- `docs/superpowers/` — historical design spec + TDD implementation plans
- `README.md` — human entry point (setup/usage); `CLAUDE.md` — agent guidance
