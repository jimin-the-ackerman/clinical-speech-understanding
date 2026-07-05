# STT Benchmark Harness — Design

**Date:** 2026-07-05
**Status:** Approved design, pending implementation plan

## Context and Goal

The project's end goal is a Korean transcription system for healthcare conversations
(doctor–nurse–patient, "AI scribe"). This first phase benchmarks existing STT models on
English medical audio to establish a harness, conventions, and a model shortlist that
carry over to the Korean phase.

**This phase measures transcription accuracy only (WER).** Speaker attribution
(who said what) and clinical-note generation are explicitly out of scope for now.

## Decisions Made

| Decision | Choice |
| --- | --- |
| Model scope | Both open models and commercial cloud APIs |
| Eval scope | Transcription only — WER; no diarization, no note generation |
| Datasets | PriMock57 + MedDialog-Audio + LibriSpeech test-other |
| Model lineup | Korean-ready bias: prefer models usable in phase 2 |
| Compute | Local NVIDIA GPU(s) + remote GPU server; harness must run headless |
| Approach | Minimal custom harness (not an existing eval framework) |

## Model Lineup (8)

Local (GPU):

- `whisper-large-v3` — via faster-whisper
- `whisper-large-v3-turbo` — via faster-whisper
- `Qwen3-ASR-0.6B` — dedicated open ASR (Apache-2.0), 30 languages incl. Korean;
  speed/size reference point
- `Qwen3-ASR-1.7B` — same family, SOTA open WER; transformers or vLLM backend,
  long-form support

Cloud APIs (all four support Korean, so phase 2 reuses them):

- OpenAI `gpt-4o-transcribe` (latest snapshot pinned at implementation)
- Deepgram `nova-3-medical` (English round; Korean uses their multilingual model)
- AssemblyAI `universal-3` (latest revision pinned at implementation)
- Soniox `stt-async-v5`

Considered and excluded:

- NVIDIA Parakeet, NVIDIA Canary, Voxtral — no Korean support.
- Qwen2-Audio-7B — superseded as the open multilingual entry by the dedicated
  Qwen3-ASR models.
- OpenAI `gpt-realtime-2`, `gpt-audio-1.5`, `gpt-realtime-whisper`; Soniox `stt-rt-v5`
  — voice-agent / audio-chat / streaming models, not batch transcription endpoints;
  streaming evaluation is out of scope this phase.
Korean-phase additions (verify APIs when we get there): Naver Clova Speech, RaonSpeech,
Daglo, ReturnZero (VITO).

## Datasets (English round)

### PriMock57 — real conversational acoustics (primary signal)

57 mock primary-care consultations (Babylon Health, GitHub), ~10 min each, audio recorded
as separate doctor/patient channels with TextGrid transcripts.

- Prep: mix the two channels to one mono wav per consultation (ffmpeg) to simulate the
  room-mic condition a real scribe faces.
- Reference: time-ordered concatenation of both speakers' TextGrid utterances.
- Known caveat: overlapping speech makes reference word order slightly ambiguous; this
  adds noise to WER but is the honest test for the use case.

### MedDialog-Audio — medical vocabulary + noise robustness

Synthetic corpus: MedDialog-EN dialogue text normalized with gpt-4o-mini, synthesized
with Orpheus TTS, with white noise and hospital ambient sounds added at multiple SNR
levels. 10k+ dialogues, hosted at
`huggingface.co/datasets/aline-gassenn/MedDialog-Audio`.

- Fixed-seed subsample of ~300 dialogues per noise condition; the sampled manifest is
  committed so runs are reproducible and API cost stays low.
- Each record carries a `condition` field (SNR level) → per-model noise-robustness curve.
- Known caveat: TTS speech is acoustically easier than real speech. This set measures
  vocabulary and noise handling, not real-room performance; it complements PriMock57,
  it does not replace it.

### LibriSpeech test-other — sanity check

Standard per-utterance read-speech benchmark. Sole purpose: validate the harness by
comparing our numbers against published results.

## Architecture

```text
src/stt_eval/
  transcribers/        # one module per backend, registry name → factory
    base.py            # Transcriber protocol: transcribe(wav_path) -> str
    whisper_local.py   # faster-whisper: large-v3, large-v3-turbo
    qwen3_asr.py       # Qwen3-ASR-0.6B / 1.7B
    openai_api.py      # gpt-4o-transcribe
    deepgram_api.py    # nova-3-medical
    assemblyai_api.py  # universal-3
    soniox_api.py      # stt-async-v5
  datasets/            # each yields records: (file_id, wav_path, reference_text, condition)
    primock57.py       # download, channel-mix prep, TextGrid → reference
    meddialog_audio.py # HF loader, seeded subsample, materializes wavs, condition = SNR
    librispeech.py     # test-other
  normalize.py         # Whisper English text normalizer (Open ASR Leaderboard convention)
  metrics.py           # corpus + per-file WER via jiwer, grouped by (dataset, condition)
  run.py               # CLI entry point
data/                  # gitignored: downloads + materialized wavs
results/
  transcripts/{dataset}/{model}/{file_id}.json   # committed cache
  wer_summary.csv                                 # per (model, dataset, condition)
```

- One consultation/dialogue/utterance → one hypothesis text; each transcriber handles
  long-form audio its own way (Whisper decodes sequentially; APIs use async endpoints).
- `condition` is null for PriMock57 and LibriSpeech.
- Dataset loaders materialize audio to wav files under `data/` so the `wav_path`
  interface holds for both local models and API uploads.

## CLI and Configuration

- `stt-eval transcribe --models <names> --datasets <names>` — run models over datasets,
  writing per-file JSON caches. Existing cache files are skipped, so interrupted runs
  resume and API calls are never repeated.
- `stt-eval score` — read caches, apply normalization, emit `wer_summary.csv` and a
  markdown summary table.
- API keys via environment variables (names documented in README). A model whose key is
  missing is skipped with a warning, not an error.
- Runs headless (no display, no interactivity) for the remote GPU server.
- Package managed with `uv`; heavyweight/model-specific SDKs imported lazily inside
  their transcriber module so unused backends don't have to be installed.

## Metrics

- Primary: corpus WER (sum of edit errors / sum of reference words) per
  (model, dataset, condition), computed with `jiwer` after the Whisper English text
  normalizer — the Open ASR Leaderboard convention, so numbers are comparable to
  published results.
- Also reported: per-file WER distribution; wall-clock seconds per file as a rough
  speed/cost signal.
- Raw (unnormalized) transcripts are always stored, so re-scoring under different
  normalization never re-runs a model.
- Korean phase switches primary metric to CER — a contained change in `metrics.py`.

## Error Handling

- API calls: simple retry with exponential backoff. A file that still fails is recorded
  as `{"failed": true, ...}` in its cache slot and excluded from scoring with a visible
  warning (so a model's WER is never silently computed over a different subset without
  the report saying so).
- Scoring reports, per model, how many files were scored vs failed.

## Testing

- No-network unit tests: normalization + WER math against hand-computed examples;
  scoring aggregation with a fake cache directory.
- End-to-end smoke test with a fake transcriber over bundled tiny fixtures.
- Real model runs are the integration test; no network calls in the test suite.

## Korean Phase (forward look — not built now)

- Datasets: AI Hub Korean medical/conversation speech corpora (require application;
  Korean collaborator can apply), KsponSpeech; possibly TTS-synthesized Korean medical
  dialogues (relates to the `tts-dev` branch direction).
- Models: reuse Whisper family, Qwen2-Audio, and all four APIs; add Naver Clova Speech,
  RaonSpeech, Daglo, ReturnZero (VITO).
- Metric: CER primary; Korean-appropriate text normalization needs its own investigation.
- Everything else (harness, caching, CLI, results format) carries over unchanged.

## Out of Scope (this phase)

- Speaker diarization / speaker-attributed WER
- Clinical note generation and its evaluation
- Streaming/real-time transcription latency testing
- Fine-tuning any model
