# Spec — Speaker-attribution (cpWER) probe on PriMock57

**Status:** throwaway probe — results not committed to `findings/`, no permanent protocol change.
*(2026-07-13: the no-findings rule is superseded — with the Fareez extension complete, the two-corpus result is recorded in `knowledge/findings/speaker-attribution-cost.md`. The no-protocol-change rule still holds.)*
**Date:** 2026-07-09
**Precedes:** a possible Fareez extension + permanent metric (gated on this probe looking good).

## Problem Statement

The benchmark scores STT on a **flattened** transcript — both speakers merged into one
reference string — so it measures *what was said* but is blind to *who said it*. For a clinical
scribe, "the doctor prescribed paracetamol" and "the patient prescribed paracetamol" are the same
WER but a different clinical record. We don't yet know whether the diarization-capable backends
attribute words to the right speaker on real consultation audio, and the scribe architecture that
would consume those labels isn't decided. We need that number *before* committing to it, because
it's an input to the architecture choice — not a consequence of it.

## Solution

A standalone probe that, on PriMock57 only, re-requests the three diarization-capable API
backends with diarization enabled, recovers per-speaker reference text (which the loader already
parses but currently discards), and reports **cpWER** (concatenated minimum-permutation WER)
alongside the existing flat WER. One afternoon, a few API calls, nothing wired into `stt-eval`.
The output answers one question: *is speaker attribution solved-enough on clinical audio to build
on?* If yes → extend to Fareez and make it permanent. If no → Architecture A (flat transcript in),
and we never paid for it.

## User Stories

1. As a benchmark author, I want cpWER on PriMock57 for the diarization-capable backends, so that I can see whether speaker attribution is reliable on clinical audio before designing the scribe around it.
2. As a benchmark author, I want the probe to reuse the flat WER from the same diarized run, so that I can confirm enabling diarization didn't move the plain WER numbers I already trust.
3. As a benchmark author, I want the probe kept out of the `stt-eval` CLI and out of `findings/`, so that a disposable experiment doesn't accrete into permanent surface area.
4. As a benchmark author, I want cpWER to reduce to plain corpus WER when there's one speaker, so that the metric is continuous with the existing WER story rather than a separate universe.
5. As a benchmark author, I want the probe to run on the mixed (room-mic) PriMock57 audio, so that the diarizer does the real separation task rather than being handed oracle channels.

## Implementation Decisions

- **Metric — cpWER, not DER.** Concatenate each speaker's normalized words into one string per
  speaker; for the 2-speaker case try both hyp↔ref speaker assignments; return the assignment with
  the lower corpus WER. Reuse `normalize_en` per speaker before concatenation. DER is explicitly
  rejected: the product question is attribution, not timeline, and Fareez has no timestamps anyway.
- **`cpwer()` lives in `metrics.py`**, beside `corpus_wer`/`file_wer`, as a pure function over
  per-speaker text (`dict[str, str]` ref, `dict[str, str]` hyp → float). No I/O, no network.
- **Reference recovery.** PriMock57's speaker identity is recoverable from the two TextGrid files
  (`*_doctor.TextGrid` / `*_patient.TextGrid`); the un-flatten yields two speaker strings. The
  existing `merge_reference` flat path stays untouched — the probe adds a sibling that keeps
  identity rather than replacing the flattener.
- **Backend — Soniox only.** `stt-async-v5` with `"enable_speaker_diarization": true` added to
  the transcription request; each transcript token gains a `speaker` field (numeric id).
  - **gpt-4o-transcribe-diarize was trialled at n=3 then dropped**: Soniox already ranks #1–2 on
    flat WER, gpt-4o was ~85% of the probe's API cost (~$11 vs ~$2 for the full 57), and it
    over-segmented 2-party audio into 3 speakers. It remains a valid second backend if wanted
    later (`POST /v1/audio/transcriptions`, `response_format="diarized_json"`,
    `chunking_strategy="auto"`, `segments[].speaker`) — the recipe is kept in a code comment.
  - Deepgram / AssemblyAI dropped (no API keys). Whisper / Qwen dropped (no native diarization —
    they need a pyannote/NeMo pipeline, deferred).
- **Audio condition.** The already-produced mixed 16 kHz mono wav (room-mic). Do **not** feed
  separate channels — that's channel separation, an easier different task.
- **No protocol change.** `transcribe(Path) -> str` is untouched. The probe calls the diarization
  endpoints directly in its own script; the permanent segment-output protocol is deferred to the
  (gated) permanent version.
- **Output.** A small table (model × {flat WER, cpWER}) printed to stdout; raw per-file transcript
  dumps under `results/diarize-probe/` are **committed for sharing** (2026-07-10 revision — the
  original "nothing committed" rule was relaxed to share transcripts with a colleague). Numbers
  still stay out of `findings/`.

## Testing Decisions

Good tests here assert *external behavior* of pure logic, not network or wiring. Two seams:

- **`cpwer()` — `tests/test_metrics.py::test_cpwer`.** Prior art: the existing `corpus_wer`/
  `file_wer` tests in the same file. Assert: identical ref/hyp → 0.0; swapping which hyp speaker
  maps to which ref speaker returns the same (min) value; a single-speaker input equals
  `corpus_wer` on the same text. This covers the only non-trivial logic — the permutation min.
- **Reference un-flatten — `tests/test_primock57.py::test_speaker_split`.** Prior art: the existing
  PriMock loader tests. Assert the doctor TextGrid's utterances land in one speaker bucket and the
  patient's in the other (using the existing fixtures, no audio).

Network paths (the diarization API calls) get **no** test — they're throwaway and would only
exercise `MockTransport` plumbing already covered by `test_api_transcribers.py`.

## Out of Scope

- **Fareez.** Gated on this probe looking good; it's the paid transcription run (~$30–50 + GPU).
- **DER / JER / any timeline metric.** Rejected; Fareez has no timestamps regardless.
- **Open-weight diarizers** (pyannote 3.1, NeMo Sortformer/MSDD) and the ASR+diarizer pipelines
  that need them (WhisperX, Qwen3-ASR+pyannote) — deferred until after the API probe. Note the
  WhisperX ≥3.8.6 trap: accepting the wrong (old `speaker-diarization-3.1`) HF license yields
  silent empty speaker labels; the current backend is `speaker-diarization-community-1`.
- **pyannote or any local diarization pipeline** and any new dependency.
- **A permanent segment-output protocol** (`transcribe` returning speaker-tagged segments) and any
  `stt-eval` CLI subcommand — deferred to the permanent version if the probe succeeds.
- **Role identification** (labelling a cluster as "doctor" vs "patient"). cpWER's permutation
  handles the assignment; naming the roles is a separate concern.
- **Committing numbers** to `findings/` or the knowledge bundle.

## Further Notes

- cpWER's permutation is trivial at 2 speakers (2 assignments). If a later version admits >2
  speakers it becomes the assignment problem (Hungarian) — out of scope now, noted so the pure
  function's signature doesn't foreclose it.
- PriMock57's overlapping room-mic speech is simultaneously its WER-noise source and the hardest
  diarization case, which makes it the honest stress test: attribution that survives here will
  survive Fareez's cleaner Teams audio.
- The whole point is decision-timing: this measurement feeds the not-yet-made scribe-architecture
  choice (flat-transcript-in vs speaker-tagged-turns-in). See session memory `diarization-cpwer-probe`.
