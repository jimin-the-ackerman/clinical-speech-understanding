---
type: Research Finding
title: Speaker attribution is roughly free on the deployment candidate
description: cpWER vs flat WER on two clinical corpora shows Soniox's diarization costs at most half a WER point, so speaker-tagged scribe input is viable. A single-backend case study, not a diarization benchmark.
resource: https://github.com/jimin-the-ackerman/clinical-speech-understanding/blob/stt-dev/knowledge/findings/speaker-attribution-cost.md
tags: [asr, diarization, cpwer, speaker-attribution, primock57, fareez, soniox]
timestamp: 2026-07-13
---

# Speaker attribution cost on the deployment candidate

**TL;DR** — A clinical scribe needs to know who said what. We measured what speaker attribution
costs in transcription accuracy on the benchmark's shortlisted backend (Soniox), by scoring the
same API responses two ways: flat WER on the time-ordered text, cpWER on the speaker-attributed
text against oracle references. On both clinical corpora the difference is within half a WER
point. **Attribution is roughly free on the deployment candidate**, so a speaker-tagged-turns
scribe architecture is viable at negligible accuracy cost.

## How to read this (the claim, sized honestly)

This is a **depth-on-the-winner case study, not a diarization benchmark**. The benchmark's
purpose is selecting a system to build on; once Soniox emerged as the candidate (top-tier WER,
#1 [medical-term recall](medical-term-recall.md) on both clinical corpora), the architectural
question was what attribution costs *on that system*, not how diarization backends compare.
Soniox was also the only practical choice in the lineup: the local models emit no speaker
labels, gpt-4o's diarizing variant failed a pilot (over-segmented 2-party audio into 3
speakers), and Deepgram/AssemblyAI await keys. The results below characterize one commercial
diarizer and should not be read as a property of speaker-attributed ASR generally.

## Result

cpWER (concatenated minimum-permutation WER, see [WER](../metrics/wer.md)) next to flat WER,
same Soniox responses, oracle per-speaker references from each dataset's unflattened source
(PriMock57 TextGrid tiers; Fareez `D:`/`P:` tags):

| corpus | n | audio | flat WER | cpWER | attribution cost |
|---|---|---|---|---|---|
| [PriMock57](../datasets/primock57.md) (room-mic, overlapping) | 57 | ~9 h | .1224 | .1166 | **−0.6 pt** |
| [Fareez/OSCE](../datasets/fareez-interviews.md) (clean turn-taking) | 272 | ~52 h | .0971 | .1016 | **+0.45 pt** |

The sign flip is mechanistic, not noise. cpWER differs from flat WER in two opposing ways: it
forgives interleaving (word order across speakers stops mattering) and it punishes
misattribution (a word in the wrong speaker's pile scores as error). On PriMock57's overlapping
audio the forgiveness dominates, so cpWER comes out *below* flat WER; on Fareez's clean
turn-taking there is little interleaving to forgive, leaving only clustering mistakes. Both
effects are small.

Two caveats needed to act on these numbers:

- **Per-file clustering is nondeterministic.** Recognition is stable across runs, but the
  speaker clustering draw varies; a crosstalk-heavy file can double its cpWER between runs
  (PriMock57 `day1_consultation11`: .149 vs .233). The corpus numbers reproduce; individual
  files may not. On Fareez the corpus delta sits in a small outlier tail (cpWER ≤ flat on
  148/272 files).
- **Not comparable to published cpWER.** We normalize with the Whisper English normalizer so
  cpWER is comparable to *our own* flat WER; published cpWER applies no linguistic
  normalization.

## Decision fed

Soft GO for the scribe-architecture option of speaker-tagged-turns input: half a WER point buys
attribution the scribe needs anyway, and the alternative is a separate diarization stage with
its own errors. Permanent cpWER integration into `stt-eval score` is deferred until a second
diarizing backend or the Korean datasets make it consumable (see [status](../status.md)).

## Pointers

- [WER](../metrics/wer.md) — cpWER mechanism and implementation (`cpwer`, `cpwer_align`).
- Probe design and gate logic: `docs/superpowers/specs/2026-07-09-diarization-cpwer-probe.md`.
- Raw speaker-attributed outputs: the `by_speaker` field in
  `results/transcripts/{primock57,fareez-interviews}/soniox-stt-async-v5/` (one API call per
  file produced both views). Scored offline by `scripts/score_cpwer.py`.
