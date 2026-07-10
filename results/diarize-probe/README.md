# Diarization-probe transcript dumps

Raw per-file output of `scripts/diarize_probe.py` (speaker-attribution / cpWER probe), laid out
as `<dataset>/<model>/<file>.json`. Each JSON: `{model, file, flat_text, by_speaker:
{vendor_speaker_id: raw_text}, metrics: {flat_wer, cpwer, hyp_speakers}}` — text is
un-normalized, so any normalizer can re-score offline without re-paying the API.

Diarization is nondeterministic per file (speaker clustering, not recognition); each dump records
one run's draw. Spec: `docs/superpowers/specs/2026-07-09-diarization-cpwer-probe.md`.
