# Change log

Chronological history of this knowledge bundle (OKF reserved file).

- **2026-07-13** — **gpt-4o on OSCE via OpenRouter; the 6-model cross-family table is complete.**
  New transcriber `gpt-4o-transcribe-openrouter` (OpenRouter's OpenAI-compatible multipart
  `/audio/transcriptions`; the documented chat-completions `input_audio` route 400s for this
  model). Separate registry entry for provenance — spot-check showed the two routes give
  near-but-not-identical transcripts (nondeterministic decode), so rows are not merged. 272/272,
  ~$19 in credits: flat WER **0.1483** (worst on OSCE), recall last or second-to-last on all
  four methods, echoing PriMock57. Soniox #1 unchanged; finding table updated to six models.
  gpt-4o-diarize (OpenAI-direct only) noted in status as the candidate second diarizer.

- **2026-07-13** — Added **`findings/speaker-attribution-cost.md`**, promoting the two-corpus
  cpWER result out of status/log into a durable finding, framed as a depth-on-the-winner case
  study (single backend, claim sized accordingly). Supersedes the probe spec's "numbers stay
  out of findings/" rule (noted in the spec); linked from `index.md`.

- **2026-07-11** — **Cross-family rerank confirmed on OSCE; finding updated.** `entity-score`
  over the four frozen manifests (no rebuild — manifests key on *reference* entities, already
  covering OSCE since the local round) put Soniox #1 on OSCE on every method (bc5cdr .964,
  med7 .935, stanza .963, medgemma .970), qwen3-asr-0.6b (WER co-winner) last on all four.
  `findings/medical-term-recall.md` rewritten: the OSCE section is now cross-family (5 models),
  superseding the local-only replication; TL;DR notes the two-corpus reproduction. Todo #2
  shrank to the skipped gpt-4o-on-OSCE pass (~$19, cost).

- **2026-07-11** — **Soniox on OSCE with diarization; Gate B answered.** `speaker_reference` in
  `datasets/fareez.py` (D:/P: split, the cpWER oracle) and `stt-eval transcribe --diarize`
  (Soniox-only flag; runner caches `by_speaker` next to the flat text, so one paid pass feeds
  WER and cpWER). Full 272-consult run: flat WER 0.0971 (ties qwen3-asr-0.6b for #1), cpWER
  0.1016 → attribution +0.45 pt, sign flipped vs PriMock57 (status.md Done has the reading).
  gpt-4o on OSCE skipped on cost (~$19 = 51.9 h × $0.006/min); todo #2 rewritten to the manifest
  rebuild. Also rescoped the probe dumps to `results/diarize-probe/<dataset>/<model>/` with a
  README (dumps: layout + nondeterminism caveat).

- **2026-07-10** — Documented the **speaker-attribution (cpWER) probe**. New code: `cpwer` /
  `cpwer_align` in `metrics.py` (documented in `metrics/wer.md`), `speaker_reference` in the
  PriMock57 loader (`datasets/primock57.md`), and the throwaway `scripts/diarize_probe.py`
  (Soniox, `enable_speaker_diarization`). Result on full n=57: flat WER 0.1224 vs cpWER 0.1164 —
  attribution ≈ free, gate GO (`status.md` Done + Fareez todo updated; `project/research-plan.md`
  scope line qualified). Raw per-file transcript dumps committed under `results/diarize-probe/`
  (the spec's no-commit rule was relaxed for sharing; diarization is nondeterministic per file, so
  dumps record one run's draw).

- **2026-07-10** — Hardened and validated the `medgemma` extractor (no change to the finding).
  Greedy decoding could loop on a token and truncate its JSON list, dropping 2 OSCE refs
  (`GAS0003`, `MSK0027`) to empty; `_parse_entity_list` now salvages the pre-loop terms (with unit
  tests in `tests/test_entity_llm.py`), so they recover to 6 and 3 terms and no reference is
  silently dropped. Checked two properties: the extractor is **~99% faithful** (extracted terms are
  present in the reference, on par with the NER methods), and its ranking is **invariant to
  decoding** — a full rebuild at `repetition_penalty=1.3` (the most divergent setting, ~32%
  entity-set churn) gave an identical model order on both datasets (qwen3-asr-1.7b #1 on OSCE,
  Soniox #1 on PriMock57), with absolute recall shifting ~1 pt. The rep-1.3 manifest was scratch and
  discarded; the committed `medgemma` manifest stays greedy.

- **2026-07-09** — OSCE / Fareez joined the comparison for the four **local** models. Transcribed
  all 272 consults with whisper-large-v3 / -turbo and qwen3-asr-0.6b / -1.7b; re-scored WER and
  rebuilt all four entity manifests to include OSCE. The medical-term-recall rerank **reproduces**
  (qwen3-asr-1.7b #1 on every method; the WER winner qwen3-asr-0.6b is the recall loser) — see
  `findings/medical-term-recall.md`. Found and fixed a loader bug on the way: 2 of the 272
  transcripts (`RES0002`, `RES0054`) ship as UTF-16, which `datasets/fareez.py` read as UTF-8 into
  null-byte garbage; added a BOM-aware `_read_transcript`. Metered APIs (Soniox, gpt-4o) still
  pending on OSCE.

- **2026-07-08** — Fixed bundle drift an `okf-freshness` audit surfaced: corrected the `Record`
  module path (`records.py`, not `datasets/records.py`); documented the missing `--data-dir` /
  `--out` CLI flags in `components/cli.md`; noted the model-suffixed manifest name
  (`<method>_<model>.json`) in `metrics/medical-term-recall.md` and `components/cli.md`; and
  clarified that the Qwen registry passes bare ids normalized to `-hf` at load time.

- **2026-07-08** — Added `project/research-plan.md` (research questions, design in brief, phased
  roadmap) as a first-class plan concept; linked from `index.md`. Fills the gap where the plan was
  only implicit — scattered across the overview, the finding, and status.

- **2026-07-08** — Bundle created. Adopted **OKF v0.1** as the repo's knowledge protocol and
  cataloged the codebase (project, datasets, models, metrics, entity methods, components,
  runbooks, findings, status). Migrated the reproducible/pedagogical content out of four
  `docs/` files and retired them: `docs/research_plan.md` (empty), `docs/gpu-runbook.md`
  (→ runbooks/gpu-benchmark), `docs/tutorials/evaluation-workflow.md` (→ metrics/wer +
  text-normalization), `docs/entity-metric-comparison.md` (→ entity-methods + findings +
  status). Moved `docs/reports/medical-term-recall.md` → findings/. Kept the literature
  survey and superpowers spec/plans (unique/historical); repointed README/CLAUDE and the
  `entity_llm.py` comment at the bundle.
