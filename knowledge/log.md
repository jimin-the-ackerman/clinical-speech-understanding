# Change log

Chronological history of this knowledge bundle (OKF reserved file).

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
