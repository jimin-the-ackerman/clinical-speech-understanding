# Change log

Chronological history of this knowledge bundle (OKF reserved file).

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
