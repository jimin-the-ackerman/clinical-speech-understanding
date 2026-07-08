---
name: okf-freshness
description: Check whether the knowledge/ OKF bundle has drifted stale against recent code changes. Spawns a subagent to cross-reference a git diff with the bundle and report which concept files likely need updating — the agent proposes, you verify. Use before a release, after a batch of code changes, or when you suspect drift.
user-invocable: true
argument-hint: "[git ref/range, default main...HEAD]"
---

# Check knowledge-bundle freshness

Conformance (broken links, frontmatter) is mechanical — that is `okf-validate`. **Freshness** is a
judgment: did a code change make a concept's *prose* stale? No regex can decide that, so this is
"agent proposes, you verify."

Cross-referencing a diff against ~30 concept files is token-heavy and would clutter this
conversation, so **do the whole analysis in a subagent** and surface only its verdict. The main
thread should never read the diff or the concept files itself.

## Steps

1. Pick the range to check. Default `main...HEAD` (what this branch changed); use any ref or range
   the user passes (a commit, `HEAD~5`, `--staged`). You do not run the diff yourself — the
   subagent does, so neither the diff nor the concept files enter this context.

2. Spawn ONE general-purpose subagent with this task (substitute the range for `<RANGE>`):

   > Audit whether this repo's OKF knowledge bundle is stale after code changes. Edit nothing.
   > 1. Run `git diff <RANGE> -- src/ pyproject.toml` to see what changed. Ignore tests,
   >    `results/`, and `knowledge/` itself.
   > 2. Read the code→concept **trigger map** in `AGENTS.md` (the "Keep it current" table).
   > 3. For each code change, map it to the candidate concept file(s) under `knowledge/`, then
   >    read both the changed code and those concepts. Read only what you need.
   > 4. Judge staleness: a renamed / added / removed model, dataset, method, CLI flag, or extra;
   >    a behavior the concept describes that changed; a number or result the bundle states that
   >    moved; a removed feature still documented as present.
   > Return a compact report only:
   >   - For each concept that looks stale: `path — reason (confidence: high/med/low) — the
   >     specific edit needed`.
   >   - One closing line naming the concepts you checked and judged still fresh.
   >   - If nothing is stale, say so in one line.

3. Relay the subagent's report, highest-confidence items first. For each stale item, offer to make
   the edit — the user confirms, because freshness is a judgment call, not a mechanical fix.

## Notes

- On-demand only. Run it when you choose (before a release, after a batch of changes); never wire
  it to a commit hook — a per-commit staleness check is mostly false positives (most code changes
  do not touch documented behavior).
- Pairs with `okf-validate` (mechanical conformance gate) and the `AGENTS.md` trigger map — the
  subagent uses the same code→concept mapping a human would.
