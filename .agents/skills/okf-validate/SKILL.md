---
name: okf-validate
description: Check that this repo's OKF knowledge bundle (knowledge/) is conformant with OKF v0.1 (§9). Use when asked to validate, lint, or check the bundle, or before committing changes to it. Runs a deterministic Python checker, not an eyeball pass.
user-invocable: true
argument-hint: "[bundle-dir] [--strict] [--json]"
allowed-tools: Bash
---

# Validate the OKF knowledge bundle

Run the deterministic conformance checker. Default to this repo's `knowledge/` bundle when no
path is given. The script declares its own `pyyaml` dependency (PEP 723), so `uv run` needs no
setup:

```bash
uv run .agents/skills/okf-validate/scripts/okf_validate.py knowledge
```

Interpret the result:

- **ERROR** — a hard §9 failure (no parseable frontmatter, or a missing/empty `type`). The bundle
  is non-conformant; fix every one.
- **warn** — soft guidance (missing recommended field, non-ISO `log.md` date, broken cross-link).
  Broken links are explicitly tolerated by the spec (§5.3) and never block. Fix when cheap.

Exit code is non-zero on any error. Add `--strict` to also fail on warnings (catches broken links,
missing recommended fields) and `--json` for machine-readable output. Run this before committing
bundle changes — it pairs with the AGENTS.md rule to keep `knowledge/` current.

Adapted from scaccogatto/okf-skills (github.com/scaccogatto/okf-skills), MIT © 2026 Marco Boffo
(see `NOTICE.upstream`); vendored here so it travels with the repo and defaults to our bundle.
