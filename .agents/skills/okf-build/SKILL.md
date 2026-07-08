---
name: okf-build
description: Scaffold a new OKF knowledge bundle from a project's intent (a PRD/plan) and/or its codebase. Proposes a taxonomy fit to the project and writes empty concept stubs — it never auto-writes the prose; the author fills the "why". Use to start a knowledge/ bundle in a project that has none.
user-invocable: true
argument-hint: "[target-dir=knowledge] [--from <PRD/plan path>]"
---

# Scaffold an OKF knowledge bundle

Start a curated OKF v0.1 bundle: propose a structure, then write empty concept **stubs** for the
author to fill.

**Hard rule — scaffold only, never auto-author.** A concept body gets a one-line description and a
`<!-- TODO: the why -->` marker, nothing more. Writing prose *from* code produces docs that are
accurate about structure, blind about *why*, and stale the moment the code moves — the
"auto-generated codebase docs" anti-pattern OKF exists to avoid. The taxonomy is the machine's job;
the knowledge is the author's. If you ever feel tempted to fill a body, stop.

## Before anything: do not clobber

If the target dir (default `knowledge/`) already exists with an `index.md`, STOP — this builds a
*new* bundle. Offer to (a) target a different dir, or (b) add only the missing stubs. Never
overwrite an existing concept file.

## Steps

1. Decide the input(s): an intent doc (a PRD, research plan, design spec) via `--from`, a scan of
   the codebase, or both. More signal gives a better taxonomy.

2. Spawn ONE subagent for the analysis — it keeps this context clean and the reading cheap. Task:

   > Propose an OKF v0.1 bundle taxonomy for this project. Write no prose and create no files.
   > 1. If given an intent doc (`<PATH>`), read it. Otherwise, or in addition, survey the codebase
   >    structure — top-level dirs, modules, entry points, data models, external services, config.
   >    Read enough to identify the real *concepts*, not every file.
   > 2. Propose a flat set of concept files grouped into a few OKF subdirectories that fit THIS
   >    project (an ML repo might want `project/ datasets/ models/ metrics/ findings/`; a web app
   >    `project/ services/ data-models/ decisions/`). Do not force any fixed taxonomy. Every
   >    bundle needs `project/overview.md` and usually `project/architecture.md`.
   > 3. Return a compact list and nothing else: for each concept — `group/name.md`, its `type`, a
   >    one-line description, and the source it came from (which file or PRD section). Flag any
   >    concept you are unsure belongs.

3. Present the proposed taxonomy to the user and get approval or edits. The taxonomy is a judgment
   call — it is theirs to shape before anything is written.

4. On approval, scaffold (these writes are mechanical and tiny). Get the date once from `date +%F`;
   never invent it. For each approved concept write `<dir>/<group>/<name>.md`:

   ```
   ---
   type: <proposed type>
   title: <readable title>
   description: <the one-liner>
   tags: []
   timestamp: <date +%F>
   ---

   <!-- TODO: the why — what this is, how it works, the non-obvious decisions. -->
   ```

   Then write the two reserved files:
   - `index.md` — a titled map linking every stub, grouped, with relative links.
   - `log.md` — reserved header plus one entry: "Bundle scaffolded (structure only; concepts are
     stubs awaiting content)."

5. Run `okf-validate <dir>` to confirm the scaffold is conformant. Then tell the author: the
   structure is ready — fill each `<!-- TODO -->` with the real knowledge, starting with
   `project/overview.md`.

## Notes

- Stubs are intentionally empty. A bundle full of generated prose is worse than an honest skeleton,
  because it looks finished and is silently stale from day one.
- Pairs with `okf-validate` (the scaffold passes it immediately) and `okf-summary` (which stays
  thin until the stubs are filled — a useful signal of how much knowledge is still owed).
