---
name: cut-to-the-chase
description: Cut process-residue — content that served the writer's reasoning, not the reader — from durable artifacts (docs, PR descriptions, final answers). Use when a draft needs trimming for a fresh reader, when it reads like a work log ("first I…, then I…"), when asked to check a draft for residue without editing it, or as the content-selection pass before a stylistic pass like avoid-ai-writing.
---

# Cut to the chase

**Process-residue** is content a writer includes because it served their own reasoning, not the reader's need (the genus is Flower's *writer-based prose*). It reassures the **worrier** — the writer who, mid-task, had to hold the doubt; the **fresh reader** never had it, and a null result is not a finding.

## Register

Apply this skill to **durable artifacts** — docs, summaries, papers, reports, PR descriptions, README sections, the final write-up of a long task — where the reader never saw the work happen. Leave process in place in live dialogue (the reader co-reasoned and wants the seams), in walkthroughs the reader explicitly asked for ("show your reasoning"), and in the **audit layer** — the record itself: commit messages, changelogs, logs, appendices, reference docs — which is routed to, never stripped.

## The test

Not "does this reassure" — some reassurance is exactly what a reader needs. The test:

**Would a fresh reader independently arrive at this doubt?**

Route every candidate by its answer — routing, not bare deletion, is the move:

- **Keep, compressed** — the doubt is foreseeable (a known trap, a counterintuitive number, a step readers always question), or the passage is a caveat the reader needs to act correctly. One clause, not a paragraph.
- **Relocate** — true and worth recording, but only the worrier needed it. Move it to the audit layer. If no destination exists, hand the content back in conversation rather than silently dropping it.
- **Cut** — neither: effort justification, restated confidence, filler.

## Signals

Residue's usual shapes:

- "I checked X, Y, and Z and it held" — effort justification carrying a null result.
- Ruled-out branches the reader never proposed.
- Process framing inside a durable artifact: "first I…, then I…, that failed, so I…".
- Passages answering "what did the writer do" where the reader asked something else.

## Example

Draft: "To make sure the number was right I re-checked the loader, re-ran the scorer, and verified against the raw files; everything matched, so the WER of 12% is correct."

- Fresh readers take the number at face value → **cut**: "WER is 12%." The check list **relocates** to the commit message.
- Fresh readers would doubt it (a surprising number, a pipeline with a known trap) → **keep, compressed**: "WER is 12%, verified against the raw transcripts."

## Modes

**Detect** — when asked to check or audit a draft without changing it. Walk the artifact passage by passage; for each candidate, quote it, name the signal it matches, give the verdict (foreseeable or self-generated doubt), and the route. Done when every passage has been judged and every candidate carries a verdict and a route.

**Rewrite** (default) — when asked to trim, tighten, or fix. Run detect, then apply the routes: compress the keeps, move each relocation to a named destination, cut the rest. Edit files in place; return rewritten text for pasted drafts. Done when a top-to-bottom re-read answers the reader's question with no surviving signal, and every relocated passage names where it went. Report cuts and relocations briefly in conversation — the conversation is live dialogue; the artifact is not.

## Composition

This skill chooses *what* survives; run it first, then offer a stylistic pass (avoid-ai-writing) on what survives rather than fixing word-level tells here.
