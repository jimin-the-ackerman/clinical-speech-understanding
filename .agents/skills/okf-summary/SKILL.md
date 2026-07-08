---
name: okf-summary
description: Read an OKF knowledge bundle and tell the story of what it documents — the question, what was learned, and what's left — in plain, human language. Use when someone wants to get up to speed on a project without reading the whole codebase.
---

You are briefing a smart colleague who has never seen this project. They want the story, not a
file dump. Read the knowledge bundle, then write one page a busy person actually enjoys reading.

## Find your way around the bundle

This is an [OKF](https://github.com/GoogleCloudPlatform/knowledge-catalog) v0.1 bundle — a
directory of markdown files, each with YAML frontmatter. Two things navigate it, so you never
have to know the filenames in advance:

- **`index.md`** is the reserved map of the bundle. Start there; it links everything.
- Every concept file declares a **`type`** in its frontmatter (and a title). Use those to find
  the pieces you need.

Locate at least these three roles, then read them:

1. The **overview** — why the project exists and what it's trying to do.
2. The **finding(s)** — what was actually learned, with the numbers that back it.
3. The **status** — what's done and what's still open (the todos).

Then follow any link you need to get a claim right — a dataset, a metric, a method. Don't guess.
(`log.md`, if present, is the bundle's history — skim only if you need the timeline.)

If a number or claim matters, trace it to the file that owns it. Don't invent figures. If the
bundle and a raw results file disagree, trust the raw results and say so.

## Write it like this

Give it a title that states the finding itself — a plain sentence a reader could repeat — not a
label like "Summary" or "OKF Summary".

One page, ~400–600 words. Structure it as a story, not an outline:

1. **The hook** — open with the single most surprising or counterintuitive thing. Make the very
   first sentence short and blunt: one plain claim, no clauses stacked on it. (In this bundle, for
   example: a cheaper, plainer model beats the fancy one at the thing that actually matters.) Lead
   with that, not with methodology.
2. **The question** — why the project is worth doing, in a sentence a non-specialist gets.
3. **What was done** — the shortest possible sketch. Names of the datasets and methods, not their
   internals.
4. **What was found** — the headline result, and how robust it is. A couple of real numbers, not
   a wall of them.
5. **The honest limits** — one short paragraph. Don't oversell.
6. **What's next** — the open todos, as a short list someone could act on.

## Voice — friendly and human

Write like a sharp person explaining their work to a friend over coffee, not like a model
generating a report.

- **Plain words.** "leftover" not "residual", "checked" not "validated", "makes up words" not
  "generates hallucinated output." Explain jargon the first time you use it — e.g., if the
  project is about speech recognition, "WER" is just "how many words it gets wrong."
- **Short sentences.** Vary the length. Read it aloud in your head — if you run out of breath, cut it.
- **Skip the obvious AI tells (first pass).** Cut "delve", "leverage", "robust", "seamless",
  "it's worth noting", the "It's not just X, it's Y" construction, em-dash pile-ups, emoji, and
  hype adjectives on the results — let the numbers impress on their own. This is only a first
  guardrail; the de-slop step below is the thorough backstop, so don't agonize here.
- **Tell, don't list.** Prose over bullets, except the final todos. One good sentence beats three
  bullet fragments.
- **Say the surprising part plainly.** "The fancy model was the worst one" lands harder than a
  hedged, jargon-wrapped version of the same sentence.
- **Own the uncertainty.** "We think", "so far", "on this one dataset" are honest and human. Fake
  confidence is an AI tell too.

## De-slop — the second pass

A model can't reliably see its own AI tells, so don't trust the draft to be clean. After writing
it, run the draft through the **`avoid-ai-writing`** skill's rewrite pass — a deterministic scan
catches "leverage", the fourth em-dash, and bare noun-phrase bullet lists that a self-review waves
through. If that skill isn't installed, do one honest pass yourself against the tells above and cut
anything that reads like a report.

## Deliver

Print the summary in the reply by default — the point is a fresh brief on the fly, and a file
that's never written can't go stale or clutter anything. Only if the person asks to keep it, save
it to a gitignored scratch dir (in this repo, `.tmp/okf-summary.md`); never create a tracked file.
