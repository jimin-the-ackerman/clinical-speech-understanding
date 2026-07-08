---
name: okf-visualize
description: Render this repo's OKF knowledge bundle (knowledge/) as a single self-contained, interactive HTML graph — concepts as nodes coloured by type, markdown links as edges, a wiki-style panel with rendered markdown and "Links to" / "Cited by" backlinks. Use when asked to visualize, graph, preview, or explore the bundle.
user-invocable: true
argument-hint: "[bundle-dir] [-o out.html]"
allowed-tools: Bash
---

# Visualize the OKF knowledge bundle

Render the bundle as an interactive HTML graph. The Cytoscape + marked libraries are inlined from
`vendor/`, so the output needs no network — it works offline and inside a published Artifact's CSP.

```bash
mkdir -p .tmp
uv run .agents/skills/okf-visualize/scripts/okf_visualize.py knowledge \
  -o .tmp/knowledge-viz.html \
  -t "clinical-speech-understanding — knowledge bundle" \
  -l "https://github.com/jimin-the-ackerman/clinical-speech-understanding"
```

Writes to `.tmp/knowledge-viz.html` (gitignored scratch). Open it in any browser, or hand the
single file to someone. Nodes are concepts (colour = `type`, size = body length); edges are
markdown links; click a node for its rendered markdown and backlinks. Change the initial layout
with `--layout {cose,concentric,breadthfirst,circle,grid}`.

Adapted from scaccogatto/okf-skills (github.com/scaccogatto/okf-skills), MIT © 2026 Marco Boffo
(see `NOTICE.upstream`). The one change from upstream: the JS libraries are vendored under
`vendor/` and inlined at render time (upstream loads them from a CDN), making `viz.html` truly
self-contained.
