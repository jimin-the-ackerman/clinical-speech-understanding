"""Medical-term recall: the fraction of reference clinical entities whose
surface form survives into the hypothesis. Complements WER, which weighs every
token equally — see docs/research/2026-07-06-medical-entity-asr-metrics.md.

Two deliberately separate stages so we can compare entity-identification methods
(dictionary / typed NER / LLM) on equal footing:

- build_manifest(): run an entity extractor over the unique references and freeze
  the result to results/entity_manifests/<method>.json. The *method* lives here,
  and so does all the cost/nondeterminism (a model download, API tokens). NER runs
  on the raw reference — clinical extractors want casing/punctuation that
  normalize_en strips.
- score(): pure offline aggregation of recall from a frozen manifest. No NER, no
  keys, deterministic, testable. Comparing methods is just scoring different
  manifests, and the frozen manifest keeps even an LLM-built entity set reproducible.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from stt_eval import store
from stt_eval.normalize import normalize_en


def _norm_tokens(text: str) -> list[str]:
    return normalize_en(text).split()


def entity_hit(entity: str, hyp_tokens: list[str]) -> bool:
    """True if the entity's normalized tokens appear as a contiguous run in hyp.

    Token-level (not substring) so "art" does not match inside "heart".
    """
    ent = _norm_tokens(entity)
    if not ent:
        return False
    n = len(ent)
    return any(hyp_tokens[i : i + n] == ent for i in range(len(hyp_tokens) - n + 1))


def file_recall(entities: list[str], hyp: str) -> tuple[int, int]:
    """(hits, total) for one file: how many reference entities survive in hyp."""
    hyp_tokens = _norm_tokens(hyp)
    return sum(entity_hit(e, hyp_tokens) for e in entities), len(entities)


# --- build: method-specific, run once, frozen to a manifest ---------------

def build_manifest(results_root: Path, extract) -> list[dict]:
    """Run `extract(reference) -> [surface forms]` over each unique (dataset,
    file_id) and return manifest entries. The same reference recurs across every
    model in the cache, so it is extracted once."""
    seen: dict[tuple, dict] = {}
    for r in store.read_results(results_root):
        key = (r["dataset"], r["file_id"])
        if key not in seen:
            seen[key] = {
                "dataset": r["dataset"], "file_id": r["file_id"],
                "entities": extract(r["reference"]),
            }
    return [seen[k] for k in sorted(seen)]


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")


def load_manifest(path: Path) -> dict[tuple, list[str]]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(e["dataset"], e["file_id"]): e["entities"] for e in entries}


# --- score: offline, deterministic, method-agnostic -----------------------

def score(results_root: Path, entities_by_key: dict[tuple, list[str]]) -> list[dict]:
    """Aggregate recall per (model, dataset, condition) from a frozen entity set.

    `entities_by_key` maps (dataset, file_id) -> entity surface forms; get it from
    load_manifest(). Files with no entities contribute nothing (no clinical terms
    to recall)."""
    groups: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # -> [hits, total]
    for r in store.read_results(results_root):
        if r.get("failed"):
            continue
        entities = entities_by_key.get((r["dataset"], r["file_id"]))
        if not entities:
            continue
        hits, total = file_recall(entities, r["text"])
        g = groups[(r["model"], r["dataset"], r.get("condition") or "")]
        g[0] += hits
        g[1] += total

    summary = []
    for (model, dataset, condition), (hits, total) in sorted(groups.items()):
        summary.append({
            "model": model, "dataset": dataset, "condition": condition,
            "n_entities": total, "n_hits": hits,
            "entity_recall": round(hits / total, 4) if total else None,
        })
    return summary


def write_outputs(summary: list[dict], results_root: Path, name: str) -> None:
    """Write results_root/{name}.csv and .md (name carries the method, e.g.
    entity_recall_bc5cdr)."""
    results_root.mkdir(parents=True, exist_ok=True)
    if not summary:
        return
    cols = list(summary[0])
    with (results_root / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    esc = lambda v: str(v).replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in summary]
    (results_root / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- extractors: one per entity-identification method ----------------------

def extractor_for(method: str):
    """Return extract(text) -> [surface forms] for a named method. Each method's
    heavy deps are imported only when that method is chosen."""
    if method == "bc5cdr":
        return _scispacy_extractor("en_ner_bc5cdr_md")
    raise SystemExit(f"entity method {method!r} is not implemented yet")


def _scispacy_extractor(model: str):
    try:
        import spacy
    except ImportError as e:
        raise SystemExit("scispaCy method needs: uv sync --extra entities") from e
    try:
        nlp = spacy.load(model, exclude=["parser", "lemmatizer"])
    except OSError as e:
        raise SystemExit(
            f"scispaCy NER model {model!r} is not installed. Install the matching "
            f"wheel from https://github.com/allenai/scispacy#available-models"
        ) from e

    # ponytail: per-doc NER; switch to nlp.pipe(unique_refs) if building gets slow
    def extract(text: str) -> list[str]:
        return [ent.text for ent in nlp(text).ents]

    return extract
