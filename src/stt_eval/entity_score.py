"""Medical-term recall: the fraction of reference clinical entities whose
surface form survives into the hypothesis. Complements WER, which weighs every
token equally — see docs/research/2026-07-06-medical-entity-asr-metrics.md.

Two deliberately separate stages so we can compare entity-identification methods
(typed NER / LLM) on equal footing:

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
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

def _unique_references(results_root: Path, limit: int | None = None,
                       datasets: set | None = None) -> list[tuple]:
    """(dataset, file_id, reference) for each unique file, sorted. The same
    reference recurs across every model in the cache; keep the first seen.
    `datasets` (a set of names) restricts to those datasets — used to skip
    non-clinical LibriSpeech on the expensive LLM route."""
    seen: dict[tuple, str] = {}
    for r in store.read_results(results_root):
        if datasets and r["dataset"] not in datasets:
            continue
        seen.setdefault((r["dataset"], r["file_id"]), r["reference"])
    keys = sorted(seen)
    if limit:
        keys = keys[:limit]
    return [(d, f, seen[(d, f)]) for d, f in keys]


def build_manifest(results_root, extract, cache_dir=None, workers=1, limit=None,
                   datasets=None) -> list[dict]:
    """Run extract(reference)->[surface forms] over each unique (dataset,file_id).
    With cache_dir set, results are cached per reference (skip-if-exists) and
    extraction is parallelized — a crashed/expensive LLM run resumes and never
    re-bills. cache_dir=None is the original serial in-memory path. `datasets`
    restricts to a set of dataset names."""
    refs = _unique_references(results_root, limit, datasets)
    if cache_dir is None:
        return [{"dataset": d, "file_id": f, "entities": extract(ref)} for d, f, ref in refs]

    def work(item):
        d, f, ref = item
        cp = Path(cache_dir) / d / f"{store.safe_id(f)}.json"
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        try:
            entry = {"dataset": d, "file_id": f, "entities": extract(ref)}
            store.write_result(cp, entry)          # atomic; leaves failures uncached
            return entry
        except Exception as e:
            print(f"[entity-build] {d}/{f} failed: {e!r}")
            return {"dataset": d, "file_id": f, "entities": []}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        entries = list(ex.map(work, refs))
    return sorted(entries, key=lambda e: (e["dataset"], e["file_id"]))


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

def extractor_for(method: str, results_root: Path | None = None, model: str | None = None):
    """Return extract(text) -> [surface forms] for a named method. Each method's
    heavy deps are imported only when that method is chosen."""
    if method == "bc5cdr":
        return _scispacy_extractor("en_ner_bc5cdr_md")
    if method == "ner-union":
        return _ner_union_extractor()
    if method in ("llm", "openrouter"):
        from stt_eval.entity_llm import openrouter_extractor
        if not model:
            raise SystemExit("--model required (e.g. anthropic/claude-opus-4.8)")
        return openrouter_extractor(model)
    if method == "medgemma":
        from stt_eval.entity_llm import medgemma_extractor
        return medgemma_extractor(model or "google/medgemma-27b-text-it")
    raise SystemExit(f"entity method {method!r} is not implemented yet")


def _dedupe_ci(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


_LEADING_DET = re.compile(
    r"^(?:a|an|the|this|that|these|those|some|any|your|my|our|his|her|their|its)\s+",
    re.IGNORECASE,
)


def _strip_determiner(s: str) -> str:
    return _LEADING_DET.sub("", s).strip()


def _ner_union_extractor():
    """Med7 (drug / dosage / route / frequency) plus Stanza i2b2 (problem / test /
    treatment). i2b2 supersets bc5cdr's disease+chemical and is the only model that
    catches procedures and tests (ECG, thyroid profile, X-ray); Med7 alone gets the
    numeric dosages. bc5cdr pins spacy<3.8 and Med7 pins spacy>=3.8, so they cannot
    share an env — bc5cdr stays a separate baseline manifest."""
    import spacy

    med7 = spacy.load("en_core_med7_lg")
    import stanza

    procs = {"tokenize": "mimic", "ner": "i2b2"}
    stanza.download("en", processors=procs, verbose=False)  # i2b2 model is not bundled
    i2b2 = stanza.Pipeline("en", processors=procs, download_method=None, verbose=False)

    def extract(text: str) -> list[str]:
        ents = [e.text for e in med7(text).ents]
        # i2b2 spans include leading determiners ("your electrocardiogram"); strip
        # so recall matches the medical term, not the article
        ents += [_strip_determiner(e.text) for e in i2b2(text).ents]
        return _dedupe_ci(ents)

    return extract


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
