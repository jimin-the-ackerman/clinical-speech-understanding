"""Medical-term recall: the fraction of reference clinical entities whose
surface form survives into the hypothesis. Complements WER, which weighs every
token equally — see docs/research/2026-07-06-medical-entity-asr-metrics.md.

NER runs on the *raw* reference (clinical models want casing + punctuation, which
normalize_en strips); only the match test happens in normalized space, so it
lines up with the already-normalized hypothesis. scispaCy is imported lazily in
load_scispacy_extractor, so `stt-eval score` and the test suite never touch it.
"""

import csv
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


def score(results_root: Path, extract) -> list[dict]:
    """Aggregate medical-term recall per (model, dataset, condition).

    `extract(raw_reference) -> list[str]` is injected: the CLI passes a
    scispaCy-backed extractor, tests pass a stub, so the aggregation logic needs
    no NER model. Entities are cached per unique reference — the same reference
    is NER'd once even though it recurs across every model.
    """
    rows = list(store.read_results(results_root))
    ent_cache: dict[str, list[str]] = {}
    groups: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])  # -> [hits, total]
    for r in rows:
        if r.get("failed"):
            continue
        ref = r["reference"]
        if ref not in ent_cache:
            ent_cache[ref] = extract(ref)
        entities = ent_cache[ref]
        if not entities:  # no clinical terms in this reference: nothing to recall
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


def write_outputs(summary: list[dict], results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    if not summary:
        return
    cols = list(summary[0])
    with (results_root / "entity_recall.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)
    esc = lambda v: str(v).replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in summary]
    (results_root / "entity_recall.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_scispacy_extractor(model: str = "en_ner_bc5cdr_md"):
    """Return extract(text) -> [entity surface forms] backed by a scispaCy model.

    Kept out of module import so the offline path never needs scispaCy installed.
    """
    try:
        import spacy
    except ImportError as e:
        raise SystemExit(
            "entity scoring needs the optional deps: uv sync --extra entities"
        ) from e
    try:
        # disease + chemical NER only; drop parser/lemmatizer we don't use
        nlp = spacy.load(model, exclude=["parser", "lemmatizer"])
    except OSError as e:
        raise SystemExit(
            f"scispaCy NER model {model!r} is not installed. Install the matching "
            f"wheel from https://github.com/allenai/scispacy#available-models "
            f"(e.g. uv pip install <release-url>/{model}-<version>.tar.gz)"
        ) from e

    # ponytail: per-doc NER; switch to nlp.pipe(unique_refs) if scoring gets slow
    def extract(text: str) -> list[str]:
        return [ent.text for ent in nlp(text).ents]

    return extract
