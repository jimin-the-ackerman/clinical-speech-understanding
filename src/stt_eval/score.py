import csv
import re
from collections import defaultdict
from pathlib import Path

from stt_eval import store
from stt_eval.metrics import corpus_wer, file_wer
from stt_eval.normalize import normalize_en

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]")


def score(results_root: Path) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in store.read_results(results_root):
        groups[(r["model"], r["dataset"], r.get("condition") or "")].append(r)

    summary, per_file = [], []
    for (model, dataset, condition), items in sorted(groups.items()):
        n_failed = n_empty = 0
        refs, hyps, secs, audio_secs = [], [], 0.0, 0.0
        for r in sorted(items, key=lambda x: x["file_id"]):
            if r.get("failed"):
                n_failed += 1
                continue
            ref, hyp = normalize_en(r["reference"]), normalize_en(r["text"])
            if not _NON_ALNUM.sub("", ref):
                n_empty += 1
                continue
            refs.append(ref)
            hyps.append(hyp)
            secs += r.get("seconds") or 0.0
            audio_secs += r.get("audio_seconds") or 0.0
            per_file.append({
                "model": model, "dataset": dataset, "condition": condition,
                "file_id": r["file_id"], "wer": round(file_wer(ref, hyp), 4),
                "seconds": r.get("seconds"), "audio_seconds": r.get("audio_seconds"),
            })
        summary.append({
            "model": model, "dataset": dataset, "condition": condition,
            "n_scored": len(refs), "n_failed": n_failed, "n_empty_ref": n_empty,
            "wer": round(corpus_wer(refs, hyps), 4) if refs else None,
            "rtf": round(secs / audio_secs, 4) if audio_secs else None,
        })
        if n_failed or n_empty:
            print(f"[warn] {model}/{dataset}/{condition or '-'}: "
                  f"{n_failed} failed, {n_empty} empty-reference files excluded")
    return summary, per_file


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(summary: list[dict], per_file: list[dict], results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    _write_csv(results_root / "wer_summary.csv", summary)
    _write_csv(results_root / "wer_per_file.csv", per_file)
    if summary:
        cols = list(summary[0])
        esc = lambda v: str(v).replace("|", "\\|")  # noqa: E731
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        lines += ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in summary]
        (results_root / "wer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
