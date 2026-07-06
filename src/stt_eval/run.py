import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stt-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="download and preprocess datasets")
    prep.add_argument("--datasets", required=True, help="comma-separated dataset names")

    tr = sub.add_parser("transcribe", help="run models over datasets, caching per-file JSON")
    tr.add_argument("--models", required=True, help="comma-separated model names")
    tr.add_argument("--datasets", required=True, help="comma-separated dataset names")
    tr.add_argument("--workers", type=int, default=8, help="parallel workers for API models")
    tr.add_argument("--limit", type=int, default=None, help="only first N records per dataset")

    sub.add_parser("score", help="compute WER tables from cached transcripts")

    eb = sub.add_parser("entity-build", help="extract reference entities to a frozen manifest")
    eb.add_argument("--method", required=True, help="entity-id method (e.g. bc5cdr)")
    eb.add_argument("--out", type=Path, default=None, help="manifest path (default by method)")

    es = sub.add_parser("entity-score", help="medical-term recall from an entity manifest")
    es.add_argument("--manifest", type=Path, required=True, help="entity manifest to score")

    for q in (prep, tr):
        q.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        from stt_eval import datasets

        for ds_name in args.datasets.split(","):
            print(f"preparing {ds_name}")
            datasets.prepare(ds_name, args.data_dir)
        return
    if args.cmd == "transcribe":
        from stt_eval import datasets, runner, transcribers
        from stt_eval.transcribers.base import MissingKeyError

        for ds_name in args.datasets.split(","):
            records = datasets.load(ds_name, args.data_dir)
            if args.limit:
                records = records[: args.limit]
            print(f"{ds_name}: {len(records)} records")
            for model_name in args.models.split(","):
                try:
                    t = transcribers.create(model_name)
                except MissingKeyError as e:
                    print(f"[skip] {model_name}: {e}")
                    continue
                runner.transcribe_dataset(t, records, ds_name, args.results_dir, args.workers)
        return
    if args.cmd == "score":
        from stt_eval.score import score, write_outputs

        summary, per_file = score(args.results_dir)
        write_outputs(summary, per_file, args.results_dir)
        for row in summary:
            print(row)
        return
    if args.cmd == "entity-build":
        from stt_eval.entity_score import build_manifest, extractor_for, write_manifest

        out = args.out or args.results_dir / "entity_manifests" / f"{args.method}.json"
        entries = build_manifest(args.results_dir, extractor_for(args.method))
        write_manifest(entries, out)
        n_ents = sum(len(e["entities"]) for e in entries)
        print(f"wrote {len(entries)} files, {n_ents} entities to {out}")
        return
    if args.cmd == "entity-score":
        from stt_eval.entity_score import load_manifest, score, write_outputs

        summary = score(args.results_dir, load_manifest(args.manifest))
        name = f"entity_recall_{args.manifest.stem}"
        write_outputs(summary, args.results_dir, name)
        for row in summary:
            print(row)
        return
