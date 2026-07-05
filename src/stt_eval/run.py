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

    for q in (prep, tr):
        q.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        raise SystemExit("prepare: not implemented yet")
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
