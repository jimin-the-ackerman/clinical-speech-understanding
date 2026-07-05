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
        raise SystemExit("transcribe: not implemented yet")
    if args.cmd == "score":
        raise SystemExit("score: not implemented yet")
