"""Offline cpWER vs flat WER from cached transcripts (no API calls).

Reads the `by_speaker` buckets that `stt-eval transcribe --diarize` caches next to
the flat text, pairs them with per-speaker references recovered from each dataset's
unflattened source (PriMock57 TextGrid tiers, Fareez D:/P: tags), and prints corpus
flat WER vs cpWER per (dataset, model). Replaces the retired API-calling probe
(scripts/diarize_probe.py, see docs/superpowers/specs/2026-07-09-diarization-cpwer-probe.md).

    uv run python scripts/score_cpwer.py [--results-dir results] [--data-dir data]

NOTE: normalizes with normalize_en so cpWER is comparable to our own flat WER,
not to published (unnormalized) cpWER.
"""

import argparse
import json
from pathlib import Path

from stt_eval.metrics import corpus_wer, cpwer_align
from stt_eval.normalize import normalize_en


def _ref_primock57(data_dir: Path, file_id: str) -> dict[str, str]:
    from stt_eval.datasets.primock57 import speaker_reference

    tg = data_dir / "primock57" / "repo" / "transcripts"
    return speaker_reference(tg / f"{file_id}_doctor.TextGrid", tg / f"{file_id}_patient.TextGrid")


def _ref_fareez(data_dir: Path, file_id: str) -> dict[str, str]:
    from stt_eval.datasets.fareez import _read_transcript, speaker_reference

    txt = data_dir / "fareez-interviews" / "Data" / "Clean Transcripts" / f"{file_id}.txt"
    return speaker_reference(_read_transcript(txt))


SPEAKER_REFS = {"primock57": _ref_primock57, "fareez-interviews": _ref_fareez}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    groups: dict[tuple[str, str], list[dict]] = {}
    for p in sorted((args.results_dir / "transcripts").rglob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("by_speaker") and d["dataset"] in SPEAKER_REFS:
            groups.setdefault((d["dataset"], d["model"]), []).append(d)
    if not groups:
        raise SystemExit("no cached transcripts with by_speaker — run stt-eval transcribe --diarize")

    print("| dataset | model | n | flat WER | cpWER |")
    print("| --- | --- | --- | --- | --- |")
    for (ds, model), payloads in sorted(groups.items()):
        flat_r, flat_h, cp_r, cp_h = [], [], [], []
        for d in payloads:
            ref = {k: normalize_en(v) for k, v in SPEAKER_REFS[ds](args.data_dir, d["file_id"]).items()}
            hyp = {k: normalize_en(v) for k, v in d["by_speaker"].items()}
            r, h = cpwer_align(ref, hyp)
            cp_r += r
            cp_h += h
            flat_r.append(normalize_en(d["reference"]))
            flat_h.append(normalize_en(d["text"]))
        print(f"| {ds} | {model} | {len(payloads)} "
              f"| {corpus_wer(flat_r, flat_h):.4f} | {corpus_wer(cp_r, cp_h):.4f} |")


if __name__ == "__main__":
    main()
