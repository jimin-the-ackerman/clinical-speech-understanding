"""MedDialog-Audio: synthetic medical dialogues (Orpheus TTS over MedDialog-EN)
with noise at multiple SNR levels. https://huggingface.co/datasets/aline-gassenn/MedDialog-Audio

A fixed-seed subsample per noise condition is materialized to wav and recorded
in a committed manifest, so runs are reproducible and API cost stays bounded.
`load()` needs only the manifest + wavs — `huggingface_hub` (bundled with the
`datasets` package) is used by `prepare()` alone (install extra:
`uv sync --extra data`).
"""

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from stt_eval.records import Record

HF_REPO = "aline-gassenn/MedDialog-Audio"
PER_CONDITION = 300
SEED = 42
MANIFEST = Path("results/manifests/meddialog_audio.json")

# PIN AT EXECUTION — Step 4 findings (network schema inspection, 2026-07-05):
# `load_dataset_builder(HF_REPO)` reports `features=None`/`splits=None`: this repo
# ships no dataset script or dataset_infos.json, so `datasets` auto-infers it as a
# single AudioFolderConfig with every wav file assigned to one implicit split
# (`data_files == {NamedSplit("train"): [...]}` — hence SPLIT below). There is no
# HF "column" for condition or transcript to read off a loaded dataset: transcripts
# live in a repo-root `metadata.csv` (21068 rows — one per *clean* utterance; cols
# filename, duration_s, mean_rms_energy, mean_f0_hz, mean_spectral_centroid_hz,
# hnr_db, transcription). Noise condition is encoded only in the directory path and
# a 3-char filename suffix appended to a shared base id (`{dialogue_id}_{speaker}`):
# o00 = clean, b20/b40/b60 = background (hospital) noise at 20/40/60%, w02/w06/w10 =
# white noise at 2/6/10%. Total 147476 wav files = 21068 ids x 7 conditions, and all
# 7 conditions share the exact same 21068 base ids (verified against the metadata.csv
# id set). The same transcription applies to every noise variant of a given id (same
# underlying speech, noise added synthetically) — confirmed via the dataset README.
# Because there's no tabular schema to read a "column" from, prepare() below is
# restructured per the brief's fallback note: it lists repo files (metadata only),
# derives (condition, base_id) -> path from the filename suffix, then downloads only
# the wavs picked by `pick_subsample`.
SPLIT = "train"
AUDIO_COL = None  # no column: exact repo file path is resolved via CONDITION_SUFFIXES
TEXT_COL = "transcription"  # column name in metadata.csv
CONDITION_COL = None  # no column: condition comes from the filename suffix, not a field
METADATA_FILE = "metadata.csv"
CONDITION_SUFFIXES = {
    "o00": "clean",
    "b20": "background_noise_20",
    "b40": "background_noise_40",
    "b60": "background_noise_60",
    "w02": "white_noise_2",
    "w06": "white_noise_6",
    "w10": "white_noise_10",
}


def pick_subsample(
    ids_by_condition: dict[str, list[str]], per_condition: int, seed: int
) -> dict[str, list[str]]:
    out = {}
    for cond in sorted(ids_by_condition):
        ids = sorted(ids_by_condition[cond])
        rng = random.Random(f"{seed}:{cond}")
        out[cond] = sorted(rng.sample(ids, min(per_condition, len(ids))))
    return out


def _condition_and_id(relpath: str) -> tuple[str, str] | None:
    stem = relpath.rsplit("/", 1)[-1].removesuffix(".wav")
    suffix, base_id = stem[-3:], stem[:-3]
    cond = CONDITION_SUFFIXES.get(suffix)
    return (cond, base_id) if cond else None


def prepare(data_dir: Path) -> None:
    wavs = data_dir / "meddialog" / "wav"
    if MANIFEST.exists() and wavs.exists():
        return
    from huggingface_hub import HfApi, hf_hub_download  # optional 'data' extra

    wavs.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type="dataset")  # metadata only

    path_by_cond_id: dict[tuple[str, str], str] = {}
    ids_by_cond: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parsed = _condition_and_id(f) if f.endswith(".wav") else None
        if parsed is None:
            continue
        cond, base_id = parsed
        path_by_cond_id[(cond, base_id)] = f
        ids_by_cond[cond].append(base_id)

    meta_path = hf_hub_download(HF_REPO, METADATA_FILE, repo_type="dataset")
    id_to_text: dict[str, str] = {}
    with open(meta_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            id_to_text[row["filename"].removesuffix("o00.wav")] = row[TEXT_COL]

    chosen = pick_subsample(ids_by_cond, PER_CONDITION, SEED)

    manifest = []
    for cond, ids in chosen.items():
        for base_id in ids:
            file_id = f"{cond}__{base_id}"
            out = wavs / f"{file_id}.wav"
            if not out.exists():
                src = hf_hub_download(HF_REPO, path_by_cond_id[(cond, base_id)], repo_type="dataset")
                shutil.copy(src, out)
            manifest.append({"file_id": file_id, "condition": cond, "text": id_to_text[base_id]})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest)} entries to {MANIFEST}")


def load(data_dir: Path) -> list[Record]:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wavs = data_dir / "meddialog" / "wav"
    return [
        Record(e["file_id"], wavs / f"{e['file_id']}.wav", e["text"], e["condition"])
        for e in entries
    ]
