"""LibriSpeech test-other, downloaded directly from OpenSLR (no HF dependency).

Audio stays as the original 16 kHz mono FLAC — every backend we use accepts FLAC.
"""

import tarfile
from pathlib import Path

import httpx

from stt_eval.records import Record

URL = "https://www.openslr.org/resources/12/test-other.tar.gz"


def parse_trans_file(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            fid, _, ref = line.partition(" ")
            out.append((fid, ref))
    return out


def prepare(data_dir: Path) -> None:
    dest = data_dir / "librispeech"
    if (dest / "LibriSpeech" / "test-other").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / "test-other.tar.gz"
    if not tar.exists():
        print(f"downloading {URL} (~330 MB)")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with open(tar, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    with tarfile.open(tar) as tf:
        tf.extractall(dest, filter="data")


def load(data_dir: Path) -> list[Record]:
    root = data_dir / "librispeech" / "LibriSpeech" / "test-other"
    recs = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for fid, ref in parse_trans_file(trans.read_text(encoding="utf-8")):
            recs.append(Record(fid, trans.parent / f"{fid}.flac", ref))
    return recs
