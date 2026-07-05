"""LibriSpeech test-other, downloaded directly from OpenSLR (no HF dependency).

Audio stays as the original 16 kHz mono FLAC — every backend we use accepts FLAC.
"""

import shutil
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
        tmp = tar.with_suffix(".part")
        print(f"downloading {URL} (~330 MB)")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        tmp.rename(tar)

    # extract into a sibling temp dir first so an interrupted extract never
    # leaves a partial LibriSpeech/test-other that passes the exists-check above
    extract_tmp = dest / "extract.tmp"
    shutil.rmtree(extract_tmp, ignore_errors=True)  # leftover from a prior interrupted run
    extract_tmp.mkdir()
    with tarfile.open(tar) as tf:
        try:
            tf.extractall(extract_tmp, filter="data")
        except TypeError:  # Python < 3.10.12/3.11.4 lacks the filter kwarg
            tf.extractall(extract_tmp)

    target = dest / "LibriSpeech"
    shutil.rmtree(target, ignore_errors=True)
    (extract_tmp / "LibriSpeech").rename(target)
    shutil.rmtree(extract_tmp, ignore_errors=True)
    tar.unlink()


def load(data_dir: Path) -> list[Record]:
    root = data_dir / "librispeech" / "LibriSpeech" / "test-other"
    recs = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for fid, ref in parse_trans_file(trans.read_text(encoding="utf-8")):
            recs.append(Record(fid, trans.parent / f"{fid}.flac", ref))
    return recs
