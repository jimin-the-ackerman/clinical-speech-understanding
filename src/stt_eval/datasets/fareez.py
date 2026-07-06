"""Fareez et al. 2022 simulated patient-physician OSCE interviews.

272 mock consultations (~51 h), one 16 kHz mono MP3 per interview paired with a
verbatim D:/P:-tagged transcript. Like PriMock57 we mix both speakers into one
reference (whole-interview audio only, no per-turn alignment). Long-form clinical
dialogue, cleaner audio than PriMock57's room-mic condition.

Paper: https://doi.org/10.1038/s41597-022-01423-1
Data (CC0): https://doi.org/10.6084/m9.figshare.c.5545842.v1
Needs soundfile/libsndfile >=1.1 for MP3 (project pins soundfile>=0.12).
"""

import hashlib
import re
import time
import zipfile
from pathlib import Path

import httpx

from stt_eval.records import Record

# figshare article 16550013 -> single Data.zip; presigned redirect expires fast
URL = "https://ndownloader.figshare.com/files/30598530"
MD5 = "9c79f2050dbdaf13fb1c2c5d38587d60"
_UA = "Mozilla/5.0 (X11; Linux x86_64) stt-eval/0.1"
_TURN = re.compile(r"^\s*[DP]:\s*", re.MULTILINE)


def parse_transcript(text: str) -> str:
    """Join D:/P: turns (and their unprefixed continuation lines) into one
    speaker-tag-stripped reference string."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    body = _TURN.sub(" ", text)  # drop the speaker tags, keep the words
    return re.sub(r"\s+", " ", body).strip()


def _download(url: str, dest: Path, attempts: int = 5) -> None:
    tmp = dest.with_suffix(".part")
    for i in range(attempts):
        try:
            with httpx.stream("GET", url, headers={"User-Agent": _UA},
                              follow_redirects=True, timeout=None) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            tmp.rename(dest)
            return
        except httpx.HTTPError:  # figshare 403-rate-limits bursts; back off and retry
            if i == attempts - 1:
                raise
            time.sleep(2 * 2**i)


def prepare(data_dir: Path) -> None:
    dest = data_dir / "fareez-interviews"
    if (dest / "Data" / "Audio Recordings").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "Data.zip"
    if not zip_path.exists():
        print(f"downloading {URL} (~1 GB)")
        _download(URL, zip_path)

    h = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != MD5:
        raise RuntimeError(f"Data.zip md5 {h.hexdigest()} != expected {MD5}; delete and re-run")

    # extract into a temp dir first so an interrupted unzip never leaves a
    # partial Data/ that passes the exists-check above
    extract_tmp = dest / "extract.tmp"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_tmp)
    (extract_tmp / "Data").rename(dest / "Data")
    extract_tmp.rmdir()
    zip_path.unlink()


def load(data_dir: Path) -> list[Record]:
    root = data_dir / "fareez-interviews" / "Data"
    recs = []
    for mp3 in sorted((root / "Audio Recordings").glob("*.mp3")):
        txt = root / "Clean Transcripts" / f"{mp3.stem}.txt"
        reference = parse_transcript(txt.read_text(encoding="utf-8", errors="replace"))
        recs.append(Record(mp3.stem, mp3, reference))  # condition=None, pooled like PriMock57
    return recs
