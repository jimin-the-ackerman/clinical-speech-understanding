"""PriMock57: 57 mock GP consultations (Babylon Health).

Audio ships as separate doctor/patient channel recordings; we mix them to one
16 kHz mono wav per consultation (room-mic condition) and build the reference
by time-ordering both speakers' TextGrid utterances.
"""

import os
import re
import subprocess
from pathlib import Path

import textgrid

from stt_eval.records import Record

REPO = "https://github.com/babylonhealth/primock57.git"


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # non-speech markers like <clears_throat>
    return re.sub(r"\s+", " ", text).strip()


def merge_reference(doctor_tg: Path, patient_tg: Path) -> str:
    utts: list[tuple[float, str]] = []
    for path in (doctor_tg, patient_tg):
        tg = textgrid.TextGrid.fromFile(str(path))
        for tier in tg.tiers:
            for iv in tier:
                t = _clean(iv.mark or "")
                if t:
                    utts.append((iv.minTime, t))
    utts.sort(key=lambda x: x[0])
    return " ".join(t for _, t in utts)


def prepare(data_dir: Path) -> None:
    dest = data_dir / "primock57"
    repo = dest / "repo"
    if not repo.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", REPO, str(repo)], check=True)
    # always: heals interrupted clones/pulls; no-op once objects are fetched
    subprocess.run(["git", "lfs", "pull"], cwd=repo, check=True)
    mixed = dest / "mixed"
    mixed.mkdir(exist_ok=True)
    for doc_wav in sorted((repo / "audio").glob("*_doctor.wav")):
        stem = doc_wav.name.removesuffix("_doctor.wav")
        out = mixed / f"{stem}.wav"
        if out.exists():
            continue
        pat_wav = doc_wav.with_name(f"{stem}_patient.wav")
        tmp_out = mixed / f"{stem}.tmp.wav"  # ffmpeg needs a real audio extension
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(doc_wav), "-i", str(pat_wav),
             "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_out)],
            check=True,
        )
        os.replace(tmp_out, out)


def load(data_dir: Path) -> list[Record]:
    repo = data_dir / "primock57" / "repo"
    recs = []
    for wav in sorted((data_dir / "primock57" / "mixed").glob("*.wav")):
        ref = merge_reference(
            repo / "transcripts" / f"{wav.stem}_doctor.TextGrid",
            repo / "transcripts" / f"{wav.stem}_patient.TextGrid",
        )
        recs.append(Record(wav.stem, wav, ref))
    return recs
