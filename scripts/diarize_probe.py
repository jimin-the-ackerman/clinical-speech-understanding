"""Throwaway speaker-attribution probe on PriMock57 (see
docs/superpowers/specs/2026-07-09-diarization-cpwer-probe.md).

Re-requests Soniox with diarization ON, recovers per-speaker reference from the
un-flattened TextGrids, and reports cpWER next to flat WER. Nothing here is wired
into `stt-eval`; numbers are NOT committed to findings/.

    SONIOX_API_KEY=... uv run python scripts/diarize_probe.py [--limit N]

Reports corpus (micro-averaged) WER and cpWER — pooled over all files, matching
`score.py` — so the numbers are directly comparable to the leaderboard's flat WER.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import httpx

from stt_eval.datasets.primock57 import merge_reference, speaker_reference
from stt_eval.metrics import corpus_wer, cpwer, cpwer_align
from stt_eval.normalize import normalize_en
from stt_eval.transcribers.base import poll_until, require_env

DATA_DIR = Path("data")
OUT_DIR = Path("results/diarize-probe")  # raw transcript dumps, committed for sharing (see spec)
SONIOX = "https://api.soniox.com/v1"


def _norm_speakers(by_speaker: dict[str, str]) -> dict[str, str]:
    # Same Whisper/Open-ASR normalizer as flat WER, so cpWER is comparable to *our own*
    # flat WER. NOTE: this forgives casing/number/contraction/spelling diffs, which
    # published CHiME-6/MeetEval cpWER does NOT. For a fair comparison to published
    # cpWER, drop normalize_en here (compare raw whitespace-tokenized words instead).
    return {spk: normalize_en(text) for spk, text in by_speaker.items()}


def soniox_diarize(audio: Path, client: httpx.Client) -> tuple[str, dict[str, str]]:
    """-> (flat_text, {speaker_id: text}). Same flow as soniox_api.py + the
    enable_speaker_diarization flag; each transcript token carries a `speaker`."""
    headers = {"Authorization": f"Bearer {require_env('SONIOX_API_KEY')}"}
    with audio.open("rb") as f:
        up = client.post(f"{SONIOX}/files", headers=headers, files={"file": (audio.name, f)})
    up.raise_for_status()
    file_id = up.json()["id"]
    tx_id = None
    try:
        tx = client.post(f"{SONIOX}/transcriptions", headers=headers,
                         json={"file_id": file_id, "model": "stt-async-v5",
                               "enable_speaker_diarization": True})
        tx.raise_for_status()
        tx_id = tx.json()["id"]

        def fetch() -> dict:
            r = client.get(f"{SONIOX}/transcriptions/{tx_id}", headers=headers)
            r.raise_for_status()
            return r.json()

        state = poll_until(fetch, lambda d: d["status"] in ("completed", "error"))
        if state["status"] == "error":
            raise RuntimeError(f"soniox job failed: {state.get('error_message')}")
        r = client.get(f"{SONIOX}/transcriptions/{tx_id}/transcript", headers=headers)
        r.raise_for_status()
        body = r.json()
        by_speaker: dict[str, str] = defaultdict(str)
        for tok in body.get("tokens", []):
            spk = tok.get("speaker")
            if spk is not None:  # skip endpoint/non-speech tokens
                by_speaker[str(spk)] += tok["text"]
        return body["text"], dict(by_speaker)
    finally:
        for url in ([f"{SONIOX}/transcriptions/{tx_id}"] if tx_id else []) + [f"{SONIOX}/files/{file_id}"]:
            try:
                client.delete(url, headers=headers)
            except httpx.HTTPError:
                pass


# gpt-4o-transcribe-diarize dropped: Soniox already ranks #1–2 on flat WER, gpt-4o
# was ~85% of the probe's API cost, and it over-segmented 2-party audio into 3
# speakers. Re-add via /v1/audio/transcriptions (response_format=diarized_json,
# chunking_strategy=auto, segments[].speaker) if a second backend is wanted later.
MODELS = {
    "soniox-stt-async-v5": soniox_diarize,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="probe only the first N consultations")
    args = ap.parse_args()

    repo = DATA_DIR / "primock57" / "repo"
    wavs = sorted((DATA_DIR / "primock57" / "mixed").glob("*.wav"))[: args.limit]
    if not wavs:
        raise SystemExit("no PriMock57 audio in data/primock57/mixed — run `stt-eval prepare primock57`")

    client = httpx.Client(timeout=600)
    rows = []
    for model, fn in MODELS.items():
        n = 0
        flat_refs, flat_hyps = [], []  # pooled for corpus (micro) flat WER
        cp_refs, cp_hyps = [], []  # pooled best-permutation strings for corpus cpWER
        for wav in wavs:
            doctor_tg = repo / "transcripts" / f"{wav.stem}_doctor.TextGrid"
            patient_tg = repo / "transcripts" / f"{wav.stem}_patient.TextGrid"
            ref = _norm_speakers(speaker_reference(doctor_tg, patient_tg))
            try:
                flat_text, hyp = fn(wav, client)
            except Exception as e:  # one bad file shouldn't sink the run
                print(f"  [skip] {model}/{wav.stem}: {type(e).__name__}: {e}")
                continue
            n += 1
            # flat ref must be TIME-ordered (merge_reference), matching the model's
            # time-ordered transcript — NOT speaker-grouped, which reorders every word.
            flat_refs.append(normalize_en(merge_reference(doctor_tg, patient_tg)))
            flat_hyps.append(normalize_en(flat_text))
            r, h = cpwer_align(ref, _norm_speakers(hyp))
            cp_refs += r
            cp_hyps += h
            file_flat = corpus_wer([flat_refs[-1]], [flat_hyps[-1]])
            file_cp = cpwer(ref, _norm_speakers(hyp))
            # dump RAW (un-normalized) text so any run can be re-scored later without
            # re-paying the API; diarization is nondeterministic, so each run's dump matters
            out = OUT_DIR / "primock57" / model / f"{wav.stem}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "model": model, "file": wav.stem,
                "flat_text": flat_text, "by_speaker": hyp,
                "metrics": {"flat_wer": round(file_flat, 4), "cpwer": round(file_cp, 4),
                            "hyp_speakers": len(hyp)},
            }, indent=2))
            print(f"  {model}/{wav.stem}: flat={file_flat:.3f} cpwer={file_cp:.3f} "
                  f"(hyp speakers={len(hyp)})")
        fw = corpus_wer(flat_refs, flat_hyps) if n else None
        cp = corpus_wer(cp_refs, cp_hyps) if n else None
        rows.append((model, n, fw, cp))

    print("\n| model | n | flat WER | cpWER |")
    print("| --- | --- | --- | --- |")
    for model, n, fw, cp in rows:
        fmt = lambda v: f"{v:.4f}" if v is not None else "-"  # noqa: E731
        print(f"| {model} | {n} | {fmt(fw)} | {fmt(cp)} |")


if __name__ == "__main__":
    main()
