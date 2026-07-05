import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from stt_eval import store
from stt_eval.records import Record
from stt_eval.transcribers.base import Transcriber, with_retries


def transcribe_record(t: Transcriber, rec: Record, dataset: str, results_root: Path) -> str:
    path = store.cache_path(results_root, dataset, t.name, rec.file_id)
    if path.exists():
        return "cached"
    payload = {
        "model": t.name,
        "dataset": dataset,
        "file_id": rec.file_id,
        "condition": rec.condition,
        "reference": rec.reference,
        "audio_seconds": round(sf.info(str(rec.audio_path)).duration, 2),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    start = time.perf_counter()
    try:
        text = with_retries(lambda: t.transcribe(rec.audio_path))
        payload.update(failed=False, text=text)
    except Exception as e:  # failure is data here, not a crash
        payload.update(failed=True, text="", error=repr(e))
    payload["seconds"] = round(time.perf_counter() - start, 3)
    store.write_result(path, payload)
    return "failed" if payload["failed"] else "done"


def transcribe_dataset(
    t: Transcriber,
    records: list[Record],
    dataset: str,
    results_root: Path,
    workers: int = 8,
) -> dict[str, int]:
    n_workers = workers if getattr(t, "parallel_safe", True) else 1
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        statuses = list(ex.map(lambda r: transcribe_record(t, r, dataset, results_root), records))
    counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    print(f"  {t.name} on {dataset}: {counts}")
    return counts
