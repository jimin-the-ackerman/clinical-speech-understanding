import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterator


def safe_id(file_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", file_id)
    if sanitized != file_id:  # disambiguate: distinct ids must never collide
        digest = hashlib.sha1(file_id.encode("utf-8")).hexdigest()[:8]
        sanitized = f"{sanitized}-{digest}"
    return sanitized


def cache_path(root: Path, dataset: str, model: str, file_id: str) -> Path:
    return root / "transcripts" / dataset / model / f"{safe_id(file_id)}.json"


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_results(root: Path) -> Iterator[dict]:
    for p in sorted((root / "transcripts").rglob("*.json")):
        yield json.loads(p.read_text(encoding="utf-8"))
