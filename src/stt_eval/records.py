from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    file_id: str
    audio_path: Path
    reference: str
    condition: str | None = None
