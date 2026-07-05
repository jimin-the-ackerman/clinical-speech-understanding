from pathlib import Path

import httpx

from stt_eval.transcribers.base import require_env

# PIN AT EXECUTION: model + params per https://developers.deepgram.com/docs
MODEL = "nova-3-medical"
CONTENT_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}


class Deepgram:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = f"deepgram-{MODEL}"
        self._key = require_env("DEEPGRAM_API_KEY")
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        resp = self._client.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": MODEL, "smart_format": "true", "language": "en"},
            headers={
                "Authorization": f"Token {self._key}",
                "Content-Type": CONTENT_TYPES[audio_path.suffix.lower()],
            },
            content=audio_path.read_bytes(),
        )
        resp.raise_for_status()
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
