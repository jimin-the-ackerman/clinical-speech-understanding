from pathlib import Path

import httpx

from stt_eval.transcribers.base import require_env

# PIN AT EXECUTION: latest dated snapshot of gpt-4o-transcribe, per
# https://developers.openai.com/api/docs/guides/speech-to-text
MODEL = "gpt-4o-transcribe"


class OpenAITranscribe:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = MODEL
        self._key = require_env("OPENAI_API_KEY")
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        if audio_path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"{audio_path.name} exceeds OpenAI 25 MB upload limit; "
                             "transcode to FLAC or chunk it")
        with audio_path.open("rb") as f:
            resp = self._client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._key}"},
                files={"file": (audio_path.name, f)},
                data={"model": MODEL, "language": "en"},
            )
        resp.raise_for_status()
        return resp.json()["text"]
