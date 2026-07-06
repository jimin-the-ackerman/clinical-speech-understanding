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
        name, payload = audio_path.name, audio_path.read_bytes()
        if len(payload) > 25 * 1024 * 1024:
            name, payload = audio_path.stem + ".flac", _to_flac(audio_path)
            if len(payload) > 25 * 1024 * 1024:
                raise ValueError(f"{audio_path.name} exceeds OpenAI's 25 MB limit even as FLAC; "
                                 "chunking not implemented")
        resp = self._client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._key}"},
            files={"file": (name, payload)},
            data={"model": MODEL, "language": "en"},
        )
        resp.raise_for_status()
        return resp.json()["text"]


def _to_flac(audio_path: Path) -> bytes:
    """Losslessly shrink oversized uploads (long wav consultations ~halve)."""
    import io

    import soundfile as sf

    data, sr = sf.read(str(audio_path))
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC")
    return buf.getvalue()
