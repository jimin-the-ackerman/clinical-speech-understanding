from pathlib import Path

import httpx

from stt_eval.transcribers.base import require_env

# OpenRouter proxies OpenAI-style multipart /audio/transcriptions (verified 2026-07-13,
# https://openrouter.ai/docs/guides/overview/multimodal/stt; the older chat-completions
# input_audio route 400s for this model). Same underlying model as
# transcribers/openai_api.py, different route + billing (OpenRouter credits) — kept as
# a separate registry entry so cached transcripts carry their provenance.
MODEL = "openai/gpt-4o-transcribe"
BASE = "https://openrouter.ai/api/v1"


class OpenRouterTranscribe:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = "gpt-4o-transcribe-openrouter"
        self._key = require_env("OPENROUTER_API_KEY")
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        name, payload = audio_path.name, audio_path.read_bytes()
        if len(payload) > 25 * 1024 * 1024:  # same 25 MB multipart cap as OpenAI
            from stt_eval.transcribers.openai_api import _to_flac

            name, payload = audio_path.stem + ".flac", _to_flac(audio_path)
        resp = self._client.post(
            f"{BASE}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._key}"},
            files={"file": (name, payload)},
            data={"model": MODEL, "language": "en"},
        )
        resp.raise_for_status()
        return resp.json()["text"]
