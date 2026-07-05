from pathlib import Path

import httpx

from stt_eval.transcribers.base import poll_until, require_env

BASE = "https://api.assemblyai.com/v2"
# PINNED 2026-07-05 per https://www.assemblyai.com/docs/api-reference/transcripts/submit:
# the singular `speech_model` field is deprecated in favor of `speech_models` (a
# priority-ordered list). There is no plain "universal-3" enum value; the enum is
# exactly ["universal-3-5-pro", "universal-2"], and "universal-3-5-pro" (branded
# "Universal-3.5 Pro") is the current top-tier/default model.
SPEECH_MODEL = "universal-3-5-pro"


class AssemblyAI:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = f"assemblyai-{SPEECH_MODEL}"
        self._headers = {"authorization": require_env("ASSEMBLYAI_API_KEY")}
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        up = self._client.post(f"{BASE}/upload", headers=self._headers,
                               content=audio_path.read_bytes())
        up.raise_for_status()
        job = self._client.post(
            f"{BASE}/transcript", headers=self._headers,
            json={"audio_url": up.json()["upload_url"],
                  "speech_models": [SPEECH_MODEL], "language_code": "en"},
        )
        job.raise_for_status()
        job_id = job.json()["id"]

        def fetch() -> dict:
            r = self._client.get(f"{BASE}/transcript/{job_id}", headers=self._headers)
            r.raise_for_status()
            return r.json()

        state = poll_until(fetch, lambda d: d["status"] in ("completed", "error"))
        if state["status"] == "error":
            raise RuntimeError(f"assemblyai job failed: {state.get('error')}")
        return state["text"]
