from pathlib import Path

import httpx

from stt_eval.transcribers.base import poll_until, require_env

BASE = "https://api.soniox.com/v1"
# VERIFIED 2026-07-05 per https://soniox.com/docs/stt/async/async-transcription:
# POST /v1/files (multipart "file" -> {"id": ...}), POST /v1/transcriptions
# ({"file_id", "model": "stt-async-v5"} -> {"id", "status"}), GET
# /v1/transcriptions/{id} (status in {queued, processing?, completed, error};
# error detail in "error_message"), GET /v1/transcriptions/{id}/transcript
# (-> {"text": ..., "tokens": [...]})  Auth: `Authorization: Bearer <key>`.
# All matched the brief's original implementation; no code changes needed.
MODEL = "stt-async-v5"


class Soniox:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = f"soniox-{MODEL}"
        self._headers = {"Authorization": f"Bearer {require_env('SONIOX_API_KEY')}"}
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        with audio_path.open("rb") as f:
            up = self._client.post(f"{BASE}/files", headers=self._headers,
                                   files={"file": (audio_path.name, f)})
        up.raise_for_status()
        file_id = up.json()["id"]
        tx_id = None
        try:
            tx = self._client.post(f"{BASE}/transcriptions", headers=self._headers,
                                   json={"file_id": file_id, "model": MODEL})
            tx.raise_for_status()
            tx_id = tx.json()["id"]

            def fetch() -> dict:
                r = self._client.get(f"{BASE}/transcriptions/{tx_id}", headers=self._headers)
                r.raise_for_status()
                return r.json()

            state = poll_until(fetch, lambda d: d["status"] in ("completed", "error"))
            if state["status"] == "error":
                raise RuntimeError(f"soniox job failed: {state.get('error_message')}")
            r = self._client.get(f"{BASE}/transcriptions/{tx_id}/transcript", headers=self._headers)
            r.raise_for_status()
            return r.json()["text"]
        finally:
            # stored files/transcriptions count against account limits (uploads
            # start 429ing once full); the local JSON cache is our copy anyway
            for url in ([f"{BASE}/transcriptions/{tx_id}"] if tx_id else []) + [f"{BASE}/files/{file_id}"]:
                try:
                    self._client.delete(url, headers=self._headers)
                except httpx.HTTPError:
                    pass  # cleanup is best-effort; never mask the real result
