from pathlib import Path


class WhisperLocal:
    parallel_safe = False  # one GPU model instance; batching happens inside

    def __init__(self, model_id: str, language: str = "en"):
        from faster_whisper import WhisperModel  # local extra: uv sync --extra local

        self.name = f"whisper-{model_id}"
        self.language = language
        self._model = WhisperModel(model_id, device="auto", compute_type="auto")

    def transcribe(self, audio_path: Path) -> str:
        segments, _info = self._model.transcribe(str(audio_path), language=self.language)
        return " ".join(s.text.strip() for s in segments)
