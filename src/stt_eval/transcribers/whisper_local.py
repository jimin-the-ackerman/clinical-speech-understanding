from pathlib import Path


def _preload_cuda_libs():
    # ponytail: CT2 wheels don't bundle CUDA libs; dlopen the copies from
    # torch's nvidia-*-cu12 pip wheels so libcublas/libcudnn resolve without
    # LD_LIBRARY_PATH. No-op off Linux/CUDA-12 installs.
    import ctypes

    try:
        import nvidia.cublas.lib, nvidia.cudnn.lib
    except ImportError:
        return
    for mod in (nvidia.cublas.lib, nvidia.cudnn.lib):
        for so in sorted(Path(mod.__path__[0]).glob("*.so.*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class WhisperLocal:
    parallel_safe = False  # one GPU model instance; batching happens inside

    def __init__(self, model_id: str, language: str = "en"):
        _preload_cuda_libs()
        from faster_whisper import WhisperModel  # local extra: uv sync --extra local

        self.name = f"whisper-{model_id}"
        self.language = language
        self._model = WhisperModel(model_id, device="auto", compute_type="auto")

    def transcribe(self, audio_path: Path) -> str:
        segments, _info = self._model.transcribe(str(audio_path), language=self.language)
        return " ".join(s.text.strip() for s in segments)
