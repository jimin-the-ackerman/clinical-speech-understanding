"""Model registry. Factories import their backend lazily so an uninstalled
SDK or missing API key only affects the model that needs it."""

from typing import Callable

from stt_eval.transcribers.base import Transcriber


def _whisper(model_id: str) -> Callable[[], Transcriber]:
    def make() -> Transcriber:
        from stt_eval.transcribers.whisper_local import WhisperLocal

        return WhisperLocal(model_id)

    return make


def _qwen(repo: str) -> Callable[[], Transcriber]:
    def make() -> Transcriber:
        from stt_eval.transcribers.qwen3_asr import Qwen3ASR

        return Qwen3ASR(repo)

    return make


def _openai() -> Transcriber:
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    return OpenAITranscribe()


def _deepgram() -> Transcriber:
    from stt_eval.transcribers.deepgram_api import Deepgram

    return Deepgram()


def _assemblyai() -> Transcriber:
    from stt_eval.transcribers.assemblyai_api import AssemblyAI

    return AssemblyAI()


def _soniox() -> Transcriber:
    from stt_eval.transcribers.soniox_api import Soniox

    return Soniox()


REGISTRY: dict[str, Callable[[], Transcriber]] = {
    "whisper-large-v3": _whisper("large-v3"),
    "whisper-large-v3-turbo": _whisper("large-v3-turbo"),
    "qwen3-asr-0.6b": _qwen("Qwen/Qwen3-ASR-0.6B"),
    "qwen3-asr-1.7b": _qwen("Qwen/Qwen3-ASR-1.7B"),
    "gpt-4o-transcribe": _openai,
    "deepgram-nova-3-medical": _deepgram,
    "assemblyai-universal-3-5-pro": _assemblyai,
    "soniox-stt-async-v5": _soniox,
}


def create(name: str) -> Transcriber:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(sorted(REGISTRY))}")
    t = REGISTRY[name]()
    t.name = name
    return t
