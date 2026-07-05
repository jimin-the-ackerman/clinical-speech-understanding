import sys


def test_module_import_stays_light():
    import stt_eval.transcribers.whisper_local  # noqa: F401
    import stt_eval.transcribers.qwen3_asr  # noqa: F401

    assert "faster_whisper" not in sys.modules
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_classes_declare_not_parallel_safe():
    from stt_eval.transcribers.qwen3_asr import Qwen3ASR
    from stt_eval.transcribers.whisper_local import WhisperLocal

    assert WhisperLocal.parallel_safe is False
    assert Qwen3ASR.parallel_safe is False
