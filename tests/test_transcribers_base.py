import sys

import pytest

import stt_eval.transcribers as tr
from stt_eval.transcribers.base import MissingKeyError, require_env, with_retries

EXPECTED_MODELS = {
    "whisper-large-v3", "whisper-large-v3-turbo",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b",
    "gpt-4o-transcribe", "gpt-4o-transcribe-openrouter", "deepgram-nova-3-medical",
    "assemblyai-universal-3-5-pro", "soniox-stt-async-v5",
}


def test_registry_keys_exact():
    assert set(tr.REGISTRY) == EXPECTED_MODELS


def test_create_unknown_model_raises_keyerror():
    with pytest.raises(KeyError, match="unknown model"):
        tr.create("nope")


def test_registry_import_is_lazy():
    assert "faster_whisper" not in sys.modules
    assert "transformers" not in sys.modules


def test_require_env(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        require_env("SOME_KEY")
    monkeypatch.setenv("SOME_KEY", "v")
    assert require_env("SOME_KEY") == "v"


def test_with_retries_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return "ok"

    assert with_retries(flaky) == "ok"
    assert len(calls) == 3


def test_with_retries_gives_up(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def always_fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with_retries(always_fails)


def test_with_retries_does_not_retry_missing_key():
    calls = []

    def missing():
        calls.append(1)
        raise MissingKeyError("no key")

    with pytest.raises(MissingKeyError):
        with_retries(missing)
    assert len(calls) == 1
