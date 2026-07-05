import httpx
import pytest

from stt_eval.transcribers.base import MissingKeyError


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF-fake-wav")
    return p


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_missing_key_raises(monkeypatch):
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        OpenAITranscribe()


def test_openai_transcribes(monkeypatch, wav):
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer sk-test"
        assert b"gpt-4o-transcribe" in request.content
        return httpx.Response(200, json={"text": "hello from openai"})

    t = OpenAITranscribe(client=_client(handler))
    assert t.transcribe(wav) == "hello from openai"


def test_openai_http_error_raises(monkeypatch, wav):
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    t = OpenAITranscribe(client=_client(lambda r: httpx.Response(500, text="oops")))
    with pytest.raises(httpx.HTTPStatusError):
        t.transcribe(wav)


def test_deepgram_transcribes(monkeypatch, wav):
    from stt_eval.transcribers.deepgram_api import Deepgram

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")

    def handler(request):
        assert request.url.path == "/v1/listen"
        assert request.url.params["model"] == "nova-3-medical"
        assert request.headers["authorization"] == "Token dg-test"
        assert request.headers["content-type"] == "audio/wav"
        return httpx.Response(200, json={
            "results": {"channels": [{"alternatives": [{"transcript": "hello from dg"}]}]}
        })

    t = Deepgram(client=_client(handler))
    assert t.transcribe(wav) == "hello from dg"


def test_deepgram_sends_flac_content_type(monkeypatch, tmp_path):
    from stt_eval.transcribers.deepgram_api import Deepgram

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    flac = tmp_path / "a.flac"
    flac.write_bytes(b"fLaC-fake")

    def handler(request):
        assert request.headers["content-type"] == "audio/flac"
        return httpx.Response(200, json={
            "results": {"channels": [{"alternatives": [{"transcript": "x"}]}]}
        })

    Deepgram(client=_client(handler)).transcribe(flac)
