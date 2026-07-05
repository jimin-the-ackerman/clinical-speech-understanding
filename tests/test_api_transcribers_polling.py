import httpx
import pytest

from stt_eval.transcribers.base import poll_until


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF-fake-wav")
    return p


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_poll_until_polls_to_completion(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    states = iter([{"status": "queued"}, {"status": "processing"}, {"status": "completed"}])
    result = poll_until(lambda: next(states), lambda d: d["status"] == "completed")
    assert result["status"] == "completed"


def test_poll_until_times_out(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        poll_until(lambda: {"status": "processing"}, lambda d: False, timeout=500)


def test_assemblyai_upload_then_poll(monkeypatch, wav):
    from stt_eval.transcribers.assemblyai_api import AssemblyAI

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aa-test")
    monkeypatch.setattr("time.sleep", lambda s: None)
    polls = {"n": 0}

    def handler(request):
        assert request.headers["authorization"] == "aa-test"
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/aa/1"})
        if request.url.path == "/v2/transcript" and request.method == "POST":
            assert request.content and b"universal-3-5-pro" in request.content
            return httpx.Response(200, json={"id": "job1", "status": "queued"})
        assert request.url.path == "/v2/transcript/job1"
        polls["n"] += 1
        if polls["n"] < 2:
            return httpx.Response(200, json={"id": "job1", "status": "processing"})
        return httpx.Response(200, json={"id": "job1", "status": "completed", "text": "hi from aa"})

    t = AssemblyAI(client=_client(handler))
    assert t.transcribe(wav) == "hi from aa"


def test_assemblyai_error_status_raises(monkeypatch, wav):
    from stt_eval.transcribers.assemblyai_api import AssemblyAI

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aa-test")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request):
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/aa/1"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job1", "status": "queued"})
        return httpx.Response(200, json={"id": "job1", "status": "error", "error": "bad audio"})

    with pytest.raises(RuntimeError, match="bad audio"):
        AssemblyAI(client=_client(handler)).transcribe(wav)


def test_soniox_upload_then_poll(monkeypatch, wav):
    from stt_eval.transcribers.soniox_api import Soniox

    monkeypatch.setenv("SONIOX_API_KEY", "sx-test")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request):
        assert request.headers["authorization"] == "Bearer sx-test"
        if request.url.path == "/v1/files":
            return httpx.Response(201, json={"id": "file1"})
        if request.url.path == "/v1/transcriptions" and request.method == "POST":
            assert request.content and b"stt-async-v5" in request.content
            return httpx.Response(201, json={"id": "tx1", "status": "queued"})
        if request.url.path == "/v1/transcriptions/tx1":
            return httpx.Response(200, json={"id": "tx1", "status": "completed"})
        assert request.url.path == "/v1/transcriptions/tx1/transcript"
        return httpx.Response(200, json={"text": "hi from soniox"})

    t = Soniox(client=_client(handler))
    assert t.transcribe(wav) == "hi from soniox"
