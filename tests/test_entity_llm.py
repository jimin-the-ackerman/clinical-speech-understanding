import httpx
import pytest

from stt_eval.entity_llm import _parse_entity_list, openrouter_extractor


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openrouter_extracts_and_sends_correct_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    seen = {}
    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": '["asthma"]'}}]})
    ext = openrouter_extractor("anthropic/claude-opus-4.8", client=_client(handler))
    assert ext("patient has asthma") == ["asthma"]
    assert seen["path"] == "/api/v1/chat/completions"
    assert seen["auth"] == "Bearer or-test"
    assert b'"temperature":0' in seen["body"] and b"anthropic/claude-opus-4.8" in seen["body"]
    assert ext.parallel_safe is True


def test_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from stt_eval.transcribers.base import MissingKeyError
    with pytest.raises(MissingKeyError):
        openrouter_extractor("anthropic/claude-opus-4.8")


def test_parse_bare_array():
    assert _parse_entity_list('["asthma", "amoxicillin"]') == ["asthma", "amoxicillin"]


def test_parse_fenced_json():
    assert _parse_entity_list('```json\n["asthma"]\n```') == ["asthma"]


def test_parse_dict_wrapped():
    assert _parse_entity_list('{"entities": ["asthma", "cough"]}') == ["asthma", "cough"]


def test_parse_array_with_prose():
    assert _parse_entity_list('Here are the terms:\n["chest pain"]\nDone.') == ["chest pain"]


def test_parse_dedupes_case_insensitively():
    assert _parse_entity_list('["Asthma", "asthma"]') == ["Asthma"]


def test_parse_refusal_or_garbage_returns_empty():
    assert _parse_entity_list("I cannot help with that.") == []
    assert _parse_entity_list("") == []
    assert _parse_entity_list("[not valid json") == []
