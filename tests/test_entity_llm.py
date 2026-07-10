import sys

import httpx
import pytest

from stt_eval.entity_llm import _parse_entity_list, _salvage_strings, openrouter_extractor


def test_import_entity_llm_pulls_no_heavy_deps():
    for mod in ("torch", "transformers", "bitsandbytes"):
        sys.modules.pop(mod, None)
    import importlib

    import stt_eval.entity_llm as m
    importlib.reload(m)
    assert not any(x in sys.modules for x in ("torch", "transformers", "bitsandbytes"))


def test_medgemma_extractor_is_registered():
    from stt_eval.entity_llm import medgemma_extractor
    assert callable(medgemma_extractor)  # loading the model needs GPU+HF token; smoke-tested e2e


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


def test_parse_salvages_truncated_array():
    # unterminated array (no closing ]) — the loop-truncation failure
    assert _parse_entity_list('```json\n[\n  "elbow",\n  "pain",\n  "arm"') == ["elbow", "pain", "arm"]


def test_parse_salvages_degenerate_repetition():
    # what greedy actually produced: a few real terms then a token loop, truncated
    # with no closing ]. Salvage the head; dedup collapses the loop tail.
    raw = '[\n  "elbow",\n  "pain",\n  "arm",\n  "arm",\n  "arm",\n  "arm'
    assert _parse_entity_list(raw) == ["elbow", "pain", "arm"]


def test_parse_salvage_handles_escaped_quotes():
    assert _parse_entity_list(r'["a \"quoted\" term", "next"') == ['a "quoted" term', "next"]


def test_salvage_needs_an_opened_array():
    # quoted words in prose that never opened an array must not be salvaged
    assert _salvage_strings('the term "cough" was mentioned') == []
