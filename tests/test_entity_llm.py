from stt_eval.entity_llm import _parse_entity_list


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
