from stt_eval.normalize import normalize_en


def test_lowercases_and_strips_punctuation():
    assert normalize_en("Hello, WORLD!") == "hello world"


def test_expands_contractions_and_digitizes_numbers():
    assert normalize_en("I've got two apples.") == "i have got 2 apples"


def test_empty_is_empty():
    assert normalize_en("") == ""
