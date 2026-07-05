from stt_eval.datasets.meddialog_audio import (
    _condition_and_id,
    pick_subsample,
    read_id_to_text,
)


def test_subsample_is_deterministic_and_capped():
    ids = {"clean": [f"c{i}" for i in range(500)], "snr0": ["a", "b"]}
    first = pick_subsample(ids, per_condition=3, seed=42)
    second = pick_subsample(ids, per_condition=3, seed=42)
    assert first == second
    assert len(first["clean"]) == 3
    assert first["snr0"] == ["a", "b"]  # fewer than requested -> take all


def test_subsample_insensitive_to_input_order():
    a = pick_subsample({"c": ["x", "y", "z", "w"]}, 2, 7)
    b = pick_subsample({"c": ["w", "z", "y", "x"]}, 2, 7)
    assert a == b


def test_condition_and_id_parses_all_condition_suffixes():
    assert _condition_and_id("100096_1o00.wav") == ("clean", "100096_1")
    assert _condition_and_id("100096_2b20.wav") == ("background_noise_20", "100096_2")
    assert _condition_and_id("100096_1b40.wav") == ("background_noise_40", "100096_1")
    assert _condition_and_id("100096_1b60.wav") == ("background_noise_60", "100096_1")
    assert _condition_and_id("100096_1w02.wav") == ("white_noise_2", "100096_1")
    assert _condition_and_id("100096_1w06.wav") == ("white_noise_6", "100096_1")
    assert _condition_and_id("100096_2w10.wav") == ("white_noise_10", "100096_2")


def test_condition_and_id_strips_directories():
    assert _condition_and_id("noise-free audio/batch_1/100096_1o00.wav") == (
        "clean",
        "100096_1",
    )
    assert _condition_and_id("white_noise/noise_6%/batch_3/100193_2w06.wav") == (
        "white_noise_6",
        "100193_2",
    )


def test_condition_and_id_rejects_unknown_suffix():
    assert _condition_and_id("100096_1x99.wav") is None
    assert _condition_and_id("background_noise/noise_20%/batch_1/100096_1z20.wav") is None
    assert _condition_and_id("README.wav") is None  # too short / no known suffix


def test_condition_and_id_non_wav_keeps_suffixless_name_out():
    # non-wav names have no ".wav" to strip, so their last 3 chars are not a
    # known condition suffix -> None (prepare() also filters non-wav upfront)
    assert _condition_and_id("metadata.csv") is None
    assert _condition_and_id("README.md") is None


def test_read_id_to_text_joins_on_clean_filename():
    rows = [
        "filename,duration_s,transcription",
        '862398_2o00.wav,12.97,"It all depends on your efforts."',
        '326895_1o00.wav,46.25,"My daughter is four years old."',
    ]
    assert read_id_to_text(rows) == {
        "862398_2": "It all depends on your efforts.",
        "326895_1": "My daughter is four years old.",
    }


def test_read_id_to_text_handles_quoted_commas_and_empty():
    rows = [
        "filename,transcription",
        '111111_1o00.wav,"Hello, doctor, I have a question."',
    ]
    assert read_id_to_text(rows) == {"111111_1": "Hello, doctor, I have a question."}
    assert read_id_to_text(["filename,transcription"]) == {}
