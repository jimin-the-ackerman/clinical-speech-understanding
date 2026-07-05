from stt_eval.datasets.meddialog_audio import pick_subsample


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
