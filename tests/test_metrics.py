from stt_eval.metrics import corpus_wer, cpwer, file_wer


def test_corpus_wer_pools_errors_over_words():
    # 1 substitution over 5 reference words = 0.2 (NOT the mean of per-file WERs)
    refs = ["a b c", "d e"]
    hyps = ["a x c", "d e"]
    assert corpus_wer(refs, hyps) == 0.2


def test_file_wer():
    assert file_wer("a b c", "a x c") == 1 / 3
    assert file_wer("a b", "a b") == 0.0


def test_cpwer():
    ref = {"doctor": "a b c", "patient": "x y"}
    # perfect, correctly attributed -> 0
    assert cpwer(ref, {"0": "a b c", "1": "x y"}) == 0.0
    # speaker labels swapped: permutation still finds the 0-error pairing
    assert cpwer(ref, {"0": "x y", "1": "a b c"}) == 0.0
    # single speaker collapses to plain corpus WER
    assert cpwer({"d": "a b c"}, {"0": "a x c"}) == corpus_wer(["a b c"], ["a x c"])
