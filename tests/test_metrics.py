from stt_eval.metrics import corpus_wer, file_wer


def test_corpus_wer_pools_errors_over_words():
    # 1 substitution over 5 reference words = 0.2 (NOT the mean of per-file WERs)
    refs = ["a b c", "d e"]
    hyps = ["a x c", "d e"]
    assert corpus_wer(refs, hyps) == 0.2


def test_file_wer():
    assert file_wer("a b c", "a x c") == 1 / 3
    assert file_wer("a b", "a b") == 0.0
