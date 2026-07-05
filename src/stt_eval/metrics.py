import jiwer


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus-level WER: total edit errors / total reference words."""
    return jiwer.wer(refs, hyps)


def file_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(ref, hyp)
