import itertools

import jiwer


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus-level WER: total edit errors / total reference words."""
    return jiwer.wer(refs, hyps)


def file_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(ref, hyp)


def cpwer_align(
    ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Pair hypothesis speakers to reference speakers by whichever permutation
    minimises WER, returning the aligned (refs, hyps) speaker lists. Pool these
    across files and call corpus_wer ONCE to micro-average cpWER (matching how
    score.py pools flat WER); call cpwer() for a single file's number. Inputs are
    one already-normalized string per speaker; speaker *keys* are ignored.
    ponytail: O(n!) over speakers; fine for clinical 2-speaker audio. If >~8
    speakers ever matter, swap the permutation search for Hungarian assignment.
    """
    refs = list(ref_by_speaker.values())
    hyps = list(hyp_by_speaker.values())
    n = max(len(refs), len(hyps))
    refs += [""] * (n - len(refs))  # pad so speaker counts match for the corpus
    hyps += [""] * (n - len(hyps))
    best = min(itertools.permutations(range(n)),
               key=lambda perm: corpus_wer(refs, [hyps[i] for i in perm]))
    return refs, [hyps[i] for i in best]


def cpwer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str]) -> float:
    """Concatenated minimum-permutation WER for one file. Reduces to corpus_wer
    for a single speaker."""
    return corpus_wer(*cpwer_align(ref_by_speaker, hyp_by_speaker))
