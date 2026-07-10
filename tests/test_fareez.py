from stt_eval.datasets.fareez import parse_transcript, speaker_reference


def test_merges_turns_strips_tags_joins_continuations():
    # continuation lines have no D:/P: prefix and must join the preceding turn
    text = "D: OK, um how old are you, Jen?\n\nP: Um 52.\nI think.\n\nD: Right.\n"
    assert parse_transcript(text) == "OK, um how old are you, Jen? Um 52. I think. Right."


def test_preserves_disfluencies_and_handles_crlf():
    # verbatim "um/uh" and repetitions stay; CRLF normalizes
    text = "P: Um I I think maybe 8 days ago.\r\nP: Uh, yeah.\r\n"
    assert parse_transcript(text) == "Um I I think maybe 8 days ago. Uh, yeah."


def test_empty_and_blank_lines():
    assert parse_transcript("\n\n  \n") == ""


def test_speaker_reference_buckets_by_speaker():
    # continuation lines follow their speaker; same cleaning as parse_transcript
    text = "D: OK, um how old are you, Jen?\n\nP: Um 52.\nI think.\n\nD: Right.\r\n"
    assert speaker_reference(text) == {
        "doctor": "OK, um how old are you, Jen? Right.",
        "patient": "Um 52. I think.",
    }


def test_speaker_reference_words_match_flat_reference():
    # cpWER and flat WER must score the same word multiset
    text = "D: OK, um how old are you, Jen?\n\nP: Um 52.\nI think.\n\nD: Right.\n"
    ref = speaker_reference(text)
    both = f"{ref['doctor']} {ref['patient']}".split()
    assert sorted(both) == sorted(parse_transcript(text).split())
