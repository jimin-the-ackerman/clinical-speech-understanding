from stt_eval.datasets.fareez import parse_transcript


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
