import pytest

from stt_eval.run import build_parser


def test_parser_subcommands():
    args = build_parser().parse_args(["transcribe", "--models", "a,b", "--datasets", "d"])
    assert args.cmd == "transcribe"
    assert args.models == "a,b"
    assert args.workers == 8


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
