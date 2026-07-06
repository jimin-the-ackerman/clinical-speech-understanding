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


def test_entity_build_accepts_model_workers_limit():
    args = build_parser().parse_args(
        ["entity-build", "--method", "openrouter", "--model", "anthropic/claude-opus-4.8",
         "--workers", "8", "--limit", "15"])
    assert args.method == "openrouter" and args.model == "anthropic/claude-opus-4.8"
    assert args.workers == 8 and args.limit == 15
