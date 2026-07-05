import pytest

from stt_eval import datasets
from stt_eval.datasets.librispeech import parse_trans_file


def test_dataset_names():
    assert datasets.DATASETS == ("primock57", "meddialog-audio", "librispeech-test-other")


def test_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        datasets.load("nope", None)


def test_parse_trans_file():
    text = "8280-266249-0000 MARY HAD A LAMB\n8280-266249-0001 IT WAS WHITE\n\n"
    assert parse_trans_file(text) == [
        ("8280-266249-0000", "MARY HAD A LAMB"),
        ("8280-266249-0001", "IT WAS WHITE"),
    ]
