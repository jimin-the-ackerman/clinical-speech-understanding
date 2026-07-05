import contextlib

import pytest

from stt_eval import datasets
from stt_eval.datasets import librispeech
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


class _FakeStream(contextlib.AbstractContextManager):
    def __init__(self, chunks, fail_after=None):
        self.chunks = chunks
        self.fail_after = fail_after

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_bytes(self):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise ConnectionError("dropped")
            yield c


def test_interrupted_download_leaves_no_tarball(tmp_path, monkeypatch):
    monkeypatch.setattr(
        librispeech.httpx, "stream",
        lambda *a, **k: _FakeStream([b"x", b"y"], fail_after=1),
    )
    with pytest.raises(ConnectionError):
        librispeech.prepare(tmp_path)
    dest = tmp_path / "librispeech"
    assert not (dest / "test-other.tar.gz").exists()  # retry will re-download
