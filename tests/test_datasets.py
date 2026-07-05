import contextlib
import io
import tarfile

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


def _make_tar_gz(top_level: str, member: str, content: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=f"{top_level}/{member}")
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_prepare_extracts_atomically_and_cleans_up_tarball(tmp_path, monkeypatch):
    tar_bytes = _make_tar_gz(
        "LibriSpeech", "test-other/1/1/1-1-0000.trans.txt", b"1-1-0000 HELLO WORLD\n"
    )
    monkeypatch.setattr(
        librispeech.httpx, "stream",
        lambda *a, **k: _FakeStream([tar_bytes]),
    )
    librispeech.prepare(tmp_path)

    dest = tmp_path / "librispeech"
    assert (dest / "LibriSpeech" / "test-other" / "1" / "1" / "1-1-0000.trans.txt").exists()
    assert not (dest / "test-other.tar.gz").exists()  # tarball cleaned up on success
    assert not (dest / "extract.tmp").exists()  # temp extraction dir cleaned up


def test_interrupted_extraction_leaves_no_partial_dataset(tmp_path, monkeypatch):
    # a tarball with the wrong top-level directory simulates a broken/interrupted
    # archive: extraction "succeeds" but there's no LibriSpeech/ to promote into place
    tar_bytes = _make_tar_gz("not-librispeech", "file.txt", b"garbage\n")
    monkeypatch.setattr(
        librispeech.httpx, "stream",
        lambda *a, **k: _FakeStream([tar_bytes]),
    )
    with pytest.raises(FileNotFoundError):
        librispeech.prepare(tmp_path)

    dest = tmp_path / "librispeech"
    assert not (dest / "LibriSpeech" / "test-other").exists()
