from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt_eval.records import Record


class FakeTranscriber:
    parallel_safe = True

    def __init__(self, text="hello world", fail_ids=()):
        self.name = "fake"
        self.text = text
        self.fail_ids = set(fail_ids)
        self.calls = []

    def transcribe(self, audio_path: Path) -> str:
        self.calls.append(audio_path)
        if audio_path.stem in self.fail_ids:
            raise RuntimeError("simulated failure")
        return self.text


@pytest.fixture
def fake_transcriber():
    return FakeTranscriber


@pytest.fixture
def tiny_records(tmp_path):
    """Two 0.1s silent wavs + Records."""
    recs = []
    for i in range(2):
        p = tmp_path / f"utt{i}.wav"
        sf.write(p, np.zeros(1600, dtype=np.float32), 16000)
        recs.append(Record(f"utt{i}", p, f"reference text {i}", condition=None))
    return recs
