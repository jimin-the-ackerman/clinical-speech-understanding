import json

from stt_eval import runner, store


def test_transcribe_writes_cache_with_schema(tmp_path, fake_transcriber, tiny_records):
    t = fake_transcriber()
    counts = runner.transcribe_dataset(t, tiny_records, "ds", tmp_path / "results", workers=2)
    assert counts == {"done": 2}
    row = json.loads(store.cache_path(tmp_path / "results", "ds", "fake", "utt0").read_text())
    assert row["model"] == "fake"
    assert row["dataset"] == "ds"
    assert row["text"] == "hello world"
    assert row["reference"] == "reference text 0"
    assert row["failed"] is False
    assert row["audio_seconds"] == 0.1
    assert row["seconds"] >= 0
    assert row["condition"] is None
    assert "created_at" in row


def test_rerun_skips_cached(tmp_path, fake_transcriber, tiny_records):
    root = tmp_path / "results"
    t = fake_transcriber()
    runner.transcribe_dataset(t, tiny_records, "ds", root)
    t2 = fake_transcriber(text="DIFFERENT")
    t2.name = "fake"
    counts = runner.transcribe_dataset(t2, tiny_records, "ds", root)
    assert counts == {"cached": 2}
    assert t2.calls == []


def test_failure_recorded_not_raised(tmp_path, fake_transcriber, tiny_records, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # skip retry backoff
    t = fake_transcriber(fail_ids={"utt1"})
    counts = runner.transcribe_dataset(t, tiny_records, "ds", tmp_path / "results")
    assert counts == {"done": 1, "failed": 1}
    row = json.loads(store.cache_path(tmp_path / "results", "ds", "fake", "utt1").read_text())
    assert row["failed"] is True
    assert "simulated failure" in row["error"]


def test_unreadable_audio_recorded_as_failure(tmp_path, fake_transcriber, monkeypatch):
    from pathlib import Path

    from stt_eval.records import Record

    monkeypatch.setattr("time.sleep", lambda s: None)
    missing = Record("ghost", Path(tmp_path / "nope.wav"), "some reference")
    t = fake_transcriber()
    counts = runner.transcribe_dataset(t, [missing], "ds", tmp_path / "results")
    assert counts == {"failed": 1}
    row = json.loads(store.cache_path(tmp_path / "results", "ds", "fake", "ghost").read_text())
    assert row["failed"] is True
    assert row["audio_seconds"] is None
    assert row["reference"] == "some reference"
