import json
from pathlib import Path

from stt_eval.datasets import meddialog_audio
from stt_eval.datasets.meddialog_audio import (
    _condition_and_id,
    pick_subsample,
    read_id_to_text,
)


class _FakeHfApi:
    """Stand-in for huggingface_hub.HfApi: returns a fixed repo file listing."""

    def __init__(self, files):
        self._files = files

    def list_repo_files(self, repo, repo_type=None):
        return self._files


def _write_fake_download(path: Path, tag: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"content:{tag}".encode())
    return str(path)


def test_subsample_is_deterministic_and_capped():
    ids = {"clean": [f"c{i}" for i in range(500)], "snr0": ["a", "b"]}
    first = pick_subsample(ids, per_condition=3, seed=42)
    second = pick_subsample(ids, per_condition=3, seed=42)
    assert first == second
    assert len(first["clean"]) == 3
    assert first["snr0"] == ["a", "b"]  # fewer than requested -> take all


def test_subsample_insensitive_to_input_order():
    a = pick_subsample({"c": ["x", "y", "z", "w"]}, 2, 7)
    b = pick_subsample({"c": ["w", "z", "y", "x"]}, 2, 7)
    assert a == b


def test_condition_and_id_parses_all_condition_suffixes():
    assert _condition_and_id("100096_1o00.wav") == ("clean", "100096_1")
    assert _condition_and_id("100096_2b20.wav") == ("background_noise_20", "100096_2")
    assert _condition_and_id("100096_1b40.wav") == ("background_noise_40", "100096_1")
    assert _condition_and_id("100096_1b60.wav") == ("background_noise_60", "100096_1")
    assert _condition_and_id("100096_1w02.wav") == ("white_noise_2", "100096_1")
    assert _condition_and_id("100096_1w06.wav") == ("white_noise_6", "100096_1")
    assert _condition_and_id("100096_2w10.wav") == ("white_noise_10", "100096_2")


def test_condition_and_id_strips_directories():
    assert _condition_and_id("noise-free audio/batch_1/100096_1o00.wav") == (
        "clean",
        "100096_1",
    )
    assert _condition_and_id("white_noise/noise_6%/batch_3/100193_2w06.wav") == (
        "white_noise_6",
        "100193_2",
    )


def test_condition_and_id_rejects_unknown_suffix():
    assert _condition_and_id("100096_1x99.wav") is None
    assert _condition_and_id("background_noise/noise_20%/batch_1/100096_1z20.wav") is None
    assert _condition_and_id("README.wav") is None  # too short / no known suffix


def test_condition_and_id_non_wav_keeps_suffixless_name_out():
    # non-wav names have no ".wav" to strip, so their last 3 chars are not a
    # known condition suffix -> None (prepare() also filters non-wav upfront)
    assert _condition_and_id("metadata.csv") is None
    assert _condition_and_id("README.md") is None


def test_read_id_to_text_joins_on_clean_filename():
    rows = [
        "filename,duration_s,transcription",
        '862398_2o00.wav,12.97,"It all depends on your efforts."',
        '326895_1o00.wav,46.25,"My daughter is four years old."',
    ]
    assert read_id_to_text(rows) == {
        "862398_2": "It all depends on your efforts.",
        "326895_1": "My daughter is four years old.",
    }


def test_read_id_to_text_handles_quoted_commas_and_empty():
    rows = [
        "filename,transcription",
        '111111_1o00.wav,"Hello, doctor, I have a question."',
    ]
    assert read_id_to_text(rows) == {"111111_1": "Hello, doctor, I have a question."}
    assert read_id_to_text(["filename,transcription"]) == {}


def test_prepare_with_existing_manifest_downloads_only_listed_ids_and_never_rewrites_it(
    tmp_path, monkeypatch
):
    manifest_path = tmp_path / "manifest.json"
    entries = [
        {"file_id": "clean__100001_1", "condition": "clean", "text": "hello there"},
        {
            "file_id": "background_noise_20__100002_1",
            "condition": "background_noise_20",
            "text": "how are you",
        },
    ]
    manifest_path.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    original_bytes = manifest_path.read_bytes()
    monkeypatch.setattr(meddialog_audio, "MANIFEST", manifest_path)

    # HF listing includes extra files not in the manifest -- these must never be requested.
    files = [
        "batch_1/100001_1o00.wav",
        "batch_1/100002_1b20.wav",
        "batch_1/100003_1o00.wav",
        "batch_1/100001_1b20.wav",
        "metadata.csv",
    ]
    requested: list[str] = []
    hub_cache = tmp_path / "hub_cache"

    def fake_hf_hub_download(repo, path, repo_type=None):
        assert repo == meddialog_audio.HF_REPO
        assert path != meddialog_audio.METADATA_FILE, "manifest path must not re-fetch metadata"
        requested.append(path)
        return _write_fake_download(hub_cache / Path(path).name, path)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(files))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    data_dir = tmp_path / "data"
    meddialog_audio.prepare(data_dir)

    wavs = data_dir / "meddialog" / "wav"
    assert (wavs / "clean__100001_1.wav").exists()
    assert (wavs / "background_noise_20__100002_1.wav").exists()
    assert sorted(requested) == sorted(
        ["batch_1/100001_1o00.wav", "batch_1/100002_1b20.wav"]
    )
    assert manifest_path.read_bytes() == original_bytes  # never rewritten


def test_prepare_with_existing_manifest_skips_files_already_on_disk(tmp_path, monkeypatch):
    entries = [{"file_id": "clean__100001_1", "condition": "clean", "text": "hello there"}]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(meddialog_audio, "MANIFEST", manifest_path)

    data_dir = tmp_path / "data"
    wavs = data_dir / "meddialog" / "wav"
    wavs.mkdir(parents=True)
    (wavs / "clean__100001_1.wav").write_bytes(b"already here")

    def fail_HfApi():
        raise AssertionError("no network call expected when all wavs are already on disk")

    monkeypatch.setattr("huggingface_hub.HfApi", fail_HfApi)

    meddialog_audio.prepare(data_dir)  # must not raise / not touch the network

    assert (wavs / "clean__100001_1.wav").read_bytes() == b"already here"


def test_prepare_without_manifest_derives_subsample_and_writes_one(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(meddialog_audio, "MANIFEST", manifest_path)
    assert not manifest_path.exists()

    files = [
        "batch_1/100001_1o00.wav",
        "batch_1/100002_1o00.wav",
        "batch_1/100001_1b20.wav",
        "batch_1/100002_1b20.wav",
    ]
    metadata_src = tmp_path / "metadata.csv"
    metadata_src.write_text(
        "filename,transcription\n"
        '100001_1o00.wav,"hello there"\n'
        '100002_1o00.wav,"how are you"\n',
        encoding="utf-8",
    )

    requested: list[str] = []
    hub_cache = tmp_path / "hub_cache"

    def fake_hf_hub_download(repo, path, repo_type=None):
        requested.append(path)
        if path == meddialog_audio.METADATA_FILE:
            return str(metadata_src)
        return _write_fake_download(hub_cache / Path(path).name, path)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeHfApi(files))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    data_dir = tmp_path / "data"
    meddialog_audio.prepare(data_dir)

    assert manifest_path.exists()
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {e["condition"] for e in entries} == {"clean", "background_noise_20"}
    assert len(entries) == 4  # 2 ids x 2 conditions, all picked (fewer than PER_CONDITION)
    wavs = data_dir / "meddialog" / "wav"
    for e in entries:
        assert (wavs / f"{e['file_id']}.wav").exists()
    assert meddialog_audio.METADATA_FILE in requested
