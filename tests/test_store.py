from pathlib import Path

from stt_eval import store
from stt_eval.records import Record


def test_record_defaults():
    r = Record("id1", Path("a.wav"), "hello")
    assert r.condition is None


def test_cache_path_sanitizes_file_id(tmp_path):
    p = store.cache_path(tmp_path, "ds", "model", "weird/id with spaces")
    assert p.parent == tmp_path / "transcripts" / "ds" / "model"
    assert p.name.startswith("weird_id_with_spaces-")
    assert p.suffix == ".json"
    # unchanged-safe ids get no hash suffix
    clean = store.cache_path(tmp_path, "ds", "model", "clean-id_1.x")
    assert clean.name == "clean-id_1.x.json"


def test_distinct_ids_never_collide(tmp_path):
    a = store.cache_path(tmp_path, "ds", "m", "a/b")
    b = store.cache_path(tmp_path, "ds", "m", "a_b")
    assert a != b


def test_write_then_read_roundtrip(tmp_path):
    p = store.cache_path(tmp_path, "ds", "m", "f1")
    store.write_result(p, {"file_id": "f1", "text": "안녕 hi"})
    rows = list(store.read_results(tmp_path))
    assert rows == [{"file_id": "f1", "text": "안녕 hi"}]
    assert not list(tmp_path.rglob("*.tmp"))
