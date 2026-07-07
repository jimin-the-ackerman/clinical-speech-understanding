from stt_eval import store
from stt_eval.entity_score import (
    build_manifest, entity_hit, file_recall,
    load_manifest, score, write_manifest, write_outputs,
)


def _put(root, dataset, model, file_id, reference, text, **kw):
    store.write_result(
        store.cache_path(root, dataset, model, file_id),
        {"model": model, "dataset": dataset, "file_id": file_id,
         "condition": kw.get("condition"), "reference": reference, "text": text,
         "failed": kw.get("failed", False), "created_at": "2026-07-06T00:00:00+00:00"},
    )


def test_entity_hit_token_level_and_normalized():
    hyp = "i will prescribe amoxicillin 500 mg for the pneumonia".split()
    assert entity_hit("Amoxicillin", hyp)          # case-normalized
    assert entity_hit("pneumonia", hyp)
    assert not entity_hit("penicillin", hyp)        # absent
    assert not entity_hit("art", "we discussed the heart".split())  # not a substring match


def test_entity_hit_multiword_must_be_contiguous():
    assert entity_hit("myocardial infarction", "acute myocardial infarction today".split())
    assert not entity_hit("myocardial infarction", "myocardial and then infarction".split())


def test_file_recall_counts_hits():
    hits, total = file_recall(["asthma", "salbutamol", "eczema"], "history of asthma, takes salbutamol")
    assert (hits, total) == (2, 3)


def _drug_disease_stub(ref):
    """Fake NER: returns whichever known terms appear in the raw reference."""
    known = ["asthma", "amoxicillin", "pneumonia", "eczema"]
    return [t for t in known if t in ref.lower()]


def test_build_manifest_dedups_by_file(tmp_path):
    # same (dataset, file_id) transcribed by two models -> one manifest entry
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma and amoxicillin", text="x")
    _put(tmp_path, "ds", "m2", "f1", reference="Asthma and amoxicillin", text="y")
    _put(tmp_path, "ds", "m1", "f2", reference="Eczema only", text="z")
    entries = build_manifest(tmp_path, _drug_disease_stub)
    assert len(entries) == 2
    by_id = {e["file_id"]: e["entities"] for e in entries}
    assert by_id["f1"] == ["asthma", "amoxicillin"]
    assert by_id["f2"] == ["eczema"]


def test_build_manifest_resumable_caches_and_skips(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="asthma here", text="x")
    _put(tmp_path, "ds", "m1", "f2", reference="cough here", text="y")
    calls = []
    def extract(ref):
        calls.append(ref)
        if "cough" in ref:
            raise RuntimeError("boom")   # not cached -> retried next run
        return ["asthma"]
    cache = tmp_path / "cache"
    entries = build_manifest(tmp_path, extract, cache_dir=cache, workers=1)
    by_id = {e["file_id"]: e["entities"] for e in entries}
    assert by_id == {"f1": ["asthma"], "f2": []}          # f2 failed -> []
    assert (cache / "ds" / "f1.json").exists()
    assert not (cache / "ds" / "f2.json").exists()          # failure uncached
    # resume: f1 skipped (cached), only f2 retried
    calls.clear()
    build_manifest(tmp_path, extract, cache_dir=cache, workers=1)
    assert calls == ["cough here"]


def test_build_manifest_limit(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="a", text="x")
    _put(tmp_path, "ds", "m1", "f2", reference="b", text="y")
    entries = build_manifest(tmp_path, lambda r: [r], limit=1)
    assert len(entries) == 1


def test_build_manifest_dataset_filter(tmp_path):
    _put(tmp_path, "primock57", "m1", "f1", reference="asthma", text="x")
    _put(tmp_path, "librispeech-test-other", "m1", "l1", reference="THE KING", text="y")
    entries = build_manifest(tmp_path, lambda r: [r], datasets={"primock57"})
    assert [e["file_id"] for e in entries] == ["f1"]  # librispeech skipped


def test_manifest_roundtrip(tmp_path):
    entries = [{"dataset": "ds", "file_id": "f1", "entities": ["asthma"]}]
    write_manifest(entries, tmp_path / "m.json")
    assert load_manifest(tmp_path / "m.json") == {("ds", "f1"): ["asthma"]}


def test_score_pools_recall_per_group(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma and amoxicillin", text="asthma and amoxicillin")
    _put(tmp_path, "ds", "m1", "f2", reference="Pneumonia here", text="unrelated words")
    ents = {("ds", "f1"): ["asthma", "amoxicillin"], ("ds", "f2"): ["pneumonia"]}
    summary = score(tmp_path, ents)
    assert len(summary) == 1
    row = summary[0]
    assert (row["n_entities"], row["n_hits"]) == (3, 2)  # pneumonia missed
    assert row["entity_recall"] == round(2 / 3, 4)


def test_score_skips_failed_and_files_absent_from_manifest(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma", text="x", failed=True)   # excluded
    _put(tmp_path, "ds", "m1", "f2", reference="no clinical terms", text="whatever")  # not in manifest
    _put(tmp_path, "ds", "m1", "f3", reference="Eczema", text="eczema")
    ents = {("ds", "f1"): ["asthma"], ("ds", "f3"): ["eczema"]}  # f2 absent
    summary = score(tmp_path, ents)
    assert len(summary) == 1
    assert (summary[0]["n_entities"], summary[0]["n_hits"]) == (1, 1)


def test_score_groups_by_condition(tmp_path):
    _put(tmp_path, "ds", "m1", "a", condition="clean", reference="Asthma", text="asthma")
    _put(tmp_path, "ds", "m1", "b", condition="noisy", reference="Asthma", text="nothing")
    ents = {("ds", "a"): ["asthma"], ("ds", "b"): ["asthma"]}
    by_cond = {r["condition"]: r["entity_recall"] for r in score(tmp_path, ents)}
    assert by_cond == {"clean": 1.0, "noisy": 0.0}


def test_write_outputs_names_by_method(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma", text="asthma")
    write_outputs(score(tmp_path, {("ds", "f1"): ["asthma"]}), tmp_path, "entity_recall_bc5cdr")
    assert (tmp_path / "entity_recall_bc5cdr.csv").exists()
    md = (tmp_path / "entity_recall_bc5cdr.md").read_text()
    assert "| model |" in md and "m1" in md
