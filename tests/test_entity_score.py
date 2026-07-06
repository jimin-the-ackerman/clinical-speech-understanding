from stt_eval import store
from stt_eval.entity_score import entity_hit, file_recall, score, write_outputs


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


def test_score_pools_recall_per_group(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma and amoxicillin", text="asthma and amoxicillin")
    _put(tmp_path, "ds", "m1", "f2", reference="Pneumonia here", text="unrelated words")  # 0/1
    summary = score(tmp_path, _drug_disease_stub)
    assert len(summary) == 1
    row = summary[0]
    assert (row["n_entities"], row["n_hits"]) == (3, 2)  # asthma+amoxicillin hit, pneumonia missed
    assert row["entity_recall"] == round(2 / 3, 4)


def test_score_skips_failed_and_entity_free_refs(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma", text="x", failed=True)  # excluded
    _put(tmp_path, "ds", "m1", "f2", reference="no clinical terms here", text="whatever")  # 0 entities
    _put(tmp_path, "ds", "m1", "f3", reference="Eczema", text="eczema")  # 1/1
    summary = score(tmp_path, _drug_disease_stub)
    assert len(summary) == 1
    assert (summary[0]["n_entities"], summary[0]["n_hits"]) == (1, 1)


def test_score_groups_by_condition(tmp_path):
    _put(tmp_path, "ds", "m1", "a", condition="clean", reference="Asthma", text="asthma")
    _put(tmp_path, "ds", "m1", "b", condition="noisy", reference="Asthma", text="nothing")
    summary = score(tmp_path, _drug_disease_stub)
    by_cond = {r["condition"]: r["entity_recall"] for r in summary}
    assert by_cond == {"clean": 1.0, "noisy": 0.0}


def test_write_outputs(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="Asthma", text="asthma")
    write_outputs(score(tmp_path, _drug_disease_stub), tmp_path)
    assert (tmp_path / "entity_recall.csv").exists()
    md = (tmp_path / "entity_recall.md").read_text()
    assert "| model |" in md and "m1" in md
