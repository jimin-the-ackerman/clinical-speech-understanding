from stt_eval import store
from stt_eval.score import score, write_outputs


def _put(root, dataset, model, file_id, **kw):
    payload = {
        "model": model, "dataset": dataset, "file_id": file_id,
        "condition": kw.get("condition"), "reference": kw.get("reference", "a b c"),
        "text": kw.get("text", "a b c"), "failed": kw.get("failed", False),
        "seconds": kw.get("seconds", 1.0), "audio_seconds": kw.get("audio_seconds", 10.0),
        "created_at": "2026-07-05T00:00:00+00:00",
    }
    if payload["failed"]:
        payload["error"] = "boom"
    store.write_result(store.cache_path(root, dataset, model, file_id), payload)


def test_score_groups_and_pools(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", reference="a b c", text="a x c")   # 1/3 errors
    _put(tmp_path, "ds", "m1", "f2", reference="d e", text="d e")       # 0/2
    summary, per_file = score(tmp_path)
    assert len(summary) == 1
    row = summary[0]
    assert (row["model"], row["dataset"], row["n_scored"]) == ("m1", "ds", 2)
    assert row["wer"] == 0.2  # pooled: 1 error / 5 words
    assert row["rtf"] == 0.1  # 2s processing / 20s audio
    assert [p["wer"] for p in per_file] == [round(1 / 3, 4), 0.0]


def test_failed_and_empty_refs_counted_not_scored(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", failed=True)
    _put(tmp_path, "ds", "m1", "f2", reference="...", text="x")  # normalizes to empty
    _put(tmp_path, "ds", "m1", "f3", reference="a b", text="a b")
    summary, _ = score(tmp_path)
    row = summary[0]
    assert row["n_failed"] == 1
    assert row["n_empty_ref"] == 1
    assert row["n_scored"] == 1
    assert row["wer"] == 0.0


def test_conditions_scored_separately(tmp_path):
    _put(tmp_path, "ds", "m1", "c0_f1", condition="snr0", reference="a b", text="x y")
    _put(tmp_path, "ds", "m1", "c1_f1", condition="clean", reference="a b", text="a b")
    summary, _ = score(tmp_path)
    by_cond = {r["condition"]: r["wer"] for r in summary}
    assert by_cond == {"snr0": 1.0, "clean": 0.0}


def test_write_outputs(tmp_path):
    _put(tmp_path, "ds", "m1", "f1")
    summary, per_file = score(tmp_path)
    write_outputs(summary, per_file, tmp_path)
    assert (tmp_path / "wer_summary.csv").exists()
    assert (tmp_path / "wer_per_file.csv").exists()
    md = (tmp_path / "wer_summary.md").read_text()
    assert "| model |" in md and "m1" in md


def test_markdown_escapes_pipes(tmp_path):
    _put(tmp_path, "ds", "m1", "f1", condition="snr|weird")
    summary, per_file = score(tmp_path)
    write_outputs(summary, per_file, tmp_path)
    md = (tmp_path / "wer_summary.md").read_text()
    assert "snr\\|weird" in md


def test_divergent_references_across_models_warns_but_scores_normally(tmp_path, capsys):
    _put(tmp_path, "ds", "m1", "f1", reference="a b c", text="a b c")
    _put(tmp_path, "ds", "m2", "f1", reference="a b x", text="a b c")  # same file, drifted ref
    summary, per_file = score(tmp_path)

    out = capsys.readouterr().out
    assert "divergent references" in out
    assert "ds/f1" in out

    # scoring itself is unaffected: each row is still scored against its own reference
    assert len(summary) == 2
    wer_by_model = {p["model"]: p["wer"] for p in per_file}
    assert wer_by_model == {"m1": 0.0, "m2": round(1 / 3, 4)}


def test_no_divergence_warning_when_references_agree(tmp_path, capsys):
    _put(tmp_path, "ds", "m1", "f1", reference="a b c", text="a b c")
    _put(tmp_path, "ds", "m2", "f1", reference="a b c", text="a x c")
    score(tmp_path)
    out = capsys.readouterr().out
    assert "divergent references" not in out
