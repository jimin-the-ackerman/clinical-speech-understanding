# LLM Entity-Identification Method Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **On execution, copy this plan to `docs/superpowers/plans/2026-07-07-llm-entity-method.md`** (repo convention; plan mode only allowed the scratch plan file).

**Goal:** Add a 4th entity-identification method to the medical-term-recall metric — an LLM extractor with two routes (local MedGemma, free; OpenRouter general model, keyed) — so we can test whether a *selective* clinical entity set ranks models with the NER methods (Soniox #1 on PriMock57) or the dictionary (qwen-1.7b #1).

**Architecture:** Two extractors share the existing `extract(reference:str) -> list[str]` seam, live in a new `entity_llm.py`, and are dispatched by `extractor_for()` / frozen by `build_manifest()`. `build_manifest` becomes resumable (per-reference cache mirroring the transcript cache in `store.py`) so a ~5,100-reference LLM run survives crashes and never re-bills. All heavy imports stay lazy so the offline test suite is untouched.

**Tech Stack:** Python, httpx (OpenRouter, core dep), transformers + torch + bitsandbytes (MedGemma, `local` extra + overlay), pytest with httpx.MockTransport.

## Global Constraints (verbatim from repo conventions)

- **Offline test suite stays green:** `uv run pytest` needs no network, GPU, or API keys. All `transformers`/`torch`/`bitsandbytes` imports are lazy (inside functions), as `_ner_union_extractor` does with `spacy`/`stanza`.
- **Entity seam:** every method is `extract(reference:str) -> list[str]` of surface forms, extracted from the **raw** reference (pre-normalization); scoring re-normalizes.
- **Reproducibility:** the frozen manifest is the reproducibility boundary; LLM calls use `temperature=0` / greedy decode.
- **Overlays, not env churn:** entity runs use ephemeral `uv run --with` / `--extra` overlays; torch/torchcodec are pinned to the cu126 index in `[tool.uv.sources]`.
- **No OpenAI model** as the LLM extractor (gpt-4o-transcribe is a scored ASR system).
- **Priority:** Tasks 1–2 + 4 are P0 (free/unblocked: parser, resumable build, MedGemma). Tasks 3, 5, 6 are P1 (OpenRouter needs `OPENROUTER_API_KEY`). Task 7 synthesis closes the exercise. OSCE transcription is **held** by the user (out of scope here).

---

### Task 1: JSON entity-list parser

**Files:**
- Create: `src/stt_eval/entity_llm.py`
- Test: `tests/test_entity_llm.py`

**Interfaces:**
- Consumes: `stt_eval.entity_score._dedupe_ci(items) -> list[str]` (case-insensitive dedupe, already used by `_ner_union_extractor`).
- Produces: `_parse_entity_list(raw:str) -> list[str]` — robust JSON-array parse; any failure returns `[]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entity_llm.py
from stt_eval.entity_llm import _parse_entity_list

def test_parse_bare_array():
    assert _parse_entity_list('["asthma", "amoxicillin"]') == ["asthma", "amoxicillin"]

def test_parse_fenced_json():
    assert _parse_entity_list('```json\n["asthma"]\n```') == ["asthma"]

def test_parse_dict_wrapped():
    assert _parse_entity_list('{"entities": ["asthma", "cough"]}') == ["asthma", "cough"]

def test_parse_array_with_prose():
    assert _parse_entity_list('Here are the terms:\n["chest pain"]\nDone.') == ["chest pain"]

def test_parse_dedupes_case_insensitively():
    assert _parse_entity_list('["Asthma", "asthma"]') == ["Asthma"]

def test_parse_refusal_or_garbage_returns_empty():
    assert _parse_entity_list("I cannot help with that.") == []
    assert _parse_entity_list("") == []
    assert _parse_entity_list("[not valid json") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: stt_eval.entity_llm`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/stt_eval/entity_llm.py
"""LLM entity extractors (method 4): MedGemma local + OpenRouter general model.
Both satisfy extract(reference)->[surface forms] and are dispatched by
entity_score.extractor_for. Heavy deps (transformers/torch/bitsandbytes) import
lazily; httpx is a core dep. See docs/entity-metric-comparison.md."""

import json
import re

from stt_eval.entity_score import _dedupe_ci

_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def _parse_entity_list(raw: str) -> list[str]:
    """Best-effort: strip fences/prose, pull a JSON array, coerce to a deduped
    list of non-empty strings. Any failure (refusal, empty, malformed) -> []."""
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    items = None
    try:
        obj = json.loads(text)
        items = obj if isinstance(obj, list) else next(
            (obj[k] for k in ("entities", "terms", "list") if isinstance(obj.get(k), list)), None)
    except Exception:
        m = _ARRAY.search(text)
        if m:
            try:
                items = json.loads(m.group(0))
            except Exception:
                items = None
    if not isinstance(items, list):
        return []
    return _dedupe_ci([s.strip() for s in items if isinstance(s, str) and s.strip()])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_llm.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/entity_llm.py tests/test_entity_llm.py
git commit -m "feat: robust JSON entity-list parser for LLM extractors"
```

---

### Task 2: Resumable / parallel `build_manifest`

**Files:**
- Modify: `src/stt_eval/entity_score.py:52-64` (`build_manifest`)
- Test: `tests/test_entity_score.py` (add cases)

**Interfaces:**
- Consumes: `store.read_results`, `store.write_result(path, payload)` (atomic tmp+os.replace), `store.safe_id(file_id)`.
- Produces: `build_manifest(results_root, extract, cache_dir=None, workers=1, limit=None) -> list[dict]`; `_unique_references(results_root, limit=None) -> list[tuple[str,str,str]]`. Default (`cache_dir=None`) path is byte-for-byte the current behavior.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entity_score.py — add
from stt_eval.entity_score import build_manifest  # already imported

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_score.py -k resumable -v`
Expected: FAIL — `build_manifest() got an unexpected keyword argument 'cache_dir'`.

- [ ] **Step 3: Write minimal implementation** (replace `build_manifest`, lines 52-64)

```python
from concurrent.futures import ThreadPoolExecutor  # add to imports


def _unique_references(results_root: Path, limit: int | None = None) -> list[tuple]:
    seen: dict[tuple, str] = {}
    for r in store.read_results(results_root):
        seen.setdefault((r["dataset"], r["file_id"]), r["reference"])
    keys = sorted(seen)
    if limit:
        keys = keys[:limit]
    return [(d, f, seen[(d, f)]) for d, f in keys]


def build_manifest(results_root, extract, cache_dir=None, workers=1, limit=None) -> list[dict]:
    """Run extract(reference)->[surface forms] over each unique (dataset,file_id).
    With cache_dir set, results are cached per reference (skip-if-exists) and
    extraction is parallelized — a crashed/expensive LLM run resumes and never
    re-bills. cache_dir=None is the original serial in-memory path."""
    refs = _unique_references(results_root, limit)
    if cache_dir is None:
        return [{"dataset": d, "file_id": f, "entities": extract(ref)} for d, f, ref in refs]

    def work(item):
        d, f, ref = item
        cp = Path(cache_dir) / d / f"{store.safe_id(f)}.json"
        if cp.exists():
            return json.loads(cp.read_text(encoding="utf-8"))
        try:
            entry = {"dataset": d, "file_id": f, "entities": extract(ref)}
            store.write_result(cp, entry)          # atomic; leaves failures uncached
            return entry
        except Exception as e:
            print(f"[entity-build] {d}/{f} failed: {e!r}")
            return {"dataset": d, "file_id": f, "entities": []}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        entries = list(ex.map(work, refs))
    return sorted(entries, key=lambda e: (e["dataset"], e["file_id"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_entity_score.py -v`
Expected: PASS (existing `test_build_manifest_dedups_by_file` still green via the `cache_dir=None` branch, plus 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/entity_score.py tests/test_entity_score.py
git commit -m "feat: resumable per-reference cache + parallel build_manifest"
```

---

### Task 3: OpenRouter extractor + `extractor_for` wiring

**Files:**
- Modify: `src/stt_eval/entity_llm.py` (add prompt + extractor)
- Modify: `src/stt_eval/entity_score.py:126-136` (`extractor_for` gains `model` param)
- Test: `tests/test_entity_llm.py`

**Interfaces:**
- Consumes: `stt_eval.transcribers.base.require_env`, `with_retries`; `_parse_entity_list`.
- Produces: `openrouter_extractor(model_id, client=None) -> callable` with `.parallel_safe=True`; `_SYSTEM`, `_user(reference)`; `extractor_for(method, results_root=None, model=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_entity_llm.py — add
import httpx, pytest
from stt_eval.entity_llm import openrouter_extractor

def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))

def test_openrouter_extracts_and_sends_correct_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    seen = {}
    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": '["asthma"]'}}]})
    ext = openrouter_extractor("anthropic/claude-opus-4.8", client=_client(handler))
    assert ext("patient has asthma") == ["asthma"]
    assert seen["path"] == "/api/v1/chat/completions"
    assert seen["auth"] == "Bearer or-test"
    assert b'"temperature": 0' in seen["body"] and b"anthropic/claude-opus-4.8" in seen["body"]
    assert ext.parallel_safe is True

def test_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from stt_eval.transcribers.base import MissingKeyError
    with pytest.raises(MissingKeyError):
        openrouter_extractor("anthropic/claude-opus-4.8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_llm.py -k openrouter -v`
Expected: FAIL — `cannot import name 'openrouter_extractor'`.

- [ ] **Step 3: Write minimal implementation** (append to `entity_llm.py`)

```python
_SYSTEM = (
    "You are a clinical NLP annotator. From a transcript of a medical consultation, "
    "extract the clinically important terms: medications and drugs, dosages, diagnoses, "
    "conditions and symptoms, procedures, tests and investigations, anatomy, and clinical "
    "findings. Do NOT include greetings, filler, or generic non-clinical words. Return each "
    "term using its EXACT surface form as it appears in the text — do not normalize, correct, "
    "expand abbreviations, or invent terms not present. Respond with ONLY a JSON array of strings."
)


def _user(reference: str) -> str:
    return f"Transcript:\n{reference}\n\nJSON array of clinical terms:"


def openrouter_extractor(model_id: str, client=None):
    import httpx

    from stt_eval.transcribers.base import require_env, with_retries

    key = require_env("OPENROUTER_API_KEY")
    client = client or httpx.Client(timeout=120)

    def extract(reference: str) -> list[str]:
        def call():
            r = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model_id, "temperature": 0,
                      "messages": [{"role": "system", "content": _SYSTEM},
                                   {"role": "user", "content": _user(reference)}]},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return _parse_entity_list(with_retries(call))

    extract.parallel_safe = True
    return extract
```

Then modify `entity_score.py` `extractor_for` (line 126) — add `model` param and the branch:

```python
def extractor_for(method, results_root=None, model=None):
    ...  # existing bc5cdr / ner-union / dictionary branches unchanged
    if method in ("llm", "openrouter"):
        from stt_eval.entity_llm import openrouter_extractor
        if not model:
            raise SystemExit("--model required (e.g. anthropic/claude-opus-4.8)")
        return openrouter_extractor(model)
    raise SystemExit(f"entity method {method!r} is not implemented yet")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/entity_llm.py src/stt_eval/entity_score.py tests/test_entity_llm.py
git commit -m "feat: OpenRouter LLM entity extractor + extractor_for wiring"
```

---

### Task 4: MedGemma extractor (local) + import-light guard

**Files:**
- Modify: `src/stt_eval/entity_llm.py` (add extractor)
- Modify: `src/stt_eval/entity_score.py` (`extractor_for` medgemma branch)
- Test: `tests/test_entity_llm.py`

**Interfaces:**
- Consumes: transformers `AutoModelForCausalLM`/`AutoTokenizer`/`BitsAndBytesConfig`, torch (all lazy).
- Produces: `medgemma_extractor(model_id="google/medgemma-27b-text-it") -> callable` with `.parallel_safe=False`.

- [ ] **Step 1: Write the failing test** (import-light guard — no GPU/model needed)

```python
# tests/test_entity_llm.py — add
import sys

def test_import_entity_llm_pulls_no_heavy_deps():
    for mod in ("torch", "transformers", "bitsandbytes"):
        sys.modules.pop(mod, None)
    import importlib, stt_eval.entity_llm as m
    importlib.reload(m)
    assert not any(x in sys.modules for x in ("torch", "transformers", "bitsandbytes"))

def test_medgemma_extractor_is_registered():
    from stt_eval.entity_llm import medgemma_extractor
    assert callable(medgemma_extractor)  # loading the model needs GPU+HF token; smoke-tested e2e
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_entity_llm.py -k "medgemma or heavy" -v`
Expected: FAIL — `cannot import name 'medgemma_extractor'`.

- [ ] **Step 3: Write minimal implementation** (append to `entity_llm.py`)

```python
def medgemma_extractor(model_id: str = "google/medgemma-27b-text-it"):
    """Local, medical-specialized. Gated HF repo: needs HF_TOKEN + license accept.
    27B loads in 4-bit to fit 24 GB; fall back to google/medgemma-4b-it bf16 on OOM
    (that variant is multimodal -> AutoModelForImageTextToText/AutoProcessor)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map="auto", dtype=torch.bfloat16)

    def extract(reference: str) -> list[str]:
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user(reference)}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt").to(model.device)
        out = model.generate(inputs, max_new_tokens=512, do_sample=False)
        text = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        return _parse_entity_list(text)

    extract.parallel_safe = False
    return extract
```

Then add the `extractor_for` branch (`entity_score.py`):

```python
    if method == "medgemma":
        from stt_eval.entity_llm import medgemma_extractor
        return medgemma_extractor(model or "google/medgemma-27b-text-it")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_entity_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/entity_llm.py src/stt_eval/entity_score.py tests/test_entity_llm.py
git commit -m "feat: MedGemma local entity extractor (4-bit) + import-light guard"
```

---

### Task 5: CLI wiring for `entity-build`

**Files:**
- Modify: `src/stt_eval/run.py` (parser lines ~20-22, dispatch lines ~67-75)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `extractor_for(method, results_root, model)`, `build_manifest(..., cache_dir, workers, limit)`, `store.safe_id`.
- Produces: `entity-build --method X [--model ID] [--workers N] [--limit N]`; LLM methods get a per-method `cache_dir` and a model-slugged manifest name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py — add
from stt_eval.run import build_parser

def test_entity_build_accepts_model_workers_limit():
    args = build_parser().parse_args(
        ["entity-build", "--method", "openrouter", "--model", "anthropic/claude-opus-4.8",
         "--workers", "8", "--limit", "15"])
    assert args.method == "openrouter" and args.model == "anthropic/claude-opus-4.8"
    assert args.workers == 8 and args.limit == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k entity_build -v`
Expected: FAIL — `unrecognized arguments: --model`.

- [ ] **Step 3: Write minimal implementation**

Parser (after the `entity-build` parser's existing `--method`/`--out`):

```python
    eb.add_argument("--model", default=None, help="model id for llm/openrouter/medgemma")
    eb.add_argument("--workers", type=int, default=8, help="parallel workers for API extractors")
    eb.add_argument("--limit", type=int, default=None, help="only first N unique references")
```

Dispatch (replace the `entity-build` branch body):

```python
    if args.cmd == "entity-build":
        from stt_eval import store
        from stt_eval.entity_score import build_manifest, extractor_for, write_manifest

        extract = extractor_for(args.method, args.results_dir, args.model)
        llm = args.method in {"medgemma", "llm", "openrouter"}
        cache_dir = args.results_dir / "entity_cache" / args.method if llm else None
        workers = args.workers if getattr(extract, "parallel_safe", False) else 1
        slug = args.method + (f"_{store.safe_id(args.model)}" if args.model else "")
        out = args.out or args.results_dir / "entity_manifests" / f"{slug}.json"
        entries = build_manifest(args.results_dir, extract, cache_dir=cache_dir,
                                 workers=workers, limit=args.limit)
        write_manifest(entries, out)
        print(f"wrote {len(entries)} files, "
              f"{sum(len(e['entities']) for e in entries)} entities to {out}")
        return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v` and `uv run pytest -q`
Expected: PASS (full suite green, ~80 tests).

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/run.py tests/test_cli.py
git commit -m "feat: entity-build --model/--workers/--limit + resumable-cache wiring"
```

---

### Task 6: Bake-off subcommand

**Files:**
- Modify: `src/stt_eval/entity_llm.py` (add `run_bakeoff`)
- Modify: `src/stt_eval/run.py` (`entity-bakeoff` parser + dispatch)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `entity_score._unique_references`, `extractor_for`.
- Produces: `entity-bakeoff --specs medgemma,openrouter:<id>,... --limit N` — prints each model's entities per reference; writes nothing.

- [ ] **Step 1: Write the failing test** (parser only; extraction needs models)

```python
# tests/test_cli.py — add
def test_entity_bakeoff_parses_specs():
    args = build_parser().parse_args(
        ["entity-bakeoff", "--specs", "medgemma,openrouter:google/gemini-2.5-flash", "--limit", "15"])
    assert args.specs == "medgemma,openrouter:google/gemini-2.5-flash" and args.limit == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k bakeoff -v`
Expected: FAIL — `invalid choice: 'entity-bakeoff'`.

- [ ] **Step 3: Write minimal implementation**

`entity_llm.py`:

```python
def run_bakeoff(results_root, specs: str, limit: int):
    """Print each spec's entity set per reference for a human pick. spec =
    "method" or "method:model_id"."""
    from stt_eval.entity_score import _unique_references, extractor_for
    refs = _unique_references(results_root, limit)
    extractors = []
    for spec in specs.split(","):
        method, _, model = spec.partition(":")
        extractors.append((spec, extractor_for(method, results_root, model or None)))
    for d, f, ref in refs:
        print(f"\n=== {d}/{f} ===\n{ref[:200]}")
        for name, ext in extractors:
            print(f"  [{name}] {ext(ref)}")
```

`run.py` parser + dispatch:

```python
    bk = sub.add_parser("entity-bakeoff", help="compare entity methods on N references")
    bk.add_argument("--specs", required=True, help="comma list of method[:model]")
    bk.add_argument("--limit", type=int, default=15)
```
```python
    if args.cmd == "entity-bakeoff":
        from stt_eval.entity_llm import run_bakeoff
        run_bakeoff(args.results_dir, args.specs, args.limit)
        return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/entity_llm.py src/stt_eval/run.py tests/test_cli.py
git commit -m "feat: entity-bakeoff subcommand for side-by-side model comparison"
```

---

### Task 7: Dependencies, bake-off run, full builds, synthesis

**Files:**
- Modify: `pyproject.toml` (comment noting the MedGemma overlay; optional `bitsandbytes` pin)
- Modify: `docs/entity-metric-comparison.md` (LLM results + verdict)
- Create: `results/entity_manifests/medgemma.json`, `results/entity_manifests/openrouter_<slug>.json`, `results/entity_recall_*.{csv,md}`

- [ ] **Step 1: Note the MedGemma overlay in pyproject** (no resolver change; `bitsandbytes` via overlay)

```toml
# entity LLM method: OpenRouter needs no new deps (httpx is core). MedGemma
# reuses the `local` extra + a bitsandbytes overlay for 4-bit:
#   uv run --extra local --with bitsandbytes --env-file .env stt-eval entity-build --method medgemma
# google/medgemma-27b-text-it is gated: needs HF_TOKEN + license acceptance.
```

- [ ] **Step 2: Bake-off to pick the OpenRouter model** (needs `OPENROUTER_API_KEY` + `HF_TOKEN` in `.env`)

Run: `uv run --extra local --with bitsandbytes --env-file .env stt-eval entity-bakeoff --specs "medgemma,openrouter:anthropic/claude-opus-4.8,openrouter:google/gemini-2.5-flash" --limit 15`
Expected: side-by-side entities on 15 clinical references; pick the OpenRouter model (quality vs the NER baselines).

- [ ] **Step 3: MedGemma smoke + full build** (resumable)

Run: `uv run --extra local --with bitsandbytes --env-file .env stt-eval entity-build --method medgemma --limit 5` (confirm 4-bit fits 24 GB, entities produced, re-run skips cached), then drop `--limit` for the full build.

- [ ] **Step 4: OpenRouter full build** (winning model)

Run: `uv run --env-file .env stt-eval entity-build --method openrouter --model <winner> --workers 8`
Expected: resumable; a stall never re-bills completed refs (per-ref cache under `results/entity_cache/openrouter/`).

- [ ] **Step 5: Score both + compare**

Run: `uv run stt-eval entity-score --manifest results/entity_manifests/medgemma.json` and same for the openrouter manifest. Then rank models on PriMock57 across all methods (reuse the comparison snippet in `docs/entity-metric-comparison.md`). Answer: does the LLM set side with the NER methods (Soniox #1) or the dictionary (qwen-1.7b #1)? Does medical-specialized (MedGemma) differ from general (OpenRouter)?

- [ ] **Step 6: Update docs + commit**

Update `docs/entity-metric-comparison.md` (method 4 status → done; the specialized-vs-general finding). Commit manifests, recall tables, and the doc:

```bash
git add results/entity_manifests/ results/entity_recall_*.{csv,md} pyproject.toml docs/entity-metric-comparison.md
git commit -m "results: LLM entity method (MedGemma + OpenRouter) + 4-method synthesis"
```

Note: `results/entity_cache/` (per-reference LLM cache) is scratch — add to `.gitignore`; the frozen manifests are the committed record.

---

## Self-Review

- **Coverage:** parser (T1), resumable build (T2), OpenRouter (T3), MedGemma (T4), CLI (T5), bake-off (T6), deps+run+synthesis (T7) — covers both routes, the cost-protection requirement, and the comparison deliverable. OSCE is intentionally out of scope (user held it).
- **Placeholders:** none — every code step has complete code; commands have expected output.
- **Type consistency:** `extract(reference)->list[str]` with `.parallel_safe` attr used identically in T3/T4/T5; `build_manifest(cache_dir, workers, limit)` signature consistent T2↔T5; `_parse_entity_list`/`_dedupe_ci` reused T1↔T3↔T4; `store.safe_id`/`store.write_result` reused T2↔T5.
- **Offline safety:** T1–T6 tests need no network/GPU/keys (MockTransport, stubs, import guard); heavy deps lazy-imported.

## Execution Handoff

Two execution options: **(1) Subagent-Driven** (fresh subagent per task, review between) or **(2) Inline** (batch with checkpoints). Tasks 1–6 are runnable now; Task 7 needs `OPENROUTER_API_KEY` + `HF_TOKEN` in `.env`.
