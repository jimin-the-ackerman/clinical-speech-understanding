# STT Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reproducible harness that benchmarks 8 STT models (4 local, 4 API) on 3 English datasets (PriMock57, MedDialog-Audio, LibriSpeech test-other), producing WER tables per (model, dataset, condition).

**Architecture:** `src/stt_eval/` package with a `Transcriber` protocol (registry of lazy factories, one module per backend), dataset modules that each yield `Record(file_id, audio_path, reference, condition)`, per-file JSON transcript caches under `results/transcripts/` (committed), and a scorer that reads only the caches — references are embedded in the cache JSON so scoring needs no dataset access.

**Tech Stack:** Python ≥3.10, uv, jiwer, httpx (all 4 APIs called via plain REST), soundfile, textgrid, vendored openai/whisper English normalizer, faster-whisper + transformers (optional `local` extra), HF `datasets` (optional `data` extra), pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`

## Global Constraints

- Python `>=3.10`, uv-managed, src layout, hatchling build.
- Test suite: no network, no GPU, no API keys required. Real model runs are manual (Task 14).
- Model registry keys (exact): `whisper-large-v3`, `whisper-large-v3-turbo`, `qwen3-asr-0.6b`, `qwen3-asr-1.7b`, `gpt-4o-transcribe`, `deepgram-nova-3-medical`, `assemblyai-universal-3-5-pro`, `soniox-stt-async-v5`.
- Dataset registry keys (exact): `primock57`, `meddialog-audio`, `librispeech-test-other`.
- Env vars (exact): `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY`, `SONIOX_API_KEY`. Missing key ⇒ model skipped with a warning, never an error.
- `data/` is gitignored; `results/` (transcripts, manifests, summaries) is committed.
- Cache JSON schema (all writers/readers use exactly these keys): `model, dataset, file_id, condition, reference, text, failed, error?, seconds, audio_seconds, created_at`.
- Constants marked `# PIN AT EXECUTION` carry a best-guess value plus a docs URL; the implementer verifies the value against that URL before the task's commit and records any change in the commit message.
- `ffmpeg` and `git` must be on PATH for dataset prep (document in README).
- CLI: `stt-eval prepare|transcribe|score` (prepare is an addition to the spec's CLI section, agreed as the home for dataset downloads).

---

### Task 1: Package scaffold + CLI skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/stt_eval/__init__.py`
- Create: `src/stt_eval/run.py`
- Create: `tests/test_cli.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Produces: console script `stt-eval` with subcommands `prepare`, `transcribe`, `score` (stubs); `stt_eval.run.main()`.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "stt-eval"
version = "0.1.0"
description = "STT benchmark harness for clinical speech understanding"
requires-python = ">=3.10"
dependencies = [
    "jiwer>=3.0",
    "httpx>=0.27",
    "soundfile>=0.12",
    "textgrid>=1.6",
    "regex>=2024.4.16",
    "more-itertools>=10.0",
]

[project.optional-dependencies]
data = ["datasets[audio]>=3.0"]
# PIN AT EXECUTION: transformers floor = first release with native Qwen3-ASR
# support, per https://huggingface.co/Qwen/Qwen3-ASR-1.7B model card.
local = ["faster-whisper>=1.1", "transformers>=4.57", "torch>=2.4", "accelerate>=1.0"]

[project.scripts]
stt-eval = "stt_eval.run:main"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package and CLI skeleton**

`src/stt_eval/__init__.py` — empty file.

`src/stt_eval/run.py`:

```python
import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stt-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="download and preprocess datasets")
    prep.add_argument("--datasets", required=True, help="comma-separated dataset names")

    tr = sub.add_parser("transcribe", help="run models over datasets, caching per-file JSON")
    tr.add_argument("--models", required=True, help="comma-separated model names")
    tr.add_argument("--datasets", required=True, help="comma-separated dataset names")
    tr.add_argument("--workers", type=int, default=8, help="parallel workers for API models")
    tr.add_argument("--limit", type=int, default=None, help="only first N records per dataset")

    sub.add_parser("score", help="compute WER tables from cached transcripts")

    for q in (prep, tr):
        q.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare":
        raise SystemExit("prepare: not implemented yet")
    if args.cmd == "transcribe":
        raise SystemExit("transcribe: not implemented yet")
    if args.cmd == "score":
        raise SystemExit("score: not implemented yet")
```

Note: `--results-dir` is a top-level flag (`stt-eval --results-dir X score`).

- [ ] **Step 3: Write the test**

`tests/test_cli.py`:

```python
import pytest

from stt_eval.run import build_parser


def test_parser_subcommands():
    args = build_parser().parse_args(["transcribe", "--models", "a,b", "--datasets", "d"])
    assert args.cmd == "transcribe"
    assert args.models == "a,b"
    assert args.workers == 8


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
```

- [ ] **Step 4: Sync and run**

Run: `uv sync --group dev && uv run pytest -q`
Expected: 2 passed.

Run: `uv run stt-eval score; echo "exit=$?"`
Expected: prints `score: not implemented yet`, `exit=1`.

- [ ] **Step 5: Append to .gitignore and commit**

Append to `.gitignore`:

```gitignore
# stt-eval
data/
```

```bash
git add pyproject.toml uv.lock src tests .gitignore
git commit -m "feat: scaffold stt-eval package with CLI skeleton"
```

---

### Task 2: Vendored Whisper normalizer + normalize.py

**Files:**
- Create: `src/stt_eval/_whisper_norm/` (vendored: `basic.py`, `english.py`, `english.json`, `LICENSE`, `__init__.py`)
- Create: `src/stt_eval/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Produces: `stt_eval.normalize.normalize_en(text: str) -> str`

- [ ] **Step 1: Vendor the normalizer (pinned upstream, MIT-licensed)**

```bash
mkdir -p src/stt_eval/_whisper_norm
BASE=https://raw.githubusercontent.com/openai/whisper/v20240930/whisper
curl -fsSL $BASE/normalizers/basic.py    -o src/stt_eval/_whisper_norm/basic.py
curl -fsSL $BASE/normalizers/english.py  -o src/stt_eval/_whisper_norm/english.py
curl -fsSL $BASE/normalizers/english.json -o src/stt_eval/_whisper_norm/english.json
curl -fsSL https://raw.githubusercontent.com/openai/whisper/v20240930/LICENSE \
     -o src/stt_eval/_whisper_norm/LICENSE
printf 'from .english import EnglishTextNormalizer\n\n__all__ = ["EnglishTextNormalizer"]\n' \
     > src/stt_eval/_whisper_norm/__init__.py
```

If the `v20240930` tag path 404s, use tag `v20250625` (or the latest release tag) — the normalizer is stable across releases.

- [ ] **Step 2: Write normalize.py**

```python
"""English text normalization, vendored from openai/whisper (MIT).

Same convention as the HF Open ASR Leaderboard, so WER numbers are
comparable to published results.
"""

from functools import lru_cache

from stt_eval._whisper_norm import EnglishTextNormalizer


@lru_cache(maxsize=1)
def _normalizer() -> EnglishTextNormalizer:
    return EnglishTextNormalizer()


def normalize_en(text: str) -> str:
    return _normalizer()(text)
```

- [ ] **Step 3: Write the failing test**

`tests/test_normalize.py`:

```python
from stt_eval.normalize import normalize_en


def test_lowercases_and_strips_punctuation():
    assert normalize_en("Hello, WORLD!") == "hello world"


def test_expands_contractions_and_digitizes_numbers():
    assert normalize_en("I've got two apples.") == "i have got 2 apples"


def test_empty_is_empty():
    assert normalize_en("") == ""
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: PASS. The vendored normalizer is the ground truth: if an assertion
fails, print the actual output (`uv run python -c "from stt_eval.normalize import normalize_en; print(repr(normalize_en('I\\'ve got two apples.')))"`),
confirm it is a sane normalization, and update the expected string once.

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/_whisper_norm src/stt_eval/normalize.py tests/test_normalize.py
git commit -m "feat: add English text normalizer (vendored from openai/whisper, MIT)"
```

---

### Task 3: metrics.py

**Files:**
- Create: `src/stt_eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `corpus_wer(refs: list[str], hyps: list[str]) -> float`, `file_wer(ref: str, hyp: str) -> float`. Inputs are already-normalized strings; refs must be non-empty.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:

```python
from stt_eval.metrics import corpus_wer, file_wer


def test_corpus_wer_pools_errors_over_words():
    # 1 substitution over 5 reference words = 0.2 (NOT the mean of per-file WERs)
    refs = ["a b c", "d e"]
    hyps = ["a x c", "d e"]
    assert corpus_wer(refs, hyps) == 0.2


def test_file_wer():
    assert file_wer("a b c", "a x c") == 1 / 3
    assert file_wer("a b", "a b") == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stt_eval.metrics'`.

- [ ] **Step 3: Implement**

`src/stt_eval/metrics.py`:

```python
import jiwer


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus-level WER: total edit errors / total reference words."""
    return jiwer.wer(refs, hyps)


def file_wer(ref: str, hyp: str) -> float:
    return jiwer.wer(ref, hyp)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (jiwer pools errors when given lists — the first test guards exactly that).

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/metrics.py tests/test_metrics.py
git commit -m "feat: add corpus and per-file WER metrics"
```

---

### Task 4: records.py + store.py (cache I/O)

**Files:**
- Create: `src/stt_eval/records.py`
- Create: `src/stt_eval/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces: `Record` dataclass — `file_id: str, audio_path: Path, reference: str, condition: str | None = None`.
- Produces: `store.cache_path(root: Path, dataset: str, model: str, file_id: str) -> Path`; `store.write_result(path: Path, payload: dict) -> None` (atomic); `store.read_results(root: Path) -> Iterator[dict]` (walks `root/transcripts/**/*.json`).

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:

```python
from pathlib import Path

from stt_eval import store
from stt_eval.records import Record


def test_record_defaults():
    r = Record("id1", Path("a.wav"), "hello")
    assert r.condition is None


def test_cache_path_sanitizes_file_id(tmp_path):
    p = store.cache_path(tmp_path, "ds", "model", "weird/id with spaces")
    assert p == tmp_path / "transcripts" / "ds" / "model" / "weird_id_with_spaces.json"


def test_write_then_read_roundtrip(tmp_path):
    p = store.cache_path(tmp_path, "ds", "m", "f1")
    store.write_result(p, {"file_id": "f1", "text": "안녕 hi"})
    rows = list(store.read_results(tmp_path))
    assert rows == [{"file_id": "f1", "text": "안녕 hi"}]
    assert not list(tmp_path.rglob("*.tmp"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement**

`src/stt_eval/records.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    file_id: str
    audio_path: Path
    reference: str
    condition: str | None = None
```

`src/stt_eval/store.py`:

```python
import json
import os
import re
from pathlib import Path
from typing import Iterator


def safe_id(file_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", file_id)


def cache_path(root: Path, dataset: str, model: str, file_id: str) -> Path:
    return root / "transcripts" / dataset / model / f"{safe_id(file_id)}.json"


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_results(root: Path) -> Iterator[dict]:
    for p in sorted((root / "transcripts").rglob("*.json")):
        yield json.loads(p.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/records.py src/stt_eval/store.py tests/test_store.py
git commit -m "feat: add Record type and atomic transcript cache store"
```

---

### Task 5: Transcriber protocol + registry

**Files:**
- Create: `src/stt_eval/transcribers/__init__.py`
- Create: `src/stt_eval/transcribers/base.py`
- Test: `tests/test_transcribers_base.py`

**Interfaces:**
- Produces: `base.Transcriber` protocol — attrs `name: str`, `parallel_safe: bool`; method `transcribe(self, audio_path: Path) -> str`.
- Produces: `base.MissingKeyError(RuntimeError)`; `base.require_env(var: str) -> str`; `base.with_retries(fn, attempts=3, base_delay=2.0)`.
- Produces: `transcribers.REGISTRY: dict[str, Callable[[], Transcriber]]` (the 8 keys from Global Constraints); `transcribers.create(name: str) -> Transcriber` — raises `KeyError` for unknown names, sets `t.name = name`, propagates `MissingKeyError`.
- Note: the backend modules the factories import (`whisper_local`, `qwen3_asr`, `openai_api`, `deepgram_api`, `assemblyai_api`, `soniox_api`) are created in Tasks 11–13; imports are lazy, so the registry works before they exist.

- [ ] **Step 1: Write the failing test**

`tests/test_transcribers_base.py`:

```python
import sys

import pytest

import stt_eval.transcribers as tr
from stt_eval.transcribers.base import MissingKeyError, require_env, with_retries

EXPECTED_MODELS = {
    "whisper-large-v3", "whisper-large-v3-turbo",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b",
    "gpt-4o-transcribe", "deepgram-nova-3-medical",
    "assemblyai-universal-3-5-pro", "soniox-stt-async-v5",
}


def test_registry_keys_exact():
    assert set(tr.REGISTRY) == EXPECTED_MODELS


def test_create_unknown_model_raises_keyerror():
    with pytest.raises(KeyError, match="unknown model"):
        tr.create("nope")


def test_registry_import_is_lazy():
    assert "faster_whisper" not in sys.modules
    assert "transformers" not in sys.modules


def test_require_env(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        require_env("SOME_KEY")
    monkeypatch.setenv("SOME_KEY", "v")
    assert require_env("SOME_KEY") == "v"


def test_with_retries_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("boom")
        return "ok"

    assert with_retries(flaky) == "ok"
    assert len(calls) == 3


def test_with_retries_gives_up(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    def always_fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        with_retries(always_fails)


def test_with_retries_does_not_retry_missing_key():
    calls = []

    def missing():
        calls.append(1)
        raise MissingKeyError("no key")

    with pytest.raises(MissingKeyError):
        with_retries(missing)
    assert len(calls) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_transcribers_base.py -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement**

`src/stt_eval/transcribers/base.py`:

```python
import os
import time
from pathlib import Path
from typing import Callable, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class MissingKeyError(RuntimeError):
    """A required API key env var is not set; the model should be skipped."""


@runtime_checkable
class Transcriber(Protocol):
    name: str
    parallel_safe: bool

    def transcribe(self, audio_path: Path) -> str: ...


def require_env(var: str) -> str:
    val = os.environ.get(var, "").strip()
    if not val:
        raise MissingKeyError(f"set {var} to use this model")
    return val


def with_retries(fn: Callable[[], T], attempts: int = 3, base_delay: float = 2.0) -> T:
    for i in range(attempts):
        try:
            return fn()
        except MissingKeyError:
            raise
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i)
    raise AssertionError("unreachable")
```

`src/stt_eval/transcribers/__init__.py`:

```python
"""Model registry. Factories import their backend lazily so an uninstalled
SDK or missing API key only affects the model that needs it."""

from typing import Callable

from stt_eval.transcribers.base import Transcriber


def _whisper(model_id: str) -> Callable[[], Transcriber]:
    def make() -> Transcriber:
        from stt_eval.transcribers.whisper_local import WhisperLocal

        return WhisperLocal(model_id)

    return make


def _qwen(repo: str) -> Callable[[], Transcriber]:
    def make() -> Transcriber:
        from stt_eval.transcribers.qwen3_asr import Qwen3ASR

        return Qwen3ASR(repo)

    return make


def _openai() -> Transcriber:
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    return OpenAITranscribe()


def _deepgram() -> Transcriber:
    from stt_eval.transcribers.deepgram_api import Deepgram

    return Deepgram()


def _assemblyai() -> Transcriber:
    from stt_eval.transcribers.assemblyai_api import AssemblyAI

    return AssemblyAI()


def _soniox() -> Transcriber:
    from stt_eval.transcribers.soniox_api import Soniox

    return Soniox()


REGISTRY: dict[str, Callable[[], Transcriber]] = {
    "whisper-large-v3": _whisper("large-v3"),
    "whisper-large-v3-turbo": _whisper("large-v3-turbo"),
    "qwen3-asr-0.6b": _qwen("Qwen/Qwen3-ASR-0.6B"),
    "qwen3-asr-1.7b": _qwen("Qwen/Qwen3-ASR-1.7B"),
    "gpt-4o-transcribe": _openai,
    "deepgram-nova-3-medical": _deepgram,
    "assemblyai-universal-3-5-pro": _assemblyai,
    "soniox-stt-async-v5": _soniox,
}


def create(name: str) -> Transcriber:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(sorted(REGISTRY))}")
    t = REGISTRY[name]()
    t.name = name
    return t
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_transcribers_base.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/transcribers tests/test_transcribers_base.py
git commit -m "feat: add Transcriber protocol, retry helper, and lazy model registry"
```

---

### Task 6: Runner + `stt-eval transcribe`

**Files:**
- Create: `src/stt_eval/runner.py`
- Modify: `src/stt_eval/run.py` (wire `transcribe`)
- Test: `tests/test_runner.py`
- Test helper: `tests/conftest.py`

**Interfaces:**
- Consumes: `store.cache_path/write_result`, `Record`, `base.with_retries`, `transcribers.create`.
- Produces: `runner.transcribe_record(t: Transcriber, rec: Record, dataset: str, results_root: Path) -> str` returning `"cached" | "done" | "failed"`; `runner.transcribe_dataset(t, records, dataset, results_root, workers=8) -> dict[str, int]`.
- Produces: working `stt-eval transcribe` that skips models raising `MissingKeyError` with a `[skip]` warning. Dataset loading is delegated to `stt_eval.datasets.load(name, data_dir)` (Task 8); until Task 8 lands, `transcribe` exits with "unknown dataset".

- [ ] **Step 1: Write the fake transcriber fixture**

`tests/conftest.py`:

```python
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
```

Add `numpy` to dev dependencies (soundfile needs it to write arrays):
in `pyproject.toml` `[dependency-groups]`, change to `dev = ["pytest>=8.0", "numpy>=1.26"]`, then `uv sync --group dev`.

- [ ] **Step 2: Write the failing test**

`tests/test_runner.py`:

```python
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
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL — `stt_eval.runner` doesn't exist.

- [ ] **Step 4: Implement runner.py**

```python
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from stt_eval import store
from stt_eval.records import Record
from stt_eval.transcribers.base import Transcriber, with_retries


def transcribe_record(t: Transcriber, rec: Record, dataset: str, results_root: Path) -> str:
    path = store.cache_path(results_root, dataset, t.name, rec.file_id)
    if path.exists():
        return "cached"
    payload = {
        "model": t.name,
        "dataset": dataset,
        "file_id": rec.file_id,
        "condition": rec.condition,
        "reference": rec.reference,
        "audio_seconds": round(sf.info(str(rec.audio_path)).duration, 2),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    start = time.perf_counter()
    try:
        text = with_retries(lambda: t.transcribe(rec.audio_path))
        payload.update(failed=False, text=text)
    except Exception as e:  # failure is data here, not a crash
        payload.update(failed=True, text="", error=repr(e))
    payload["seconds"] = round(time.perf_counter() - start, 3)
    store.write_result(path, payload)
    return "failed" if payload["failed"] else "done"


def transcribe_dataset(
    t: Transcriber,
    records: list[Record],
    dataset: str,
    results_root: Path,
    workers: int = 8,
) -> dict[str, int]:
    n_workers = workers if getattr(t, "parallel_safe", True) else 1
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        statuses = list(ex.map(lambda r: transcribe_record(t, r, dataset, results_root), records))
    counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    print(f"  {t.name} on {dataset}: {counts}")
    return counts
```

- [ ] **Step 5: Wire the CLI**

In `src/stt_eval/run.py`, replace the `transcribe` branch of `main()`:

```python
    if args.cmd == "transcribe":
        from stt_eval import datasets, runner, transcribers
        from stt_eval.transcribers.base import MissingKeyError

        for ds_name in args.datasets.split(","):
            records = datasets.load(ds_name, args.data_dir)
            if args.limit:
                records = records[: args.limit]
            print(f"{ds_name}: {len(records)} records")
            for model_name in args.models.split(","):
                try:
                    t = transcribers.create(model_name)
                except MissingKeyError as e:
                    print(f"[skip] {model_name}: {e}")
                    continue
                runner.transcribe_dataset(t, records, ds_name, args.results_dir, args.workers)
        return
```

(`stt_eval.datasets` arrives in Task 8; until then this import fails at runtime, which is fine — tests exercise `runner` directly.)

- [ ] **Step 6: Run all tests and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add pyproject.toml uv.lock src/stt_eval/runner.py src/stt_eval/run.py tests/conftest.py tests/test_runner.py
git commit -m "feat: add caching transcribe runner with parallel workers and failure records"
```

---

### Task 7: Scorer + `stt-eval score`

**Files:**
- Create: `src/stt_eval/score.py`
- Modify: `src/stt_eval/run.py` (wire `score`)
- Test: `tests/test_score.py`

**Interfaces:**
- Consumes: `store.read_results`, `normalize_en`, `corpus_wer`, `file_wer`.
- Produces: `score.score(results_root: Path) -> tuple[list[dict], list[dict]]` — `(summary_rows, per_file_rows)`. Summary row keys: `model, dataset, condition, n_scored, n_failed, n_empty_ref, wer, rtf`. Per-file row keys: `model, dataset, condition, file_id, wer, seconds, audio_seconds`.
- Produces: `score.write_outputs(summary, per_file, results_root)` → `results/wer_summary.csv`, `results/wer_per_file.csv`, `results/wer_summary.md`.

- [ ] **Step 1: Write the failing test**

`tests/test_score.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_score.py -v`
Expected: FAIL — `stt_eval.score` doesn't exist.

- [ ] **Step 3: Implement score.py**

```python
import csv
from collections import defaultdict
from pathlib import Path

from stt_eval import store
from stt_eval.metrics import corpus_wer, file_wer
from stt_eval.normalize import normalize_en


def score(results_root: Path) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in store.read_results(results_root):
        groups[(r["model"], r["dataset"], r.get("condition") or "")].append(r)

    summary, per_file = [], []
    for (model, dataset, condition), items in sorted(groups.items()):
        n_failed = n_empty = 0
        refs, hyps, secs, audio_secs = [], [], 0.0, 0.0
        for r in sorted(items, key=lambda x: x["file_id"]):
            if r.get("failed"):
                n_failed += 1
                continue
            ref, hyp = normalize_en(r["reference"]), normalize_en(r["text"])
            if not ref.strip():
                n_empty += 1
                continue
            refs.append(ref)
            hyps.append(hyp)
            secs += r.get("seconds") or 0.0
            audio_secs += r.get("audio_seconds") or 0.0
            per_file.append({
                "model": model, "dataset": dataset, "condition": condition,
                "file_id": r["file_id"], "wer": round(file_wer(ref, hyp), 4),
                "seconds": r.get("seconds"), "audio_seconds": r.get("audio_seconds"),
            })
        summary.append({
            "model": model, "dataset": dataset, "condition": condition,
            "n_scored": len(refs), "n_failed": n_failed, "n_empty_ref": n_empty,
            "wer": round(corpus_wer(refs, hyps), 4) if refs else None,
            "rtf": round(secs / audio_secs, 4) if audio_secs else None,
        })
        if n_failed or n_empty:
            print(f"[warn] {model}/{dataset}/{condition or '-'}: "
                  f"{n_failed} failed, {n_empty} empty-reference files excluded")
    return summary, per_file


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(summary: list[dict], per_file: list[dict], results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    _write_csv(results_root / "wer_summary.csv", summary)
    _write_csv(results_root / "wer_per_file.csv", per_file)
    if summary:
        cols = list(summary[0])
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in summary]
        (results_root / "wer_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Wire the CLI**

In `src/stt_eval/run.py`, replace the `score` branch:

```python
    if args.cmd == "score":
        from stt_eval.score import score, write_outputs

        summary, per_file = score(args.results_dir)
        write_outputs(summary, per_file, args.results_dir)
        for row in summary:
            print(row)
        return
```

- [ ] **Step 5: Run all tests, then end-to-end with the fake**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/stt_eval/score.py src/stt_eval/run.py tests/test_score.py
git commit -m "feat: add WER scorer with per-condition grouping and CSV/markdown outputs"
```

---

### Task 8: Dataset registry + LibriSpeech test-other + `stt-eval prepare`

**Files:**
- Create: `src/stt_eval/datasets/__init__.py`
- Create: `src/stt_eval/datasets/librispeech.py`
- Modify: `src/stt_eval/run.py` (wire `prepare`)
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces: `datasets.DATASETS` tuple of the 3 names; `datasets.prepare(name: str, data_dir: Path) -> None`; `datasets.load(name: str, data_dir: Path) -> list[Record]` — raises `KeyError("unknown dataset ...")` for bad names. Each dataset module exposes `prepare(data_dir)` (idempotent download+preprocess) and `load(data_dir) -> list[Record]`.
- Produces: `librispeech.parse_trans_file(text: str) -> list[tuple[str, str]]`.

- [ ] **Step 1: Write the failing test**

`tests/test_datasets.py`:

```python
import pytest

from stt_eval import datasets
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_datasets.py -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement**

`src/stt_eval/datasets/__init__.py`:

```python
import importlib
from pathlib import Path

from stt_eval.records import Record

DATASETS = ("primock57", "meddialog-audio", "librispeech-test-other")

_MODULES = {
    "primock57": "primock57",
    "meddialog-audio": "meddialog_audio",
    "librispeech-test-other": "librispeech",
}


def _module(name: str):
    if name not in _MODULES:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    return importlib.import_module(f"stt_eval.datasets.{_MODULES[name]}")


def prepare(name: str, data_dir: Path) -> None:
    _module(name).prepare(data_dir)


def load(name: str, data_dir: Path) -> list[Record]:
    return _module(name).load(data_dir)
```

`src/stt_eval/datasets/librispeech.py`:

```python
"""LibriSpeech test-other, downloaded directly from OpenSLR (no HF dependency).

Audio stays as the original 16 kHz mono FLAC — every backend we use accepts FLAC.
"""

import tarfile
from pathlib import Path

import httpx

from stt_eval.records import Record

URL = "https://www.openslr.org/resources/12/test-other.tar.gz"


def parse_trans_file(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            fid, _, ref = line.partition(" ")
            out.append((fid, ref))
    return out


def prepare(data_dir: Path) -> None:
    dest = data_dir / "librispeech"
    if (dest / "LibriSpeech" / "test-other").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    tar = dest / "test-other.tar.gz"
    if not tar.exists():
        print(f"downloading {URL} (~330 MB)")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with open(tar, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    with tarfile.open(tar) as tf:
        tf.extractall(dest, filter="data")


def load(data_dir: Path) -> list[Record]:
    root = data_dir / "librispeech" / "LibriSpeech" / "test-other"
    recs = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for fid, ref in parse_trans_file(trans.read_text(encoding="utf-8")):
            recs.append(Record(fid, trans.parent / f"{fid}.flac", ref))
    return recs
```

- [ ] **Step 4: Wire `prepare` in run.py**

Replace the `prepare` branch of `main()`:

```python
    if args.cmd == "prepare":
        from stt_eval import datasets

        for ds_name in args.datasets.split(","):
            print(f"preparing {ds_name}")
            datasets.prepare(ds_name, args.data_dir)
        return
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass (no network in tests — download code is exercised manually in Task 14).

- [ ] **Step 6: Commit**

```bash
git add src/stt_eval/datasets src/stt_eval/run.py tests/test_datasets.py
git commit -m "feat: add dataset registry and LibriSpeech test-other loader"
```

---

### Task 9: PriMock57 dataset

**Files:**
- Create: `src/stt_eval/datasets/primock57.py`
- Test: `tests/test_primock57.py`
- Test fixture: `tests/fixtures/mini_doctor.TextGrid`, `tests/fixtures/mini_patient.TextGrid`

**Interfaces:**
- Consumes: `Record`.
- Produces: `primock57.prepare(data_dir)`, `primock57.load(data_dir) -> list[Record]`, `primock57.merge_reference(doctor_tg: Path, patient_tg: Path) -> str`.

- [ ] **Step 1: Create TextGrid fixtures**

`tests/fixtures/mini_doctor.TextGrid`:

```text
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "Doctor"
        xmin = 0
        xmax = 10
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 2
            text = "Hello, how can I help?"
        intervals [2]:
            xmin = 2
            xmax = 5
            text = ""
        intervals [3]:
            xmin = 5
            xmax = 7
            text = "I see. <clears_throat> Any fever?"
```

`tests/fixtures/mini_patient.TextGrid`:

```text
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 10
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "Patient"
        xmin = 0
        xmax = 10
        intervals: size = 2
        intervals [1]:
            xmin = 2
            xmax = 5
            text = "I have a headache."
        intervals [2]:
            xmin = 7
            xmax = 9
            text = "No fever."
```

- [ ] **Step 2: Write the failing test**

`tests/test_primock57.py`:

```python
from pathlib import Path

from stt_eval.datasets.primock57 import merge_reference

FIXTURES = Path(__file__).parent / "fixtures"


def test_merge_orders_by_time_and_strips_markup():
    ref = merge_reference(FIXTURES / "mini_doctor.TextGrid", FIXTURES / "mini_patient.TextGrid")
    assert ref == "Hello, how can I help? I have a headache. I see. Any fever? No fever."
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_primock57.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement**

`src/stt_eval/datasets/primock57.py`:

```python
"""PriMock57: 57 mock GP consultations (Babylon Health).

Audio ships as separate doctor/patient channel recordings; we mix them to one
16 kHz mono wav per consultation (room-mic condition) and build the reference
by time-ordering both speakers' TextGrid utterances.
"""

import re
import subprocess
from pathlib import Path

import textgrid

from stt_eval.records import Record

REPO = "https://github.com/babylonhealth/primock57.git"


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # non-speech markers like <clears_throat>
    return re.sub(r"\s+", " ", text).strip()


def merge_reference(doctor_tg: Path, patient_tg: Path) -> str:
    utts: list[tuple[float, str]] = []
    for path in (doctor_tg, patient_tg):
        tg = textgrid.TextGrid.fromFile(str(path))
        for tier in tg.tiers:
            for iv in tier:
                t = _clean(iv.mark or "")
                if t:
                    utts.append((iv.minTime, t))
    utts.sort(key=lambda x: x[0])
    return " ".join(t for _, t in utts)


def prepare(data_dir: Path) -> None:
    dest = data_dir / "primock57"
    repo = dest / "repo"
    if not repo.exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=1", REPO, str(repo)], check=True)
    mixed = dest / "mixed"
    mixed.mkdir(exist_ok=True)
    for doc_wav in sorted((repo / "audio").glob("*_doctor.wav")):
        stem = doc_wav.name.removesuffix("_doctor.wav")
        out = mixed / f"{stem}.wav"
        if out.exists():
            continue
        pat_wav = doc_wav.with_name(f"{stem}_patient.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(doc_wav), "-i", str(pat_wav),
             "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest",
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
            check=True,
        )


def load(data_dir: Path) -> list[Record]:
    repo = data_dir / "primock57" / "repo"
    recs = []
    for wav in sorted((data_dir / "primock57" / "mixed").glob("*.wav")):
        ref = merge_reference(
            repo / "transcripts" / f"{wav.stem}_doctor.TextGrid",
            repo / "transcripts" / f"{wav.stem}_patient.TextGrid",
        )
        recs.append(Record(wav.stem, wav, ref))
    return recs
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Verify real repo layout (network, ~1 min)**

Run: `git clone --depth=1 https://github.com/babylonhealth/primock57.git /tmp/primock57-peek && ls /tmp/primock57-peek/audio | head -4 && ls /tmp/primock57-peek/transcripts | head -4 && rm -rf /tmp/primock57-peek`
Expected: files matching `day1_consultation01_doctor.wav` / `..._patient.wav` and `day1_consultation01_doctor.TextGrid` / `..._patient.TextGrid`. If the naming differs (or audio is git-lfs), adjust the glob patterns / add `git lfs pull` in `prepare()` accordingly, and open one real TextGrid to confirm `<...>` is the non-speech marker style; extend `_clean()` if other markers (e.g. `[laughs]`) appear.

- [ ] **Step 7: Commit**

```bash
git add src/stt_eval/datasets/primock57.py tests/test_primock57.py tests/fixtures
git commit -m "feat: add PriMock57 dataset with channel mixing and TextGrid reference merge"
```

---

### Task 10: MedDialog-Audio dataset

**Files:**
- Create: `src/stt_eval/datasets/meddialog_audio.py`
- Test: `tests/test_meddialog.py`

**Interfaces:**
- Consumes: `Record`.
- Produces: `meddialog_audio.prepare(data_dir)`, `meddialog_audio.load(data_dir) -> list[Record]`, `meddialog_audio.pick_subsample(ids_by_condition: dict[str, list[str]], per_condition: int, seed: int) -> dict[str, list[str]]` (pure, deterministic).
- Produces: committed manifest at `results/manifests/meddialog_audio.json` — list of `{"file_id", "condition", "text"}`; `load()` reads only the manifest + wavs (no HF dependency after prepare).

- [ ] **Step 1: Write the failing test (pure subsampling logic)**

`tests/test_meddialog.py`:

```python
from stt_eval.datasets.meddialog_audio import pick_subsample


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_meddialog.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement**

`src/stt_eval/datasets/meddialog_audio.py`:

```python
"""MedDialog-Audio: synthetic medical dialogues (Orpheus TTS over MedDialog-EN)
with noise at multiple SNR levels. https://huggingface.co/datasets/aline-gassenn/MedDialog-Audio

A fixed-seed subsample per noise condition is materialized to wav and recorded
in a committed manifest, so runs are reproducible and API cost stays bounded.
`load()` needs only the manifest + wavs — the HF `datasets` lib is used by
`prepare()` alone (install extra: `uv sync --extra data`).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from stt_eval.records import Record

HF_REPO = "aline-gassenn/MedDialog-Audio"
PER_CONDITION = 300
SEED = 42
MANIFEST = Path("results/manifests/meddialog_audio.json")

# PIN AT EXECUTION (Step 4 inspects the real schema; update these four):
SPLIT = "train"
AUDIO_COL = "audio"
TEXT_COL = "text"
CONDITION_COL = "condition"


def pick_subsample(
    ids_by_condition: dict[str, list[str]], per_condition: int, seed: int
) -> dict[str, list[str]]:
    out = {}
    for cond in sorted(ids_by_condition):
        ids = sorted(ids_by_condition[cond])
        rng = random.Random(f"{seed}:{cond}")
        out[cond] = sorted(rng.sample(ids, min(per_condition, len(ids))))
    return out


def prepare(data_dir: Path) -> None:
    wavs = data_dir / "meddialog" / "wav"
    if MANIFEST.exists() and wavs.exists():
        return
    import datasets as hfd  # optional 'data' extra
    import soundfile as sf

    wavs.mkdir(parents=True, exist_ok=True)
    ds = hfd.load_dataset(HF_REPO, split=SPLIT)
    ids_by_cond: dict[str, list[str]] = defaultdict(list)
    for i, cond in enumerate(ds[CONDITION_COL]):  # column read: no audio decode
        ids_by_cond[str(cond)].append(str(i))
    chosen = pick_subsample(ids_by_cond, PER_CONDITION, SEED)

    manifest = []
    for cond, indices in chosen.items():
        for s in indices:
            i = int(s)
            file_id = f"{cond}__{i:06d}"
            out = wavs / f"{file_id}.wav"
            ex = ds[i]
            if not out.exists():
                audio = ex[AUDIO_COL]
                sf.write(out, audio["array"], audio["sampling_rate"])
            manifest.append({"file_id": file_id, "condition": cond, "text": ex[TEXT_COL]})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest)} entries to {MANIFEST}")


def load(data_dir: Path) -> list[Record]:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wavs = data_dir / "meddialog" / "wav"
    return [
        Record(e["file_id"], wavs / f"{e['file_id']}.wav", e["text"], e["condition"])
        for e in entries
    ]
```

- [ ] **Step 4: Inspect the real schema (network, no bulk download)**

Run:

```bash
uv sync --extra data
uv run python -c "
from datasets import load_dataset_builder
b = load_dataset_builder('aline-gassenn/MedDialog-Audio')
print(b.info.features)
print(b.info.splits)
"
```

Expected: features showing the audio column, transcript column, and how noise
conditions are represented. Update `SPLIT`, `AUDIO_COL`, `TEXT_COL`,
`CONDITION_COL` to the real names. If conditions are encoded as separate
configs/splits rather than a column, restructure `prepare()` to loop
`load_dataset(HF_REPO, config, split=...)` per condition, keeping `pick_subsample`
and the manifest format exactly as-is (they are condition-source-agnostic).
If dialogues are multi-utterance rows, keep one row = one Record.

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass (prepare's network path is exercised in Task 14).

- [ ] **Step 6: Commit**

```bash
git add src/stt_eval/datasets/meddialog_audio.py tests/test_meddialog.py pyproject.toml uv.lock
git commit -m "feat: add MedDialog-Audio dataset with seeded per-condition subsample"
```

---

### Task 11: OpenAI + Deepgram transcribers

**Files:**
- Create: `src/stt_eval/transcribers/openai_api.py`
- Create: `src/stt_eval/transcribers/deepgram_api.py`
- Test: `tests/test_api_transcribers.py`

**Interfaces:**
- Consumes: `base.require_env`.
- Produces: `OpenAITranscribe(client: httpx.Client | None = None)` and `Deepgram(client: httpx.Client | None = None)`, both satisfying the `Transcriber` protocol with `parallel_safe = True`. Constructors call `require_env` (so `create()` raises `MissingKeyError` before any work).

- [ ] **Step 1: Write the failing test**

`tests/test_api_transcribers.py`:

```python
import httpx
import pytest

from stt_eval.transcribers.base import MissingKeyError


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF-fake-wav")
    return p


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_missing_key_raises(monkeypatch):
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingKeyError):
        OpenAITranscribe()


def test_openai_transcribes(monkeypatch, wav):
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer sk-test"
        assert b"gpt-4o-transcribe" in request.content
        return httpx.Response(200, json={"text": "hello from openai"})

    t = OpenAITranscribe(client=_client(handler))
    assert t.transcribe(wav) == "hello from openai"


def test_openai_http_error_raises(monkeypatch, wav):
    from stt_eval.transcribers.openai_api import OpenAITranscribe

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    t = OpenAITranscribe(client=_client(lambda r: httpx.Response(500, text="oops")))
    with pytest.raises(httpx.HTTPStatusError):
        t.transcribe(wav)


def test_deepgram_transcribes(monkeypatch, wav):
    from stt_eval.transcribers.deepgram_api import Deepgram

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")

    def handler(request):
        assert request.url.path == "/v1/listen"
        assert request.url.params["model"] == "nova-3-medical"
        assert request.headers["authorization"] == "Token dg-test"
        assert request.headers["content-type"] == "audio/wav"
        return httpx.Response(200, json={
            "results": {"channels": [{"alternatives": [{"transcript": "hello from dg"}]}]}
        })

    t = Deepgram(client=_client(handler))
    assert t.transcribe(wav) == "hello from dg"


def test_deepgram_sends_flac_content_type(monkeypatch, tmp_path):
    from stt_eval.transcribers.deepgram_api import Deepgram

    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    flac = tmp_path / "a.flac"
    flac.write_bytes(b"fLaC-fake")

    def handler(request):
        assert request.headers["content-type"] == "audio/flac"
        return httpx.Response(200, json={
            "results": {"channels": [{"alternatives": [{"transcript": "x"}]}]}
        })

    Deepgram(client=_client(handler)).transcribe(flac)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api_transcribers.py -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement**

`src/stt_eval/transcribers/openai_api.py`:

```python
from pathlib import Path

import httpx

from stt_eval.transcribers.base import require_env

# PIN AT EXECUTION: latest dated snapshot of gpt-4o-transcribe, per
# https://developers.openai.com/api/docs/guides/speech-to-text
MODEL = "gpt-4o-transcribe"


class OpenAITranscribe:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = MODEL
        self._key = require_env("OPENAI_API_KEY")
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        with audio_path.open("rb") as f:
            resp = self._client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._key}"},
                files={"file": (audio_path.name, f)},
                data={"model": MODEL, "language": "en"},
            )
        resp.raise_for_status()
        return resp.json()["text"]
```

`src/stt_eval/transcribers/deepgram_api.py`:

```python
from pathlib import Path

import httpx

from stt_eval.transcribers.base import require_env

# PIN AT EXECUTION: model + params per https://developers.deepgram.com/docs
MODEL = "nova-3-medical"
CONTENT_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg"}


class Deepgram:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = f"deepgram-{MODEL}"
        self._key = require_env("DEEPGRAM_API_KEY")
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        resp = self._client.post(
            "https://api.deepgram.com/v1/listen",
            params={"model": MODEL, "smart_format": "true", "language": "en"},
            headers={
                "Authorization": f"Token {self._key}",
                "Content-Type": CONTENT_TYPES[audio_path.suffix.lower()],
            },
            content=audio_path.read_bytes(),
        )
        resp.raise_for_status()
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass, including Task 5's `test_registry_import_is_lazy` (these modules import only httpx).

- [ ] **Step 5: Size guard note (OpenAI 25 MB limit)**

PriMock57 mixed wavs are 16 kHz mono s16le ≈ 1.9 MB/min, so a 10-minute consultation ≈ 19 MB < 25 MB. Add a guard so oversize files fail loudly instead of with an opaque 413 — in `OpenAITranscribe.transcribe`, before the POST:

```python
        if audio_path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"{audio_path.name} exceeds OpenAI 25 MB upload limit; "
                             "transcode to FLAC or chunk it")
```

Run: `uv run pytest -q` — still green.

- [ ] **Step 6: Commit**

```bash
git add src/stt_eval/transcribers/openai_api.py src/stt_eval/transcribers/deepgram_api.py tests/test_api_transcribers.py
git commit -m "feat: add OpenAI and Deepgram API transcribers"
```

---

### Task 12: AssemblyAI + Soniox transcribers (upload + poll)

**Files:**
- Create: `src/stt_eval/transcribers/assemblyai_api.py`
- Create: `src/stt_eval/transcribers/soniox_api.py`
- Modify: `src/stt_eval/transcribers/base.py` (add `poll_until`)
- Test: `tests/test_api_transcribers_polling.py`

**Interfaces:**
- Consumes: `base.require_env`.
- Produces: `base.poll_until(fetch: Callable[[], dict], is_done: Callable[[dict], bool], interval: float = 3.0, timeout: float = 1800.0) -> dict` — raises `TimeoutError` on expiry.
- Produces: `AssemblyAI(client=None)`, `Soniox(client=None)` satisfying the protocol, `parallel_safe = True`.

- [ ] **Step 1: Write the failing test**

`tests/test_api_transcribers_polling.py`:

```python
import httpx
import pytest

from stt_eval.transcribers.base import poll_until


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF-fake-wav")
    return p


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_poll_until_polls_to_completion(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    states = iter([{"status": "queued"}, {"status": "processing"}, {"status": "completed"}])
    result = poll_until(lambda: next(states), lambda d: d["status"] == "completed")
    assert result["status"] == "completed"


def test_poll_until_times_out(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr("time.monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        poll_until(lambda: {"status": "processing"}, lambda d: False, timeout=500)


def test_assemblyai_upload_then_poll(monkeypatch, wav):
    from stt_eval.transcribers.assemblyai_api import AssemblyAI

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aa-test")
    monkeypatch.setattr("time.sleep", lambda s: None)
    polls = {"n": 0}

    def handler(request):
        assert request.headers["authorization"] == "aa-test"
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/aa/1"})
        if request.url.path == "/v2/transcript" and request.method == "POST":
            return httpx.Response(200, json={"id": "job1", "status": "queued"})
        assert request.url.path == "/v2/transcript/job1"
        polls["n"] += 1
        if polls["n"] < 2:
            return httpx.Response(200, json={"id": "job1", "status": "processing"})
        return httpx.Response(200, json={"id": "job1", "status": "completed", "text": "hi from aa"})

    t = AssemblyAI(client=_client(handler))
    assert t.transcribe(wav) == "hi from aa"


def test_assemblyai_error_status_raises(monkeypatch, wav):
    from stt_eval.transcribers.assemblyai_api import AssemblyAI

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aa-test")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request):
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/aa/1"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job1", "status": "queued"})
        return httpx.Response(200, json={"id": "job1", "status": "error", "error": "bad audio"})

    with pytest.raises(RuntimeError, match="bad audio"):
        AssemblyAI(client=_client(handler)).transcribe(wav)


def test_soniox_upload_then_poll(monkeypatch, wav):
    from stt_eval.transcribers.soniox_api import Soniox

    monkeypatch.setenv("SONIOX_API_KEY", "sx-test")
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(request):
        assert request.headers["authorization"] == "Bearer sx-test"
        if request.url.path == "/v1/files":
            return httpx.Response(201, json={"id": "file1"})
        if request.url.path == "/v1/transcriptions" and request.method == "POST":
            assert request.content and b"stt-async-v5" in request.content
            return httpx.Response(201, json={"id": "tx1", "status": "queued"})
        if request.url.path == "/v1/transcriptions/tx1":
            return httpx.Response(200, json={"id": "tx1", "status": "completed"})
        assert request.url.path == "/v1/transcriptions/tx1/transcript"
        return httpx.Response(200, json={"text": "hi from soniox"})

    t = Soniox(client=_client(handler))
    assert t.transcribe(wav) == "hi from soniox"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api_transcribers_polling.py -v`
Expected: FAIL — `poll_until` and modules don't exist.

- [ ] **Step 3: Add poll_until to base.py**

Append to `src/stt_eval/transcribers/base.py`:

```python
def poll_until(
    fetch: Callable[[], dict],
    is_done: Callable[[dict], bool],
    interval: float = 3.0,
    timeout: float = 1800.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        state = fetch()
        if is_done(state):
            return state
        if time.monotonic() > deadline:
            raise TimeoutError(f"polling timed out after {timeout}s; last state: {state}")
        time.sleep(interval)
```

- [ ] **Step 4: Implement the two transcribers**

`src/stt_eval/transcribers/assemblyai_api.py`:

```python
from pathlib import Path

import httpx

from stt_eval.transcribers.base import poll_until, require_env

BASE = "https://api.assemblyai.com/v2"
# PIN AT EXECUTION: speech_model value for Universal-3, per
# https://www.assemblyai.com/docs
SPEECH_MODEL = "universal-3"


class AssemblyAI:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = f"assemblyai-{SPEECH_MODEL}"
        self._headers = {"authorization": require_env("ASSEMBLYAI_API_KEY")}
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        up = self._client.post(f"{BASE}/upload", headers=self._headers,
                               content=audio_path.read_bytes())
        up.raise_for_status()
        job = self._client.post(
            f"{BASE}/transcript", headers=self._headers,
            json={"audio_url": up.json()["upload_url"],
                  "speech_model": SPEECH_MODEL, "language_code": "en"},
        )
        job.raise_for_status()
        job_id = job.json()["id"]

        def fetch() -> dict:
            r = self._client.get(f"{BASE}/transcript/{job_id}", headers=self._headers)
            r.raise_for_status()
            return r.json()

        state = poll_until(fetch, lambda d: d["status"] in ("completed", "error"))
        if state["status"] == "error":
            raise RuntimeError(f"assemblyai job failed: {state.get('error')}")
        return state["text"]
```

`src/stt_eval/transcribers/soniox_api.py`:

```python
from pathlib import Path

import httpx

from stt_eval.transcribers.base import poll_until, require_env

BASE = "https://api.soniox.com/v1"
# PIN AT EXECUTION: endpoints + payload per https://soniox.com/docs
MODEL = "stt-async-v5"


class Soniox:
    parallel_safe = True

    def __init__(self, client: httpx.Client | None = None):
        self.name = f"soniox-{MODEL}"
        self._headers = {"Authorization": f"Bearer {require_env('SONIOX_API_KEY')}"}
        self._client = client or httpx.Client(timeout=600)

    def transcribe(self, audio_path: Path) -> str:
        with audio_path.open("rb") as f:
            up = self._client.post(f"{BASE}/files", headers=self._headers,
                                   files={"file": (audio_path.name, f)})
        up.raise_for_status()
        tx = self._client.post(f"{BASE}/transcriptions", headers=self._headers,
                               json={"file_id": up.json()["id"], "model": MODEL})
        tx.raise_for_status()
        tx_id = tx.json()["id"]

        def fetch() -> dict:
            r = self._client.get(f"{BASE}/transcriptions/{tx_id}", headers=self._headers)
            r.raise_for_status()
            return r.json()

        state = poll_until(fetch, lambda d: d["status"] in ("completed", "error"))
        if state["status"] == "error":
            raise RuntimeError(f"soniox job failed: {state.get('error_message')}")
        r = self._client.get(f"{BASE}/transcriptions/{tx_id}/transcript", headers=self._headers)
        r.raise_for_status()
        return r.json()["text"]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 6: Verify pinned endpoints against provider docs**

Open https://www.assemblyai.com/docs (transcript endpoint: `speech_model` value for Universal-3) and https://soniox.com/docs (async flow: file upload path, transcription creation payload, transcript retrieval path, terminal status names). Update the `PIN AT EXECUTION` constants and, if a path/payload differs, both module and mock test together. Note changes in the commit message.

- [ ] **Step 7: Commit**

```bash
git add src/stt_eval/transcribers/base.py src/stt_eval/transcribers/assemblyai_api.py src/stt_eval/transcribers/soniox_api.py tests/test_api_transcribers_polling.py
git commit -m "feat: add AssemblyAI and Soniox transcribers with shared polling helper"
```

---

### Task 13: Local transcribers (faster-whisper + Qwen3-ASR)

**Files:**
- Create: `src/stt_eval/transcribers/whisper_local.py`
- Create: `src/stt_eval/transcribers/qwen3_asr.py`
- Test: `tests/test_local_transcribers.py`

**Interfaces:**
- Produces: `WhisperLocal(model_id: str, language: str = "en")`, `Qwen3ASR(repo: str, language: str = "en")` — protocol-satisfying, `parallel_safe = False` (GPU: one at a time). Heavy imports happen inside `__init__`, never at module import time.

- [ ] **Step 1: Write the failing test**

`tests/test_local_transcribers.py`:

```python
import sys


def test_module_import_stays_light():
    import stt_eval.transcribers.whisper_local  # noqa: F401
    import stt_eval.transcribers.qwen3_asr  # noqa: F401

    assert "faster_whisper" not in sys.modules
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules


def test_classes_declare_not_parallel_safe():
    from stt_eval.transcribers.qwen3_asr import Qwen3ASR
    from stt_eval.transcribers.whisper_local import WhisperLocal

    assert WhisperLocal.parallel_safe is False
    assert Qwen3ASR.parallel_safe is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_local_transcribers.py -v`
Expected: FAIL — modules don't exist.

- [ ] **Step 3: Implement**

`src/stt_eval/transcribers/whisper_local.py`:

```python
from pathlib import Path


class WhisperLocal:
    parallel_safe = False  # one GPU model instance; batching happens inside

    def __init__(self, model_id: str, language: str = "en"):
        from faster_whisper import WhisperModel  # local extra: uv sync --extra local

        self.name = f"whisper-{model_id}"
        self.language = language
        self._model = WhisperModel(model_id, device="auto", compute_type="auto")

    def transcribe(self, audio_path: Path) -> str:
        segments, _info = self._model.transcribe(str(audio_path), language=self.language)
        return " ".join(s.text.strip() for s in segments)
```

`src/stt_eval/transcribers/qwen3_asr.py`:

```python
from pathlib import Path


class Qwen3ASR:
    parallel_safe = False

    def __init__(self, repo: str, language: str = "en"):
        # PIN AT EXECUTION: follow the usage snippet on the model card
        # https://huggingface.co/Qwen/Qwen3-ASR-1.7B (native transformers support;
        # verify the pipeline task/kwargs and long-form handling there).
        import torch
        from transformers import pipeline

        self.name = repo.rsplit("/", 1)[-1].lower()
        self.language = language
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def transcribe(self, audio_path: Path) -> str:
        return self._pipe(str(audio_path))["text"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/stt_eval/transcribers/whisper_local.py src/stt_eval/transcribers/qwen3_asr.py tests/test_local_transcribers.py
git commit -m "feat: add faster-whisper and Qwen3-ASR local transcribers"
```

---

### Task 14: Docs + manual verification checklist

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above; this task verifies the assembled system on real audio.

- [ ] **Step 1: Write README**

Replace `README.md` content with:

````markdown
# clinical-speech-understanding

Benchmarking STT models for a Korean healthcare "AI scribe". Phase 1: English
medical audio. Design: `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`.

## Setup

Requires Python ≥3.10, [uv](https://docs.astral.sh/uv/), `ffmpeg`, and `git` on PATH.

```bash
uv sync                      # core (scoring, API transcribers)
uv sync --extra data         # + HF datasets (MedDialog-Audio prepare)
uv sync --extra local        # + faster-whisper, transformers, torch (GPU models)
```

API keys (a missing key skips that model with a warning):
`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY`, `SONIOX_API_KEY`.

## Usage

```bash
uv run stt-eval prepare --datasets librispeech-test-other,primock57,meddialog-audio
uv run stt-eval transcribe \
  --models whisper-large-v3-turbo,gpt-4o-transcribe \
  --datasets primock57 --workers 8
uv run stt-eval score       # -> results/wer_summary.{csv,md}, results/wer_per_file.csv
```

Models: `whisper-large-v3`, `whisper-large-v3-turbo`, `qwen3-asr-0.6b`,
`qwen3-asr-1.7b`, `gpt-4o-transcribe`, `deepgram-nova-3-medical`,
`assemblyai-universal-3-5-pro`, `soniox-stt-async-v5`.

Transcripts are cached per file under `results/transcripts/` (committed);
re-runs skip cached files, so interrupted runs resume and APIs are never
double-billed. `--limit N` transcribes only the first N records (smoke tests).

## Tests

```bash
uv run pytest        # no network, no GPU, no API keys needed
```
````

- [ ] **Step 2: Update CLAUDE.md**

Replace the "Repository Status" and "Guidance" sections of `CLAUDE.md` with:

```markdown
## Project

STT benchmarking for a Korean healthcare "AI scribe" (phase 1: English).
Spec: `docs/superpowers/specs/2026-07-05-stt-benchmark-design.md`.
Plan: `docs/superpowers/plans/2026-07-05-stt-benchmark-harness.md`.

## Commands

- `uv sync --group dev` — install (extras: `--extra data`, `--extra local`)
- `uv run pytest` — test suite (no network/GPU/keys; keep it that way)
- `uv run pytest tests/test_score.py::test_score_groups_and_pools -v` — single test
- `uv run stt-eval prepare|transcribe|score` — the benchmark pipeline (see README)

## Architecture

- `src/stt_eval/transcribers/` — one module per STT backend behind a lazy
  registry (`REGISTRY` in `__init__.py`); protocol: `transcribe(Path) -> str`.
  API backends use plain httpx (tests inject `httpx.MockTransport`).
- `src/stt_eval/datasets/` — loaders yielding `Record(file_id, audio_path,
  reference, condition)`; `prepare()` downloads, `load()` is offline.
- `results/transcripts/{dataset}/{model}/{file_id}.json` — committed cache;
  the reference text is embedded, so `stt-eval score` needs no dataset access.
- WER convention: vendored Whisper English normalizer + jiwer corpus WER
  (Open ASR Leaderboard convention — numbers comparable to published ones).
- `data/` is gitignored; `results/` is committed.
```

- [ ] **Step 3: Full-suite check and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add README.md CLAUDE.md
git commit -m "docs: add setup, usage, and architecture documentation"
```

- [ ] **Step 4: Manual verification — LibriSpeech smoke (network; GPU box or Mac)**

```bash
uv sync --extra local
uv run stt-eval prepare --datasets librispeech-test-other
uv run stt-eval transcribe --models whisper-large-v3-turbo \
  --datasets librispeech-test-other --limit 20
uv run stt-eval score
```

Expected: `results/wer_summary.csv` has a `whisper-large-v3-turbo` row with
WER roughly in the 0.03–0.10 range on those 20 files. Numbers far above that
mean a harness bug (check normalization and reference pairing), not a bad model.

- [ ] **Step 5: Manual verification — one API + PriMock57 (needs one API key)**

```bash
export DEEPGRAM_API_KEY=...   # or any one key
uv run stt-eval prepare --datasets primock57
uv run stt-eval transcribe --models deepgram-nova-3-medical --datasets primock57 --limit 3
uv run stt-eval score
```

Expected: 3 committed JSONs under `results/transcripts/primock57/deepgram-nova-3-medical/`,
each with non-empty `text` and `reference`; WER plausibly 0.1–0.3 (conversational
audio is hard). Commit the transcripts:

```bash
git add results/
git commit -m "results: smoke-run transcripts (librispeech + primock57)"
```

- [ ] **Step 6: Full English round (GPU box, all keys) — the actual benchmark**

```bash
uv run stt-eval prepare --datasets librispeech-test-other,primock57,meddialog-audio
uv run stt-eval transcribe --models whisper-large-v3,whisper-large-v3-turbo,qwen3-asr-0.6b,qwen3-asr-1.7b \
  --datasets librispeech-test-other,primock57,meddialog-audio
uv run stt-eval transcribe --models gpt-4o-transcribe,deepgram-nova-3-medical,assemblyai-universal-3-5-pro,soniox-stt-async-v5 \
  --datasets librispeech-test-other,primock57,meddialog-audio --workers 8
uv run stt-eval score
git add results/ && git commit -m "results: English round WER benchmark"
```

Sanity checks before trusting the table: LibriSpeech WERs should be within a
couple of points of published numbers for whisper-large-v3 (~4–5% on
test-other); every (model, dataset) row should have `n_failed` near zero;
PriMock57 WERs will be much higher than LibriSpeech — that is the point,
not a bug.

---

## Self-Review Notes

- Spec coverage: all 8 models (Tasks 11–13), all 3 datasets (Tasks 8–10), WER + normalization convention (Tasks 2–3, 7), caching/resume/failure records (Tasks 4, 6), missing-key skip (Tasks 5–6), condition grouping (Tasks 7, 10), headless CLI (Tasks 1, 6–8), no-network tests (throughout), README env-var docs (Task 14). Korean phase is spec'd as forward-look only — no task, correct.
- Known execution-time uncertainties are localized to `PIN AT EXECUTION` constants with docs URLs and explicit verify steps (Tasks 1, 10, 11, 12, 13) plus layout checks (Task 9 Step 6, Task 10 Step 4).
- Type consistency verified: `cache_path(root, dataset, model, file_id)` callers; cache JSON keys writer (Task 6) vs reader (Task 7) vs test `_put` helper; `Record` field order `(file_id, audio_path, reference, condition)` in all loaders; registry keys in Task 5 = Global Constraints = README.
