import importlib
from pathlib import Path

from stt_eval.records import Record

DATASETS = ("primock57", "meddialog-audio", "librispeech-test-other", "fareez-interviews")

_MODULES = {
    "primock57": "primock57",
    "meddialog-audio": "meddialog_audio",
    "librispeech-test-other": "librispeech",
    "fareez-interviews": "fareez",
}


def _module(name: str):
    if name not in _MODULES:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    return importlib.import_module(f"stt_eval.datasets.{_MODULES[name]}")


def prepare(name: str, data_dir: Path) -> None:
    _module(name).prepare(data_dir)


def load(name: str, data_dir: Path) -> list[Record]:
    return _module(name).load(data_dir)
