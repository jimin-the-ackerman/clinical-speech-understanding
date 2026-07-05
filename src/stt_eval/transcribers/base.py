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
