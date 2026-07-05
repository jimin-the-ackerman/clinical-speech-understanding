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
