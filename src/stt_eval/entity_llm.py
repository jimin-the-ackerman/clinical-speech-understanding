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
