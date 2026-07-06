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


# --- shared prompt ---------------------------------------------------------

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


# --- OpenRouter route: general frontier model, keyed, parallel-safe --------

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
