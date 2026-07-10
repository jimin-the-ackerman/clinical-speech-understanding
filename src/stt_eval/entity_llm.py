"""LLM entity extractors (method 4): MedGemma local + OpenRouter general model.
Both satisfy extract(reference)->[surface forms] and are dispatched by
entity_score.extractor_for. Heavy deps (transformers/torch/bitsandbytes) import
lazily; httpx is a core dep. See knowledge/metrics/medical-term-recall.md."""

import json
import re

from stt_eval.entity_score import _dedupe_ci

_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')  # one complete JSON string literal


def _salvage_strings(text: str) -> list[str]:
    """Recover the complete quoted strings from a truncated/unterminated JSON
    array. A degenerate-repetition run (greedy decoding loops on one token) blows
    max_new_tokens and leaves `["a", "b", "c` with no closing `]`, which the
    parsers above drop to [] — silently dropping the whole reference. Salvage the
    strings that closed before the cutoff so it contributes its pre-loop entities
    instead of vanishing. Only fires when an array was opened (a leading `[`), so
    a genuine prose refusal still yields []."""
    start = text.find("[")
    if start == -1:
        return []
    out = []
    for m in _STRING.finditer(text[start:]):
        try:
            out.append(json.loads(m.group(0)))
        except Exception:
            pass
    return out


def _parse_entity_list(raw: str) -> list[str]:
    """Best-effort: strip fences/prose, pull a JSON array, coerce to a deduped
    list of non-empty strings. A truncated/unterminated array is salvaged to its
    complete elements; a genuine refusal/empty -> []."""
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
        items = _salvage_strings(text)  # truncated/unterminated array -> recover the head
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


# --- MedGemma route: local, medical-specialized, single-GPU ----------------

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
    # device_map={"":0}, not "auto": a ~15 GB 4-bit 27B fits one 24 GB GPU, and
    # accelerate's "auto" can silently offload a layer to CPU and deadlock the load.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)

    def extract(reference: str) -> list[str]:
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _user(reference)}]
        # return_dict=True → BatchEncoding (input_ids + attention_mask). A bare
        # tensor has no .shape via BatchEncoding.__getattr__, which raised a
        # message-less AttributeError; pass the dict to generate and slice the
        # new tokens off input_ids' length.
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        # 1024, not 512: long consults over-extract (~80 terms) and a list cut off
        # mid-array is unterminated JSON that parses to [] — silently dropping the file.
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        return _parse_entity_list(text)

    extract.parallel_safe = False
    return extract


# --- bake-off: side-by-side entity sets for a human pick -------------------

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
