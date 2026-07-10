---
type: Entity Method
title: medgemma (local LLM)
description: MedGemma-27B (4-bit, local) zero-shot clinical-term extraction; the medical-specialized LLM route.
tags: [entity-method, llm, medgemma, local]
timestamp: 2026-07-08
---

# medgemma — MedGemma-27B (local LLM)

The medical-specialized LLM route: `google/medgemma-27b-text-it` prompted zero-shot to extract
clinical terms open-endedly (no fixed label set). `medgemma_extractor` in `entity_llm.py`,
`parallel_safe = False`. Gated HF repo — needs `HF_TOKEN` + license acceptance.

- **Load**: 4-bit (`BitsAndBytesConfig` nf4 / double-quant, bf16 compute) via
  `AutoModelForCausalLM`; ~16 GB VRAM, fits a 24 GB GPU.
- **Gotchas (all fixed in code)**: `device_map={"":0}`, not `"auto"` (avoids a CPU-offload
  deadlock); `apply_chat_template(return_dict=True)` + `generate(**inputs)` (transformers v5
  returns a BatchEncoding, so `.shape` on it raised a bare `AttributeError`); `max_new_tokens=1024`
  (512 truncated the longest consults into unterminated JSON → silent `[]`; greedy can still loop
  on a token and truncate, so `_parse_entity_list` now **salvages** the complete terms before the
  cutoff rather than dropping the whole reference to `[]`); `bitsandbytes` in the `local` extra
  (a `--with` overlay pulls cu130 torch and disables CUDA). See [environment](../project/environment.md).
- **Faithfulness & decoding**: ~99% of extracted terms are present in the reference (token match),
  on par with the extractive NER methods, so it is not inventing entities. Absolute recall is
  decoding-sensitive (a `repetition_penalty` sweep shifted the numbers ~1 pt) but the model
  **ranking is invariant** — see the [log](../log.md).
- **Build**:
  ```
  uv run --extra local --env-file .env stt-eval entity-build --method medgemma --datasets primock57
  ```
- **Result**: extracts ~44 terms/file (broad, incl. some lifestyle context) yet ranks Soniox #1,
  identical to bc5cdr — the specialized LLM confirms the NER finding. See the
  [finding](../findings/medical-term-recall.md).
