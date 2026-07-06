# Medical-entity metric: method comparison (working status)

Status as of 2026-07-07, branch `stt-dev`. This is the live working state of the
medical-term-recall exercise and the OSCE dataset, so work can resume after a
context compaction. Background: `docs/research/2026-07-06-medical-entity-asr-metrics.md`
(literature + design), `docs/tutorials/evaluation-workflow.md` (how scoring works).

## What the metric is

Medical-term **recall**: fraction of reference clinical entities whose surface form
survives into the hypothesis, reported per (model, dataset, condition) alongside WER.
Two-stage design (`src/stt_eval/entity_score.py`):
- `entity-build --method X` → frozen per-file manifest `results/entity_manifests/X.json`
  (method-specific, heavy; where cost/nondeterminism lives).
- `entity-score --manifest P` → offline recall table `results/entity_recall_<X>.{csv,md}`
  (deterministic, no NER/keys; comparing methods = scoring different manifests).

## The exercise: 4 entity-identification methods

Goal (user's framing): try a dictionary, a typed-NER model, and an LLM, to show
whether the ranking is robust to how "medical term" is defined. Even where rankings
hold, demonstrating stability across methods is the deliverable.

| Method | What | Status | Reproduce (build) |
|---|---|---|---|
| `bc5cdr` | scispaCy disease+chemical (narrow) | DONE | `uv run --with scispacy --with "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz" stt-eval entity-build --method bc5cdr` |
| `ner-union` | Med7 (drug/dose) + Stanza i2b2 (problem/test/treatment) | DONE | `uv run --with "en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-1.1.0-py3-none-any.whl" --with stanza stt-eval entity-build --method ner-union` |
| `dictionary` | 7419-term gazetteer seeded from bc5cdr∪ner-union clinical entities, greedy longest-match; fully offline | DONE | gazetteer at `results/entity_dictionaries/medical_terms.txt`; then `uv run stt-eval entity-build --method dictionary` |
| `llm` | LLM zero-shot over references, frozen to manifest | HARNESS DONE, runs pending keys — see below | `stt-eval entity-build --method openrouter --model <id>` / `--method medgemma` |

Scoring any manifest: `uv run stt-eval entity-score --manifest results/entity_manifests/<method>.json`

Env notes: bc5cdr pins spacy<3.8, Med7 pins spacy>=3.8 — they CANNOT share one env,
so `ner-union` is Med7+Stanza only and bc5cdr stays a separate baseline. All entity
runs use ephemeral `uv run --with` overlays (never touch the project env). Med7 wheel
must be the versioned `-1.1.0-py3-none-any.whl` (the `-any-` URL is not PEP 440).

## The finding so far (PriMock57, the clinical target)

Model ranking (best→worst medical-term fidelity), by method:
- WER:        qwen3-asr-1.7b > soniox > qwen-0.6b > turbo > whisper-v3 > gpt-4o
- bc5cdr:     **soniox** > qwen-1.7b > whisper-v3 > turbo > qwen-0.6b > gpt-4o
- ner-union:  **soniox** > qwen-1.7b > whisper-v3 > qwen-0.6b > turbo > gpt-4o
- dictionary: qwen-1.7b > soniox > turbo > whisper-v3 > qwen-0.6b > gpt-4o

Takeaways:
1. The two **selective NER methods agree: Soniox #1** on clinical medical terms (WER ranks
   it #2), and whisper-v3 ranks #3 vs its #5 WER rank — its errors hit function words, not
   clinical content. Robust across narrow and broad NER.
2. The **dictionary diverges** (tracks WER, qwen-1.7b #1). Cause is mechanical: uncurated,
   it matches every occurrence of every gazetteer term (PriMock57: 14,668 matches vs
   ner-union's 2,670), so frequent generic terms dilute it toward WER. → entity-ID method
   is NOT neutral at the top; curation matters. This motivates the LLM (selective) method.
3. Overall recall correlates strongly with WER (Pearson ~-0.97) — a complement, not a
   replacement. LibriSpeech entity columns are noise (all-caps refs break NER); ignore them.

## Method 4 (LLM) — harness built (2026-07-07), runs awaiting keys

**Code is done and committed** (plan `docs/superpowers/plans/2026-07-07-llm-entity-method.md`):
`src/stt_eval/entity_llm.py` holds `openrouter_extractor` (parallel-safe, keyed) and
`medgemma_extractor` (local, 4-bit). `build_manifest` is now resumable/parallel with a
per-reference cache under `results/entity_cache/<method>/` (gitignored scratch; a crashed
or expensive run resumes and never re-bills). CLI: `entity-build --method openrouter|medgemma
[--model ID] [--workers N] [--limit N]` and `entity-bakeoff --specs method[:model],... --limit N`.
87 offline tests green. What's left is purely the runtime, which needs keys in `.env`:

1. Bake-off (pick the OpenRouter model), needs `OPENROUTER_API_KEY` (+ `HF_TOKEN` if MedGemma is in the specs):
   `uv run --extra local --with bitsandbytes --env-file .env stt-eval entity-bakeoff --specs "medgemma,openrouter:anthropic/claude-opus-4.8,openrouter:google/gemini-2.5-flash" --limit 15`
2. MedGemma smoke then full build (needs `HF_TOKEN` + license accept):
   `uv run --extra local --with bitsandbytes --env-file .env stt-eval entity-build --method medgemma --limit 5` then drop `--limit`.
3. OpenRouter full build (winning model): `uv run --env-file .env stt-eval entity-build --method openrouter --model <winner> --workers 8`
4. Score both: `uv run stt-eval entity-score --manifest results/entity_manifests/<name>.json`, then rank
   PriMock57 across all 4 methods and answer: does the selective LLM set side with the NER methods
   (Soniox #1) or the dictionary (qwen-1.7b #1)? Does MedGemma (specialized) differ from OpenRouter (general)?

Two routes, ideally BOTH for a specialized-vs-general comparison (user leaned toward both):
- **Medical-specialized, local (no key, runs now on the 4090):** MedGemma — open weights on
  HF (`google/medgemma-27b-text-it` for text; 27B needs 4-bit to fit 24 GB, or `medgemma-4b-it`
  bf16 for speed). This is the closest thing to the "MedPaLM-like" model the user wanted.
  Med-PaLM 2 is NOT usable (Google MedLM API, healthcare-approval-gated, not on OpenRouter).
- **General frontier, via OpenRouter (needs `OPENROUTER_API_KEY` in `.env`):** bake off on
  ~15 clinical refs, then extract with the winner. Real catalog IDs + input $/M:
  `anthropic/claude-opus-4.8` ($5), `google/gemini-3.1-pro-preview` ($2),
  `google/gemini-2.5-flash` ($0.30). Deliberately NO OpenAI model (gpt-4o-transcribe is one
  of the scored ASR systems). Full extraction is ~$2–5 total (cost is not the constraint;
  pick for quality). No medical-specialized model exists on OpenRouter (verified).

Open question the LLM method answers: does a selective LLM entity set land with the NER
methods (Soniox #1) or the dictionary (qwen-1.7b #1)?

## OSCE / Fareez dataset (task 2) — data ready, NOT transcribed

- Loader `datasets/fareez.py` (name `fareez-interviews`), DONE + committed. 272 verbatim
  patient-physician interviews, ~51 h, CC0, 16 kHz mono MP3 (reads via soundfile). Merged
  to one reference per interview like PriMock57. Data downloaded to `data/fareez-interviews/`
  and verified loadable (272 records). Transcripts confirmed verbatim (disfluencies present).
- **NOT transcribed yet — this is the paid checkpoint.** ~51 audio-h = more than the other
  3 datasets combined. Cost ~$30–50 for the two API models (gpt-4o + soniox), plus GPU hours
  for the 4 local models. AWAITING user go-ahead on scope (all 6 models? APIs? or local-only
  first, which is free). Once transcribed, rebuild the entity manifests to include OSCE.

## Immediate next actions (pending user input)

1. LLM method: run MedGemma locally now (no key), and/or the OpenRouter bake-off once
   `OPENROUTER_API_KEY` is added. User leaned "both".
2. OSCE transcription: the paid checkpoint above — confirm scope before spending.
3. Deferred: fuzzy entity matching (current match is exact contiguous tokens; `ponytail:`
   note in entity_score.py). Deepgram + AssemblyAI ASR models still skipped (no keys).
