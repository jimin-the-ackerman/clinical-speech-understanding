# Medical-entity metric: method comparison (working status)

Status as of 2026-07-08, branch `stt-dev`. This is the live working state of the
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

## The exercise: 5 entity-identification methods

Goal (user's framing): try typed-NER models and an LLM, to show whether the ranking
is robust to how "medical term" is defined. Even where rankings hold, demonstrating
stability across methods is the deliverable.

| Method | What | Status | Reproduce (build) |
|---|---|---|---|
| `bc5cdr` | scispaCy disease+chemical (narrow) | DONE | `uv run --with scispacy --with "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz" stt-eval entity-build --method bc5cdr --datasets primock57,meddialog-audio` |
| `med7` | Med7 (`en_core_med7_lg`): drug/dosage/route/frequency; sparse (~0.9 ent/file) | **DONE** — soniox #1 | `uv run --with "en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-1.1.0-py3-none-any.whl" stt-eval entity-build --method med7 --datasets primock57,meddialog-audio` |
| `stanza-i2b2` | Stanza i2b2: problem/test/treatment; dense (~6 ent/file, most of the old union's mass) | **DONE** — soniox #1 | `uv run --with stanza stt-eval entity-build --method stanza-i2b2 --datasets primock57,meddialog-audio` |
| `medgemma` | MedGemma-27B (4-bit, local) zero-shot, PriMock57 only | **DONE** — sides with NER (soniox #1) | `uv run --extra local --env-file .env stt-eval entity-build --method medgemma --datasets primock57` |
| `openrouter` | general frontier LLM (specialized-vs-general foil) | pending `OPENROUTER_API_KEY` | `stt-eval entity-build --method openrouter --model <id>` |

Scoring any manifest: `uv run stt-eval entity-score --manifest results/entity_manifests/<method>.json`

Env notes: each NER method runs in its own ephemeral `uv run --with` overlay (never
touches the project env), so their conflicting deps never meet — `med7` needs only spaCy
(`en_core_med7_lg`, spacy>=3.8) + the model wheel; `stanza-i2b2` needs only Stanza (no
spaCy); `bc5cdr` needs scispaCy (spacy<3.8). The Med7 wheel must be the versioned
`-1.1.0-py3-none-any.whl` (the `-any-` URL is not PEP 440).

## The finding so far (PriMock57, the clinical target)

Model ranking (best→worst medical-term fidelity), by method:
- WER:        qwen3-asr-1.7b > soniox > qwen-0.6b > turbo > whisper-v3 > gpt-4o
- bc5cdr:      **soniox** > qwen-1.7b > whisper-v3 > turbo > qwen-0.6b > gpt-4o
- med7:        **soniox** > qwen-1.7b > whisper-v3 > gpt-4o > qwen-0.6b > turbo  (sparse ⇒ noisy tail)
- stanza-i2b2: **soniox** > qwen-1.7b > whisper-v3 > qwen-0.6b > turbo > gpt-4o
- medgemma:    **soniox** > qwen-1.7b > whisper-v3 > turbo > qwen-0.6b > gpt-4o  (identical to bc5cdr)

Takeaways:
1. **All 4 methods agree: Soniox #1**, with an identical top-3 (soniox > qwen-1.7b > whisper-v3)
   on clinical medical terms (WER ranks Soniox #2) — across bc5cdr (disease+chemical), med7
   (drugs), stanza-i2b2 (problem/test/treatment), AND the MedGemma LLM. Splitting the old
   Med7+Stanza union showed the two NER schemas **independently** concur — neither drove it
   (i2b2 carries ~13,100 of the union's 14,231 spans; med7 only ~1,973, yet still ranks Soniox
   #1). whisper-v3 ranks #3 vs its #5 WER rank — its errors hit function words, not clinical
   content. Only the bottom ranks wiggle (notably med7, whose ~0.9 terms/file give a small,
   noisier denominator).
2. Overall recall correlates strongly with WER (Pearson ~-0.97) — a complement, not a
   replacement. **LibriSpeech is excluded from every manifest** (all-caps read-speech with no
   clinical content — NER produces only noise there): the NER manifests are built with
   `--datasets primock57,meddialog-audio` (2,157 entries = 57 PriMock57 + 2,100 MedDialog);
   MedGemma is PriMock57-only (57 entries).

## LLM methods — MedGemma done, OpenRouter pending

**MedGemma-27B (4-bit, local) — DONE on PriMock57:** 57 files, 2,505 entities, mean 44/file,
0 empty. Manifest `results/entity_manifests/medgemma.json`, recall
`results/entity_recall_medgemma.{csv,md}`. Verdict: **sides with the NER methods (Soniox #1),
ranking identical to bc5cdr.** Run cost: ~5 s model load (warm page cache) + ~4 s/ref.
Reproducibility gotchas (all fixed in code): `bitsandbytes` must live in the `local` extra —
a `uv run --with bitsandbytes` overlay pulls cu130 torch and disables CUDA on driver <580;
`apply_chat_template` returns a BatchEncoding in transformers v5 (use `return_dict=True` +
`generate(**inputs)`); `device_map={"":0}` avoids CPU-offload stalls; `max_new_tokens=1024`
(512 truncated the longest consults' term lists into unterminated JSON → `[]`). Optional
extension: MedGemma on MedDialog (2,100 refs, ~2.5 h) for the noise-robustness cut.

**OpenRouter general model — PENDING `OPENROUTER_API_KEY`** (the one remaining method: the
specialized-vs-general foil). Code done: `entity_llm.openrouter_extractor` (parallel-safe,
keyed), `build_manifest` resumable/parallel with a per-ref cache under `results/entity_cache/`
(gitignored; a stall never re-bills). Once the key is in `.env`:
1. Bake-off to pick the model (~15 refs):
   `uv run --env-file .env stt-eval entity-bakeoff --specs "openrouter:anthropic/claude-opus-4.8,openrouter:google/gemini-2.5-flash" --limit 15`
2. Full build: `uv run --env-file .env stt-eval entity-build --method openrouter --model <winner> --workers 8 --datasets primock57`
3. Score + compare: does a *general* LLM also land Soniox #1, as MedGemma (specialized) did?

Candidate IDs + input $/M: `anthropic/claude-opus-4.8` ($5), `google/gemini-3.1-pro-preview`
($2), `google/gemini-2.5-flash` ($0.30). Deliberately NO OpenAI model (gpt-4o-transcribe is a
scored ASR system). Full extraction ~$2–5 (pick for quality). No medical-specialized model
exists on OpenRouter (verified) — MedGemma is the specialized route; Med-PaLM 2 is MedLM-API,
gated, unusable.

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

1. OpenRouter general-LLM foil: add `OPENROUTER_API_KEY`, then the bake-off + build above —
   the last method (MedGemma already answered the specialized side).
2. OSCE transcription: the paid checkpoint above — confirm scope before spending. Once
   transcribed, rebuild every entity manifest to include OSCE.
3. Deferred: fuzzy entity matching (current match is exact contiguous tokens; `ponytail:`
   note in entity_score.py) — would recover reference-spelling/abbreviation misses that hit
   all models equally (e.g. "flem"→phlegm, "a and e"→A&E). Deepgram + AssemblyAI ASR models
   still skipped (no keys).
