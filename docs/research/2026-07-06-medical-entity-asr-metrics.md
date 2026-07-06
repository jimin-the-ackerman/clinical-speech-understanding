# Medical-entity-aware ASR evaluation for this harness

Research date: 2026-07-06. Scope: metrics beyond plain corpus WER that weight
clinical terms, plus the offline NER tooling to compute them over our cached
`results/transcripts/{dataset}/{model}/{file_id}.json` (`reference` + `text`).
Every factual claim links to its primary source.

## TL;DR — recommendation for this repo

- **Add a medical-term *recall* metric** (fraction of reference clinical entities
  whose surface form survives into the hypothesis). It is what PriMock57's own
  "ECCA" and Deepgram's "Keyword Recall Rate" both measure, and it degrades
  gracefully under our lossy WER normalizer. Report it *alongside* WER, not instead.
- **NER tool: [scispaCy](https://github.com/allenai/scispacy) `en_ner_bc5cdr_md`
  (DISEASE + CHEMICAL, Apache-2.0, offline, CPU, pip)**, optionally plus
  [Med7](https://github.com/kormilitzin/med7) (MIT) for drug attributes. Both are
  local models with commercial-friendly licenses. Cloud NER (AWS Comprehend
  Medical, Google Healthcare NL) is richer but violates the offline constraint —
  keep it as an optional "silver reference", exactly as the accented-speech paper
  below did.
- **Design A (entity recall)** is the low-cost landing: run NER on the *raw*
  reference (before our normalizer), push each entity's surface form through the
  same `normalize_en`, then test presence (exact, then fuzzy) in the normalized
  hypothesis. **Design B (entity-restricted WER)** reuses jiwer's alignment and
  the existing dependency for a second, harsher view. No off-the-shelf package
  does this — you build it, ~1 module.
- Key gotcha: our normalizer ([`normalize.py`](../../src/stt_eval/normalize.py) →
  vendored Whisper `EnglishTextNormalizer`) **lowercases and strips punctuation**,
  and clinical NER models are trained on cased text — so NER must run *pre*-normalization.

## Metric landscape

- **WER (baseline).** Substitutions+deletions+insertions over reference words;
  treats every token equally. What we compute today via jiwer. PriMock57 explicitly
  flags the limitation: "The base WER metric treats all words in a transcript as
  equally important; this may be less desirable in the clinical domain"
  ([Papadopoulos Korfiatis et al., ACL 2022](https://aclanthology.org/2022.acl-short.65/)).
- **ECCA — Extracted Clinical Concepts Accuracy (Precision/Recall/F1).** PriMock57's
  own entity metric: extract SNOMED-CT concepts from reference and hypothesis with a
  fuzzy-matching clinical IE engine, then score concept overlap. Reported as Pr/Re/F1
  per ASR engine ([ACL 2022, §4 + Table 3](https://aclanthology.org/2022.acl-short.65/)).
  Engine is proprietary/unreleased — the *method* is reproducible, the tool is not.
- **Medical NE Recall / M-WER / M-CER.** For accented medical speech: silver medical
  entities from Amazon Comprehend Medical, aligned to ASR output by a fuzzy matcher
  ("MedTextAlign", `difflib.SequenceMatcher` ≥0.5). Recall = exact-match proportion of
  reference entities; M-WER/M-CER = WER/CER computed over the concatenated aligned
  entity strings only ([Olatunji et al., arXiv:2406.12387](https://arxiv.org/abs/2406.12387)).
- **Keyword Error Rate (KER) / Keyword Recall Rate (KRR).** Vendor (Deepgram) metrics:
  KER = "how often key medical terms are either missed or incorrectly transcribed";
  KRR = "how reliably the AI captures specialized language". No formula or keyword
  list is published ([Deepgram, "Introducing Nova-3 Medical"](https://deepgram.com/learn/introducing-nova-3-medical-speech-to-text-api)).
- **MTRA — Medical Term Recognition Accuracy.** MedDialog-Audio reports WER + MTRA +
  BERTScore-F1 across noise levels ([Gassenn et al., SBBD DSW 2025](https://sol.sbc.org.br/index.php/dsw/article/download/37199/36984/)).
  The paper names MTRA and shows results but the accessible text does **not** give its
  formula or term source (see Open questions).
- **ATENE.** A probabilistic metric estimating the risk that ASR errors induce
  downstream named-entity-detection errors; correlates with NER performance better
  than WER ([Ben Jannet et al., Interspeech 2015](https://www.isca-archive.org/interspeech_2015/jannet15_interspeech.html)).
  General-domain, not clinical; heavier to implement than recall.

## NER tooling comparison

| Tool | License | Offline? | Install weight | Entity types | Needs truecase/punct? | Primary source |
|---|---|---|---|---|---|---|
| scispaCy `en_ner_bc5cdr_md` | Apache-2.0 | Yes (local) | pip; pins `spacy>=3.7,<3.9` (v0.6.2) | DISEASE, CHEMICAL | Trained on cased PubMed (BC5CDR); run pre-norm | [repo](https://github.com/allenai/scispacy), [PyPI](https://pypi.org/project/scispacy/), [paper](https://arxiv.org/abs/1902.07669) |
| scispaCy `en_ner_bionlp13cg_md` | Apache-2.0 | Yes | as above | 16 cancer-genetics types (gene, cell, tissue…) | as above | [repo](https://github.com/allenai/scispacy) |
| scispaCy `en_core_sci_{sm,md,lg}` | Apache-2.0 | Yes | as above | untyped `ENTITY` spans (mention detection only) | as above | [repo](https://github.com/allenai/scispacy) |
| Med7 (`en_core_med7_lg/trf`) | MIT | Yes | pip + spaCy; MIMIC-III-pretrained | DRUG, DOSAGE, DURATION, FORM, FREQUENCY, ROUTE, STRENGTH | Cased clinical notes | [arXiv:2003.01271](https://arxiv.org/abs/2003.01271), [repo](https://github.com/kormilitzin/med7) |
| medspaCy | MIT | Yes | pip; spaCy v3 (≤3.8.2) | rule-based `TargetRule` (you supply dictionaries) + ConText negation/sectionizer; no pretrained statistical NER | Rules can be case-insensitive | [repo](https://github.com/medspacy/medspacy) |
| Stanza clinical (i2b2, radiology) | Apache-2.0 (models derive from MIMIC-III LMs) | Yes | pip + downloaded models | i2b2: PROBLEM/TEST/TREATMENT; radiology: ANATOMY/OBSERVATION/… | Cased clinical notes | [models](https://stanfordnlp.github.io/stanza/available_biomed_models.html), [repo](https://github.com/stanfordnlp/stanza) |
| QuickUMLS | MIT (code) | Yes, *after* setup | needs a local UMLS Metathesaurus install (NLM license required) | any UMLS semantic type (SNOMED/RxNorm/…) | Approximate dict-match; case-tolerant | [repo](https://github.com/Georgetown-IR-Lab/QuickUMLS) |
| AWS Comprehend Medical | Commercial cloud API (HIPAA-eligible) | **No — cloud only** | boto3 call, per-char pricing | ANATOMY, MEDICAL_CONDITION, MEDICATION, TEST_TREATMENT_PROCEDURE, BEHAVIORAL_ENVIRONMENTAL_SOCIAL, PHI, TIME_EXPRESSION | Robust to informal text | [docs](https://docs.aws.amazon.com/comprehend-medical/latest/dev/textanalysis-entitiesv2.html) |
| Google Healthcare NL API | Commercial cloud API | **No — cloud only** | REST/client lib | medications, conditions, procedures, anatomy + context (negation, temporality) | Robust | [Google Cloud](https://cloud.google.com/blog/topics/healthcare-life-sciences/now-in-preview-healthcare-natural-language-api-and-automl-entity-extraction-for-healthcare) |

Notes: scispaCy releases pin an exact spaCy range and the *model wheels are
version-matched to the scispaCy release* — a real install constraint ([PyPI
requires_dist](https://pypi.org/project/scispacy/)). Med7 and the scispaCy `en_ner_*`
models are the only fully-offline, commercial-license, typed-clinical-NER options
that emit the disease/chemical/drug types our GP-consultation datasets care about.

## Application designs

Shared prerequisite: **NER runs on the raw reference string** (pre-normalization),
because [`score.py`](../../src/stt_eval/score.py) lowercases + de-punctuates via the
Whisper normalizer and case-sensitive models lose accuracy on that. Then normalize
each extracted entity's surface form with the *same* `normalize_en` so it matches the
already-normalized hypothesis. All three run offline over the cached JSONs.

**A. Entity recall (recommended first).** For each file: `entities = NER(raw ref)`;
for each entity, `hit = normalize_en(surface) in normalize_en(hyp)` (substring/token
match; add fuzzy `SequenceMatcher ≥ τ` for spelling variants as in
[MedTextAlign](https://arxiv.org/abs/2406.12387)). Aggregate hits / total entities per
(model, dataset, condition). *Measures:* did critical terms survive — the ECCA/KRR
question. *Failure modes:* synonyms/abbreviations ("MI" vs "myocardial infarction")
miss on surface match; a hallucinating model is **not** penalized (recall ignores
false positives — that is why it pairs with WER). *Effort:* ~1 module + one dep
(scispaCy); no change to WER path.

**B. Entity-restricted WER.** Reuse `jiwer.process_words(ref, hyp)`, which returns
`AlignmentChunk(type, ref_start_idx, …)` word alignments
([jiwer docs](https://jitsi.github.io/jiwer/usage/), Apache-2.0). Mark reference token
indices that fall inside NER spans; count only substitutions/deletions touching those
indices, divided by entity-token count. *Measures:* error *rate* within entities
(mirrors M-WER). *Failure modes:* token-boundary mismatch after normalization; needs
entity char-spans mapped to normalized token indices — fiddlier than A. *Effort:*
moderate; reuses the existing jiwer dependency, adds NER.

**C. Off-the-shelf package.** None found. jiwer / HuggingFace `evaluate` compute only
the WER family ([jiwer](https://github.com/jitsi/jiwer);
[evaluate wer wraps jiwer](https://github.com/huggingface/evaluate/blob/main/metrics/wer/wer.py));
NIST SCTK/sclite (used by PriMock57 for WER) has keyword options but no clinical-entity
scoring. Published entity-ASR metrics ship as paper code, not maintained libraries.
*Conclusion:* build A (and optionally B) in-repo; don't wait for a package.

## Prior art on our datasets

- **PriMock57** ([Papadopoulos Korfiatis, Moramarco, Sarac, Savkov, ACL 2022](https://aclanthology.org/2022.acl-short.65/)):
  benchmarks Google/Azure/Amazon Transcribe Medical/Kaldi/QuartzNet/Conformer with
  **WER via SCTK sclite** *and* **ECCA** (Extracted Clinical Concepts Accuracy,
  Pr/Re/F1) from a proprietary SNOMED-CT fuzzy-matching engine. Their finding is
  directly relevant to us: ECCA "mostly match[es] the WER comparisons; the
  medical-domain Amazon model does not seem to perform better." So an entity metric is
  worth having but may correlate with WER on these data — validate, don't assume it
  reveals something new. Their post-processing (drop disfluencies, spell out numerals,
  strip punctuation, lowercase) is close to our normalizer.
- **MedDialog(-Audio)**: the source text corpus MedDialog is a *dialogue-generation*
  dataset evaluated with NLG metrics (BLEU, Distinct), **no ASR/entity metric**
  ([Zeng et al., EMNLP 2020 Findings](https://aclanthology.org/2020.emnlp-main.743/)).
  The audio derivative introduces the entity angle: **WER + MTRA + BERTScore-F1** across
  white/background-noise levels, on Whisper/Wav2Vec2/HuBERT
  ([Gassenn et al., SBBD DSW 2025](https://sol.sbc.org.br/index.php/dsw/article/download/37199/36984/);
  [dataset](https://huggingface.co/datasets/aline-gassenn/MedDialog-Audio)).

## Open questions

- **MTRA formula unverified.** MedDialog-Audio names and plots MTRA but the accessible
  PDF text does not define its computation or the term list/NER behind it — treat as
  "a medical-term recognition accuracy" only until confirmed with the authors.
- **Deepgram KER/KRR are undefined publicly** — no formula or keyword inventory
  ([source](https://deepgram.com/learn/introducing-nova-3-medical-speech-to-text-api)),
  so their headline numbers are not reproducible or directly comparable to ours.
- **Case-robustness magnitude unverified.** That scispaCy/Med7/Stanza train on cased
  text is documented; how much they degrade on our lowercased+de-punctuated strings is
  not measured in any primary source I found — the "run NER pre-normalization"
  recommendation sidesteps it rather than quantifies it.
- **Synonym/abbreviation matching:** exact/fuzzy surface matching misses UMLS synonymy;
  closing that needs a concept linker (QuickUMLS/scispaCy `UmlsEntityLinker`), which
  adds a UMLS license and weight — decide whether recall-on-surface-form is enough.
