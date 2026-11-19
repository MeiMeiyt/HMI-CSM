# HMI-CSM: manuscript-size experimental code

This package organizes the author's existing semantic mapping, uncertainty detection,
evidence-constrained repair, baseline, and evaluation scripts into one pipeline.
See [README_CN.md](README_CN.md) for a Chinese quick start.

## Scope of this release

| Dataset | Prepared evaluation size | Selection |
| --- | ---: | --- |
| NaLiBaSID | 1,212 utterances | Complete designated Bavarian test file, in its original order |
| MedNorm | 10,000 records | Seeded sampling without replacement from eligible single-SNOMED-target rows |
| MediTOD | 200 dialogues | Seeded sampling of eligible dialogues; retain at most 30 utterances each |
| KddRES | 200 dialogues | Seeded sampling of eligible dialogues; retain at most 30 utterances each |

One turn in these scripts is one speaker utterance, not a question-answer pair.
A dialogue is eligible if at least four utterances and two speakers remain after
truncation. Duplicate dialogue content is excluded before sampling. There is no
replacement sampling or silent reduction when a dataset is too small.

The default preparation seed is `20261015`. It is an explicit release default,
not a claim about the seed used for previously reported results. This package
does not contain a new full-size API experiment or claim to reproduce the numerical
tables without rerunning the methods on the actual selected data.

## Data sources and local layout

Obtain the original datasets from their providers and follow their access and reuse terms:

- NaLiBaSID: <https://github.com/mainlp/NaLiBaSID>
- MedNorm: <https://doi.org/10.17632/b9x7xxb9sz.1>; pipeline repository:
  <https://github.com/mbelousov/MedNorm-corpus>
- MediTOD: <https://github.com/dair-iitd/MediTOD>
- KddRES: <https://github.com/hrwise-nlp/KddRES>

Raw datasets, populated credential files, generated predictions, and private source
copies are not included. The intended paths are:

| Configuration field | Default local path or required contents |
| --- | --- |
| `stage1.dialect_test` | `data/de-ba.MAS.test` |
| `stage1.dialect_validation` | `data/de-ba.MAS.valid` |
| `stage1.mednorm_file` | `data/mednorm_full.tsv` |
| `stage2.meditod_directory` | `data/MediTOD/`, with `dialogs.json` under this directory |
| `stage2.kddres_directory` | `data/KddRES/`, with `data_zh_hant/` containing available `test.json`, `val.json`, and/or `train.json` files |

Edit `paper_config.json` if your paths differ. Relative paths are resolved against
the package root. The NaLiBaSID files must be extracted according to the original
provider's instructions. The parser accepts blank-separated CoNLL records with
`# intent: ...` or `# intent = ...` metadata, token text in column 2 and BIO labels
in column 4, or a two-column token/BIO format. It checks that the supplied test
file parses to exactly 1,212 records.

The KddRES loader gathers available splits in a stable order, then samples eligible
dialogues. This is a dialogue sample from the available corpus; it is not presented
as an official 200-dialogue held-out split. No repair stage reads future utterances
beyond the current case's response turn.

## Requirements

Python 3.10 or newer. The default pipeline uses the Python standard library;
no OpenAI SDK is required. Optional plots in `part3compared.py` require matplotlib
if `DRAW_FIGURES` is enabled. Plots are disabled by default.

## 1. Inspect and prepare without model calls

From the extracted package directory:

```console
python run_experiments.py inspect
python run_experiments.py prepare
```

Preparation does not call a model. It writes fixed shared inputs under `prepared/`,
including sample IDs, selection metadata, SHA-256 file/content fingerprints, and
candidate-gold recall for diagnosis. All methods reuse these prepared inputs.
Preparation is deterministic for identical input files and configuration.

For separate preparation stages:

```console
python run_experiments.py prepare1
python run_experiments.py prepare2
```

MedNorm preparation is more expensive than simple row sampling because it also
builds candidate lists using the original string-similarity ranking.

### MedNorm selection and candidate references

The earlier small-data helper restricted input to CADEC and selected one held-out
mention per concept group. Merely increasing its sample limit would not implement
the manuscript's 10,000-record sample. This release reads all eligible MedNorm
sources, samples 10,000 valid records without replacement, and records the original
TSV row numbers and source distributions.

By default, candidate aliases are drawn from unselected MedNorm rows, with **every
evaluated surface form excluded** from the reference aliases and few-shot examples.
The target answer is not inserted into a candidate list. Gold labels are kept only
for evaluation and candidate-recall diagnostics, and are not sent to the model.

For a separately prepared reference catalog, set `stage1.term_reference_catalog`
to its JSON path. Supported forms are an ID-to-aliases dictionary or a list such as:

```json
[
  {"id": "sct:YOUR_CONCEPT_ID", "aliases": ["reference synonym", "another synonym"]}
]
```

Use an appropriate non-evaluation reference source. Evaluated surface forms are
excluded even from an external catalog. The default candidate limit is 15;
candidate recall is reported rather than assumed to be 100%.

If you already have the exact full-size stage-1 manifest used in your experiments,
set `stage1.existing_manifest` to its JSONL path. This retains those cases, candidate
lists and reference annotations instead of resampling them. The manifest must
already contain exactly 1,212 `dialect` and 10,000 `term` cases in the schema used
by the original scripts. A small trial manifest is rejected.

Existing prepared inputs cannot silently change. `prepare --overwrite` replaces
generated selections only. After changing data, candidate settings or sample
selections, use a new run ID for the results.

## 2. Configure your provider through environment variables

`.env.example` documents variable names. It contains no working credentials and
is **not automatically loaded**. Do not place a real key in a Python file or commit
a populated `.env` file.

For example, in Windows PowerShell:

```powershell
$env:HMI_API_KEY = Read-Host "Enter your API key"
$env:HMI_BASE_URL = "https://YOUR_OPENAI_COMPATIBLE_HOST/v1"
$env:HMI_MODEL = "YOUR_ACTUAL_PROVIDER_MODEL_ID"
```

`HMI_BASE_URL` is the base preceding `/chat/completions`. The transport supports an
OpenAI-compatible chat-completions interface. It does not implement each vendor's
distinct native API. Configure the actual provider endpoint and exact model ID
used in your experiment.

The paper's display names (DeepSeek-V4-Pro, GPT-6.1 Sol, Claude Sonnet 5.5) are not
hardcoded as guessed API IDs. Run each chosen model separately with its real
provider configuration and a different `--run-id`.

Optional variables `HMI_DETECTOR_MODEL`, `HMI_DETECTOR_BASE_URL`,
`HMI_DETECTOR_API_KEY`, and the corresponding `HMI_EVAL_*` variables configure the
uncertainty detector and evaluation judge independently. Unset overrides fall
back to the main model configuration. The run metadata records all three actual
model IDs and endpoints, never API keys. Use the same evaluation configuration
across compared methods when designing a controlled comparison.

`HMI_TIMEOUT` defaults to 120 seconds; `HMI_RETRIES` defaults to 3. API error bodies
are omitted from saved errors, and credentials are excluded from object displays.

## 3. Execute the stages

```console
python run_experiments.py all --execute --run-id model-a
```

`--execute` enables real execution and can incur provider charges. Without it,
`all` only prints the plan. The complete sequence is:

| CLI stage | Script | Work |
| --- | --- | --- |
| `traditional` | `part2Traditional.py` | Lexicon and character 2-4-gram TF-IDF baselines; no model call |
| `map` | `part1Mapping.py` | HMI-CSM unified semantic mapping |
| `baselines1` | `part2Baseline.py` | Direct, Normalize and Few-shot mapping |
| `evaluate1` | `part3compared.py` | Programmatic stage-1 evaluation on the same cases |
| `map-dialogues` | `part4CognitiveMapping.py` | Map all retained utterances in the selected dialogues |
| `detect` | `part5Cognitive.py` | Attribute-level multilabel uncertainty detection |
| `repair` | `part6Repair.py` | Evidence-constrained HMI-CSM repair proposals |
| `baselines2` | `part7Baseline.py` | Direct, Context and Retrieval repair baselines |
| `evaluate2` | `part8Evaluate.py` | Model-assisted evaluation plus programmatic evidence checks |

Each stage can be run separately, with the same run ID and configuration:

```console
python run_experiments.py map --execute --run-id model-a
python run_experiments.py baselines1 --execute --run-id model-a
python run_experiments.py evaluate1 --execute --run-id model-a
```

Run prerequisite stages first. Results are saved under
`results/<run-id>/stage1/` and `results/<run-id>/stage2/`. Stage-1 summaries are in
`stage1/summary.csv`; stage-2 summaries are in the evaluation directory selected
by `part8Evaluate.py`. Internal `ours_v2`/`v4` filenames are retained for compatibility
with the original pipeline; they are not additional comparison methods.

Generative outputs support checkpoint reuse. Matching successful stage-1 cases
are reused, failed cases are retried, and per-case writes preserve progress on
interruption. Prompt/input fingerprints and separate run directories prevent
unintentional reuse across models or data configurations. A run ID with different
recorded model settings is rejected. Do not run two processes into the same
run directory simultaneously.

## Evaluation and interpretation

The original mapping/repair prompts, semantic validators, and metric calculations
are retained. The main changes are credential removal, unified configuration,
manuscript-size input preparation, repaired script interfaces, shared manifests,
and run/checkpoint handling. The six stage-1 methods use one fixed manifest;
all four stage-2 repair methods use the same detected attribute observations.

Stage-1 labels are dataset references; stage-2 uncertainty labels and repair
judgments are model-assisted assessments with evidence validation. They are not
independent human gold annotations. The number of detected repair cases is
determined by the selected dialogues and detection outputs, not fixed to the
number of dialogues or to a former trial count.

Changing the random selection, reference catalog, model ID or evaluation judge
can change the results. Retain the prepared manifests, configuration metadata
and actual predictions for the final experiments. This package does not insert
the manuscript's reported scores or create experimental results from test fixtures.

## Offline verification

```console
python -m unittest discover -s tests -v
```

The tests use explicitly synthetic fixtures and mocked model responses. They check
full-size preparation, fixed selection, strict shortage handling, reference alias
separation, checkpoint recovery, all six stage-1 output/evaluation interfaces, and
credential handling. No provider API is contacted by these tests.

`source_provenance.json` records the names and SHA-256 hashes of the original
scripts used for this cleanup, without embedding their contents or credentials.
`data_availability.tex` contains a manuscript-ready draft referring to the original
dataset sources and code availability from the corresponding author upon request.

