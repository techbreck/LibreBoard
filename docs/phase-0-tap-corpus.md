# Phase 0 tap-corpus preparation

`tools/prepare_tap_evaluation.py` converts consented or project-authored JSONL into the immutable
inputs replayed by the Phase 0 Android measurement harness. This preparation output is not a quality
report and cannot satisfy `tools/evaluate_engine.py` by itself.

## Source provenance

Every source JSONL has a separate manifest with exactly these fields:

- `schemaVersion`: `1`.
- `datasetId`: a stable bounded identifier.
- `dataFile` and `dataSha256`: the source leaf filename and exact lowercase SHA-256.
- `license`: `Apache-2.0`, `CC-BY-4.0`, `CC0-1.0`, or `MIT`.
- `collectionProtocol`: the documented collection/authoring procedure.
- `containsHumanContributions`: whether any row came from a person rather than project authors.
- `consentStatement`: required and non-empty for human contributions; empty for a wholly
  project-authored source.

The tool rejects a source whose hash, filename, license, human-data declaration, or consent metadata
does not match. A manifest makes provenance auditable; it does not substitute for actually obtaining
informed consent or reviewing the data before publication.

## Input rows

Every row uses schema 1 and contains `id`, `sessionId`, `category`, `target`, `raw`, `languageTag`,
`precedingContext`, `fieldClass`, `collectionMethod`, and `touchPoints`.

- Categories are `tap_error`, `valid_word`, `spacing`, and `lexical`.
- Launch languages are exactly `en-US` and `de`; permitted field classes are `plain`,
  `short_message`, and `search`.
- `collectionMethod` is `human_natural`, `human_replay`, or `project_authored`.
- Touch coordinates are normalized against the live letter-key bounds. Each point contains bounded
  finite `x`, `y`, and a monotonic relative `timeMillis`; wall-clock timestamps are forbidden.
- A `tap_error` must contain human touch data. Project-authored or empty spatial examples are
  rejected.
- `valid_word` requires `shouldCorrect`. Release data must provide 500 context-dependent
  corrections and 500 unchanged valid words.
- `lexical` requires `lexicalKind` (`contraction`, `personal`, or `compound`). A personal row also
  requires a bounded `personalWords` fixture; those fixtures must be synthetic or expressly
  consented, never silently taken from a device dictionary.

Text, contexts, points, lines, and fixture lists are hard bounded. App package names and unrestricted
surrounding prose are not part of the schema. Operators must still inspect consented context for
accidental personal information before accepting a corpus.

## Deterministic output

The committed policy assigns whole sessions to train, validation, or test with a salted hash. Raw
session and row identifiers are replaced with SHA-256 identifiers. Canonical JSONL, the source and
policy hashes, per-split/category counts, valid-word and lexical strata, license, protocol, and
consent declaration are captured in `manifest.json`.

Normal preparation fails until the held-out test split contains the plan's 3,000 tap errors, 1,000
valid-word cases, 500 spacing cases, and 500 lexical cases, with both valid-word labels and all three
lexical kinds represented; train, validation, and test must each be non-empty. `--allow-small`
exists only for tool development and leaves
`releaseEligible: false` plus the exact missing minima in the manifest.

```sh
python3 tools/prepare_tap_evaluation.py collected-taps.jsonl \
  --source-manifest collected-taps.manifest.json \
  --output-root build/evaluation-data/tap-v1
```

The prepared records still need to be replayed through stock HeliBoard, fused, personal, and neural
paths on the declared device runs. Only their artifact-bound prediction/latency JSONL is accepted by
the [Phase 0 measurement evaluator](phase-0-dataset-schema.md).

## Pinned human tap component

`models/evaluation/sources-v1.json` pins Google's CC-BY-4.0 Tap Typing with Touch Sensing Images
(TSI) dataset at an immutable Git commit, including exact sizes and SHA-256 values for its license,
documentation, keyboard geometry, prompts, and touch table. Fetching remains an explicit network
operation; later conversion and verification are offline:

```sh
python3 tools/model_sources.py --manifest models/evaluation/sources-v1.json \
  fetch google-tsi-tap-dataset-v1
python3 tools/prepare_tsi_tap_dataset.py
python3 tools/prepare_tap_evaluation.py \
  build/evaluation-sources/google-tsi-tap-v1/tap-errors.jsonl \
  --source-manifest build/evaluation-sources/google-tsi-tap-v1/source-manifest.json \
  --output-root build/evaluation-data/google-tsi-tap-v1 --allow-small
```

The adapter consumes only publisher-aligned reference characters, touch centroids, relative
timestamps, public prompts, and keyboard geometry. It deliberately ignores heatmaps, ellipse
features, submitted strings, and the publisher's language-model scores. It reconstructs complete
phrase words from non-deleted aligned touches, replays transpositions in timestamp order, normalizes
coordinates without clamping, and splits on the participant identity so one person's motor pattern
cannot cross train, validation, and test.

The pinned corpus currently yields 1,268 honest word-level spatial errors, including 236 in the
participant-disjoint test split. It is therefore useful evidence but explicitly not release-eligible
alone. Additional independently licensed human tap sources are required; duplicate or synthetic
variants must not be used to inflate the 3,000 held-out-example gate.

## Pinned noisy phone typing component

`models/evaluation/noisy-typing-v1.json` pins version 1 of Keith Vertanen and Per Ola Kristensson's
[Noisy Typing on QWERTY Keyboards](https://osf.io/5xwng/) archive: 18,052,094 bytes, SHA-256
`76d9cc798b5694686333da8baec46ce025174068bbfef909a31fe01e67dcdf5d`. The authors' OSF project
declares CC BY 4.0; the manifest records its license identifier and metadata endpoint. Attribution:
Vertanen and Kristensson, *A Dataset of Noisy Typing on QWERTY Keyboards*, IUI 2023,
[author publication page](https://www.keithv.com/pub/noisytyping/).

```sh
python3 tools/prepare_noisy_typing_dataset.py --fetch
python3 tools/prepare_tap_evaluation.py \
  build/evaluation-sources/noisy-phone-v1/tap-errors.jsonl \
  --source-manifest build/evaluation-sources/noisy-phone-v1/source-manifest.json \
  --output-root build/evaluation-data/noisy-phone-v1 --allow-small
```

Omit `--fetch` for completely offline replay. The download is versioned, byte-bounded, hash-checked
and atomically installed; preparation reads bounded ZIP entries without extracting the archive.
The adapter retains only phone `test_word` recordings from `eyesfree_exp1`, `vt_exp1`, `vt_exp2`,
`vt_exp3` and `vt_pilot2`. Pilot 1 is excluded because the publisher identifies reused participants
in Pilot 2. Development/author recordings, watches, desktop and mid-air input are excluded.

The publisher supplied force-aligned word boundaries. LibreBoard chooses exactly the first
touch-down coordinate per physical tap, replays the nearest key in the recorded keyboard geometry,
and normalizes against that layout's letter-key bounds. It does not resample motion samples into
extra taps or perturb coordinates. Ambiguous nearest keys, control-key input, out-of-bounds points,
non-monotonic timestamps, orphan input, alignment mismatches and duplicate replays are rejected and
counted. The original `LEFT`, `RIGHT`, and `ORIG` prose is discarded; emitted preceding context is empty.

The pinned conversion produces 14,995 errors: 12,090 training, 1,028 validation and 1,877 test rows
under the unchanged split policy. Conditions and sessions for each experiment-scoped participant
stay together. The publication does not provide a global cross-experiment participant map, so this
must not be described as globally participant-disjoint. The preparation report explicitly records
that limitation and `releaseEligible: false`. This tap-only source remains short of the complete
Phase 0 corpus requirements; it is not a passing quality report.
