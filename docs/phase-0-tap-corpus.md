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

Additional source screening on 2026-09-08 excluded the University of Strathclyde's
[OATS highlighting study](https://pureportal.strath.ac.uk/en/datasets/oats201411-highlighting-keyboard-study-2/)
because its files are restricted for data protection and require an access discussion. The
[MobileStress publisher](https://psi.engr.tamu.edu/mobilestress/) specifies CC BY-NC 4.0 and an
access-request procedure, so it cannot supply this project's openly licensed corpus component.
Neither source was imported or counted.

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

## Pinned noisy smartwatch typing component

`models/evaluation/noisy-watch-v1.json` pins the same reviewed OSF archive under a second dataset
identity, `vertanen-noisy-watch-v1`, with a disjoint experiment scope: `comp_exp1`, `comp_exp2`,
`impact_exp1`, `impact_exp2`, `vw_exp1`, `vw_exp3` and `vw_exp4`. These Sony Smartwatch3 recordings
carry real letter/apostrophe-key touch coordinates and are the corpus's second spatial source.

```sh
python3 tools/prepare_noisy_watch_typing.py
```

The adapter replays first touch-down samples against the recorded `keyboard_smartwatch3.txt`
geometry with the same nearest-key, bounds, monotonic-time and dedup rules as the phone adapter.
Impact experiments use the publisher's force-aligned `test_word` splits plus the native `word`
condition; VelociWatch and composition `test` logs are natively word-aligned. Practice, dev,
sentence/twoword/copy/compose pre-alignment logs, empty recordings and per-sample lock/confidence
suffixes are excluded or stripped. The apostrophe key is parsed but replays to it are rejected as
control-key input; negative or non-finite timestamps are counted rejects, not normalized.

The conversion yields 5,072 tap errors from 141 experiment-scoped sessions — 1,349 in the held-out
test split. These are watch-form-factor spatial errors, a declared limitation in the component
report; participant identities remain experiment-scoped.

## Pinned English split/join spacing component

`tools/prepare_ite_spacing.py` reuses the pinned ITE selection (`test_sections_labeled.csv` joined
to `sentences.csv`) to extract word split/join sites: token-alignment blocks where the
participant's letter tokens regroup the reference tokens without changing the concatenated
letters — `thehospital` for `the hospital`, `inter company` for `intercompany`. Only all-letter
token groups qualify; punctuation-adjacent regroupings, substitutions, insertions and deletions are
not spacing rows. Each row carries the participant's own typed prefix as bounded context, the
`ite-en:` participant session identity shared with the other ITE components so one person cannot
cross splits, and deduplication on participant/context/raw/target.

```sh
python3 tools/prepare_ite_spacing.py
```

The conversion yields 2,573 spacing rows — 509 in the held-out test split. The source is
keystroke-level; no touch coordinates are claimed.

## Project-authored lexical component

`tools/prepare_lexical_component.py` emits deterministic (seeded) `project_authored` rows for the
two lexical strata no human source supplies. Personal targets are invented names generated from a
fixed syllable list and carried in each row's `personalWords` fixture — never real personal data.
Compound targets are common English compounds; raw text is an adjacent-key typo or a split-form
variant (`news paper` for `newspaper`). Contexts are fixed synthetic prompt fragments.

```sh
python3 tools/prepare_lexical_component.py
```

The component produces 1,440 personal and 1,440 compound rows (360 and 312 held out). Its manifest
declares `containsHumanContributions: false` and no consent claim; synthetic rows measure
fixture/compound handling, not human lexical behavior.

## Pinned English valid-word component

`models/evaluation/ite-valid-words-v1.json` pins selected files from the CC-BY-4.0
[ITE Typing dataset, version 1](https://zenodo.org/records/12528163) by Katri Leino, Markku Laine,
Mikko Kurimo and Antti Oulasvirta (2024). Its license is recorded by the
[Zenodo metadata API](https://zenodo.org/api/records/12528163). The adapter downloads only three
English ZIP members using bounded HTTP ranges and checks each extracted file's SHA-256 and byte
count. The archive MD5 is publisher metadata; partial fetching does not claim to verify the entire
7.3 GB archive. Later conversion is fully offline.

First export the native static vocabulary using the Android procedure in
[swipe evaluation](swipe-ctc-evaluation.md#native-dictionary-validation-diagnostic). The dictionary
asset hash and canonical native vocabulary hash are pinned independently of the APK build hash.
Then run:

```sh
python3 tools/prepare_ite_valid_words.py --fetch
python3 tools/prepare_tap_evaluation.py \
  build/evaluation-sources/ite-valid-words-v1/valid-words.jsonl \
  --source-manifest build/evaluation-sources/ite-valid-words-v1/source-manifest.json \
  --output-root build/evaluation-data/ite-valid-words-v1 --allow-small
```

The adapter uses the publisher's pre-autocorrection `TYPED_WORD`, public reference word and actual
preceding input. It never uses the `AC_WORD` output as a label. Both words must be in the pinned
non-offensive static English vocabulary. The input must end in the typed word, its preceding prefix
must match the reference prompt, and the intended word must match the exact following reference
position. It retains only bounded reference-verified context and discards submitted sentence text,
autocorrection outputs, demographic metadata and device metadata. Participants, including all their
sessions, stay in one salted split; repeated participant/prefix/raw/target cases are deduplicated.

The conversion produces 54,519 real-word examples. The unchanged held-out split contains 6,132
corrections and 4,635 keep cases, exceeding both 500-case valid-word minima. These are text-only
examples, not touch-coordinate evidence. They are also selected from publisher-detected
autocorrection events, so they must not be described as an unbiased sample of ordinary typing.
Publisher event labels were inferred automatically; strict alignment filters reduce ambiguity but do
not turn this into a manually reviewed corpus. The component remains `releaseEligible: false`:
it does not supply the missing spatial, spacing, personal/compound, device or quality evidence.

The same pinned ITE files also support `tools/prepare_ite_contractions.py`. It uses an explicit
English contraction list, excludes noun possessives, and applies exact prefix/reference alignment.
It yields 16,406 human text-only contraction cases, including 3,244 held-out cases. Its session IDs
are deliberately identical to the valid-word adapter's participant IDs so the same person cannot
cross partitions when the components are combined. Personal words and compounds remain absent.

```sh
python3 tools/prepare_ite_contractions.py
```

## Combining verified components

`tools/merge_tap_evaluation_sources.py` validates every source hash and row, preserves original IDs,
labels and session grouping, rejects duplicate row/dataset IDs, and requires one common license.
It writes all original source manifests into a hash-bound provenance sidecar. It does not invent
consent metadata, relabel cases, or make the merged dataset release-eligible.

```sh
python3 tools/merge_tap_evaluation_sources.py \
  --component build/evaluation-sources/google-tsi-tap-v1/source-manifest.json \
  --component build/evaluation-sources/noisy-phone-v1/source-manifest.json \
  --component build/evaluation-sources/noisy-watch-v1/source-manifest.json \
  --component build/evaluation-sources/ite-valid-words-v1/source-manifest.json \
  --component build/evaluation-sources/ite-contractions-v1/source-manifest.json \
  --component build/evaluation-sources/ite-spacing-v1/source-manifest.json \
  --component build/evaluation-sources/project-lexical-v1/source-manifest.json \
  --output-root build/evaluation-sources/combined-tap-v2
python3 tools/prepare_tap_evaluation.py \
  build/evaluation-sources/combined-tap-v2/tap-cases.jsonl \
  --source-manifest build/evaluation-sources/combined-tap-v2/source-manifest.json \
  --output-root build/evaluation-data/combined-tap-v2
```

The 2026-09-11 seven-component merge contains 97,713 rows; ITE participant groups retain one shared
split across the three ITE components. Its held-out corpus is:

| Category | Held-out rows | Data gate |
| --- | ---: | --- |
| Human spatial tap errors | 3,462 | Minimum met (2,113 phone + 1,349 watch) |
| Valid-word corrections | 6,132 | Count minimum met |
| Valid-word keeps | 4,635 | Count minimum met |
| Contractions / personal / compound | 3,244 / 360 / 312 | Lexical total met; all three kinds represented |
| Split/join cases | 509 | Minimum met |

The prepared corpus reports `releaseEligible: true` for the data gates. This closes the corpus
count requirements only: it is not model-quality evidence, and the Phase 0 measurement harness,
device matrix, and latency/memory gates remain open. The spacing stratum sits nine rows above its
minimum and the personal/compound strata are synthetic; both are declared limitations, not
equivalent human coverage.
