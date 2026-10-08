# Phase 0 measurement schema

`tools/evaluate_engine.py` is the release-gate evaluator. It accepts one JSON object per line with
these fields:

- `schemaVersion`: `4`.
- `id`: stable, unique example ID.
- `sessionId`: collection session; a session may occur in exactly one split.
- `split`: `train`, `validation`, or `test`.
- `environmentKind` and `testRunId`: required for every `test` row and forbidden for training or
  validation rows. The kind is one of `stock_android_hardware`, `grapheneos_hardware`, or
  `low_ram_emulator`; the run ID must exactly match the corresponding metadata record.
- `category`: `tap_error`, `valid_word`, `spacing`, `lexical`, or `swipe`.
- `target` and `raw`: expected and observed text.
- `predictions`: ranked strings for every applicable system, capped at 32, in the order production
  ranks them. Production seats the typed word first, so this list is not the scoring order.
- `commits`: required on measured tap rows and forbidden everywhere else. One record per tap system,
  exactly `{"committed": <surface>, "willAutoCorrect": <bool>}`, describing what the editor actually
  receives when the word is terminated. A keep must commit `raw` verbatim; a correction must commit
  something else; either way the committed surface must appear in that system's own slate.
- `latencyMs`: end-to-end measurement for every applicable system.
- `strata`: swipe labels. Release evidence must include at least 500 samples in each of `short`,
  `medium`, `long`, `clean`, `sloppy`, `very_sloppy`, `double_letter`, and `return_trip`; labels may
  overlap.
- `lexicalKind`: required for `lexical` rows, exactly `contraction`, `personal`, or `compound`;
  forbidden for every other category. Release evidence must include each kind in the held-out split,
  in addition to the combined 500-case lexical minimum. The report retains these as `lexicalCounts`.
- `shouldCorrect`: required for `valid_word`; false labels measure false corrections. It must agree
  with normalized raw/target equality: unchanged words are keeps and changed words are corrections.

Release-sized evidence must contain 500 context-dependent valid-word corrections
(`shouldCorrect: true`) and 500 valid words that must remain unchanged (`shouldCorrect: false`). The
15% neural relative-error-reduction gate is calculated only over the former, while the five-point
valid-word gain uses both and the false-correction ceiling uses the latter. This prevents easy
unchanged words from diluting the context-sensitive neural test.

Tap systems are `heliboard`, `fused`, `fused_personal`, and `fused_neural`. Swipe systems are
`geometric`, `ctc`, and `fused_swipe`.

Measured rows must contain exactly their applicable prediction and latency systems; tap and swipe
results cannot be mixed in one row. Candidate slates must be normalization-distinct, identity/text
fields and JSONL lines are bounded, duplicate strata are rejected, and `shouldCorrect` cannot be
attached to any category other than `valid_word`.

Every measured LibreBoard tap slate (`fused`, `fused_personal`, and `fused_neural`) must contain the
exact `raw` surface, including capitalization and punctuation. The evaluator rejects a report that
cannot prove the one-tap raw-word fallback; normalization is used only for accuracy scoring. It does
not require that surface to come first, and a harness must not hoist it: production already places
the typed word at index 0.

### How tap rows are scored

Tap top-1 is the `commits` decision, not the head of `predictions`. This is the whole reason schema 4
exists. Production always seats the typed word ahead of a pending correction — `InputLogic` picks
`INDEX_OF_AUTO_CORRECTION` only when `SuggestedWords.mWillAutoCorrect` is set, and otherwise commits
the typed word — so a harness that ranks `raw` first and an evaluator that scores rank one together
report the typed text back to itself. Every schema-3 measurement did exactly that: all four tap
systems scored an identical 25.265% overall top-1 and 0% on tap errors, which measured the harness
rather than the keyboard. Schema-3 tap results cannot be repaired by rescoring, because the rows
never recorded the commit decision; they must be re-measured.

Tap top-3 is the committed surface together with the first three strip entries, which is what a user
reaches without and with one extra tap. The false-correction rate counts rows whose committed surface
differs from `raw`, so merely ranking a candidate above the typed word is not a false correction.
Swipe rows have no keep-or-correct decision and are still scored by `predictions` rank.

Metadata is a JSON object that binds the report to the artifacts and required environments:

- `schemaVersion`: `4`.
- `appCommit`: the full lowercase Git commit tested.
- `coreApkSha256`, `swipeModelSha256`, and `contextModelSha256`: lowercase SHA-256 values for the
  exact APK and both models used for every reported prediction.
- `artifactPin`: exactly `candidateId`, `contextModelSha256` and `swipeModelSha256`, naming the model
  candidate this evidence claims to qualify. Its hashes must equal the metadata's own model hashes,
  so a report cannot silently qualify a different candidate than the one it measured. The evaluator
  copies it into the report's evidence. `tools/run_phase0_measurement.py --artifact-pin` writes it.
- `environments`: exactly one `stock_android_hardware`, one `grapheneos_hardware`, and one
  `low_ram_emulator` record. Every record includes `deviceModel`, `buildFingerprint`, `testRunId`,
  `apiLevel`, `physicalDevice`, and its measured `peakAddedNeuralMemoryMiB`. Hardware must be
  physical and run Android 15 or newer. The
  GrapheneOS record also includes `grapheneOsBuildNumber` and
  `sandboxedGooglePlayInstalled: false`. The emulator includes `isLowRamDevice: true` and a
  `memoryMiB` value no greater than 2048.

The evaluator copies this normalized evidence into the output report. Missing device classes,
placeholder artifact identifiers, a non-physical GrapheneOS run, or a GrapheneOS run with sandboxed
Google Play cannot produce a passing report. Run IDs must be unique. Release evidence must bind at
least 100 tap and 100 swipe measurements to each environment; the 80/200 ms p95 gates are evaluated
independently for every environment, so fast stock-device samples cannot hide a slow or absent
GrapheneOS run. The report exposes these values as `environmentCounts` and
`environmentLatencyMs`; the valid-word strata are exposed as `validWordCounts`. It also computes
the report-level `peakAddedNeuralMemoryMiB` as the maximum of the three bound environment values, so
one device's memory result cannot stand in for the matrix. It computes
`measurementDatasetSha256` directly from the input JSONL and places that hash in the report; callers
cannot supply or override it.

```sh
python3 tools/evaluate_engine.py measurements.jsonl \
  --metadata measurement-metadata.json \
  --swipe-lexicon build/device-evidence/static-swipe-lexicon.json \
  --swipe-lexicon-apk path/to/measured.apk \
  --report build/reports/phase-0.json
```

`--swipe-lexicon` is the instrumented production vocabulary export. `--swipe-lexicon-apk` is the
measured APK: its SHA-256 must equal metadata `coreApkSha256`, and its dictionary asset must be
byte-identical to the one the export enumerated. Swipe accuracy gates score targets in that
vocabulary, and without both options they fail (see the gate amendments in
[Phase 0](phase-0.md)). `--context-model unavailable` evaluates a release that ships without the
context model; the default, `qualifying`, keeps every context-model gate.

The normal command enforces the release-size dataset minimums. `--allow-small-dataset` exists only
for developing the evaluator and cannot produce release evidence. A report passes only when every
quality, false-correction, latency, memory, and swipe-stratum gate from the product plan passes.
Final release verification requires both the report and its raw JSONL, recomputes every metric and
requires the result to match the report exactly.

Schema 4 adds the mandatory per-system `commits` decision and stops treating slate rank one as the
tap result. Schema 3 measurements and reports must be re-measured on device; the commit decision they
would need cannot be recovered from their stored rows. Schema 3 added mandatory lexical kinds and
coverage. Schema 2 measurements and reports must be regenerated with source-backed labels; missing
kinds must not be inferred merely to pass the gate.

## Artifact pin

`docs/phase-0-artifact-pin.json` names the exact model candidate a qualifying run may measure, and
`tools/run_phase0_measurement.py --artifact-pin` enforces it: the injected bytes are hashed on the
host and again on the device, and an artifact listed in `rejectedContextModelSha256` fails the run by
name. The driver also copies the candidate into metadata as `artifactPin`, and the evaluator refuses
metadata without one, so a passing report always states which candidate it qualified. Runs without a
pin are diagnostics and cannot produce a report. The driver also verifies the APK installed on the device
against the hash bound into metadata, requires every gated instrumented test to run and pass, and
checks that every pulled row carries the run's own id and environment before saving anything.
