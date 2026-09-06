# Phase 0 measurement schema

`tools/evaluate_engine.py` is the release-gate evaluator. It accepts one JSON object per line with
these fields:

- `schemaVersion`: `2`.
- `id`: stable, unique example ID.
- `sessionId`: collection session; a session may occur in exactly one split.
- `split`: `train`, `validation`, or `test`.
- `environmentKind` and `testRunId`: required for every `test` row and forbidden for training or
  validation rows. The kind is one of `stock_android_hardware`, `grapheneos_hardware`, or
  `low_ram_emulator`; the run ID must exactly match the corresponding metadata record.
- `category`: `tap_error`, `valid_word`, `spacing`, `lexical`, or `swipe`.
- `target` and `raw`: expected and observed text.
- `predictions`: ranked strings for every applicable system, capped at 32.
- `latencyMs`: end-to-end measurement for every applicable system.
- `strata`: swipe labels. Release evidence must include at least 500 samples in each of `short`,
  `medium`, `long`, `clean`, `sloppy`, `very_sloppy`, `double_letter`, and `return_trip`; labels may
  overlap.
- `shouldCorrect`: required for `valid_word`; false labels measure false corrections.

Release-sized evidence must contain 500 context-dependent valid-word corrections
(`shouldCorrect: true`) and 500 valid words that must remain unchanged (`shouldCorrect: false`). The
15% neural relative-error-reduction gate is calculated only over the former, while the five-point
valid-word gain uses both and the false-correction ceiling uses the latter. This prevents easy
unchanged words from diluting the context-sensitive neural test.

Tap systems are `heliboard`, `fused`, `fused_personal`, and `fused_neural`. Swipe systems are
`geometric`, `ctc`, and `fused_swipe`.

Every measured LibreBoard tap slate (`fused`, `fused_personal`, and `fused_neural`) must contain the
exact `raw` surface, including capitalization and punctuation. The evaluator rejects a report that
cannot prove the one-tap raw-word fallback; normalization is used only for accuracy scoring.

Metadata is a JSON object that binds the report to the artifacts and required environments:

- `schemaVersion`: `2`.
- `appCommit`: the full lowercase Git commit tested.
- `coreApkSha256`, `swipeModelSha256`, and `contextModelSha256`: lowercase SHA-256 values for the
  exact APK and both models used for every reported prediction.
- `peakAddedNeuralMemoryMiB`: the maximum measured added neural memory.
- `environments`: exactly one `stock_android_hardware`, one `grapheneos_hardware`, and one
  `low_ram_emulator` record. Every record includes `deviceModel`, `buildFingerprint`, `testRunId`,
  `apiLevel`, and `physicalDevice`. Hardware must be physical and run Android 15 or newer. The
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
`measurementDatasetSha256` directly from the input JSONL and places that hash in the report; callers
cannot supply or override it.

```sh
python3 tools/evaluate_engine.py measurements.jsonl \
  --metadata measurement-metadata.json \
  --report build/reports/phase-0.json
```

The normal command enforces the release-size dataset minimums. `--allow-small-dataset` exists only
for developing the evaluator and cannot produce release evidence. A report passes only when every
quality, false-correction, latency, memory, and swipe-stratum gate from the product plan passes.
Final release verification requires both the report and its raw JSONL, recomputes every metric and
requires the result to match the report exactly.
