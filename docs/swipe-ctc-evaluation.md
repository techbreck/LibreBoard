# Swipe CTC offline evaluation

`tools/evaluate_swipe_ctc.py` runs the exported `swipe-latin-v1` ONNX model over a deterministic,
stratified held-out sample using the same CTC blank/repeat rules and lexicon-constrained prefix beam
semantics as the Android decoder. It also mirrors the pure-Kotlin geometric template cost, including
live-geometry trace edits and turn evidence, then normalizes the two decoder slates independently
before their bounded union. Release-mode diagnostics require at least 5,000 evaluated gestures and 500
examples from every length, sloppiness, double-letter, and return-trip stratum.

By default the lexicon is derived from prepared train and validation targets and the evaluated
split is `test`. Use `--split validation` for development comparisons; its corpus lexicon uses
only train targets. Evaluated targets never enter lexicon construction. The command verifies the pinned prepared-corpus hashes, exported-model hash,
manifest binding, fixed tensor ABI, CPU-only provider, and exact NumPy/ONNX Runtime tool versions
before inference:

```sh
build/model-venv/bin/python tools/evaluate_swipe_ctc.py
```

The default report is `build/model-export/swipe-latin-v1/ctc-evaluation-report.json`. It contains
overall and per-stratum top-1/top-3 accuracy, vocabulary coverage, greedy accuracy, host timing, and
two rankings of the same isolated CTC slate. `metrics` preserves the prefix-beam order, while
`staticFusionMetrics` applies the production scorer's per-slate z-normalized spatial evidence and
0.65 static-frequency weight. `geometricMetrics` measures the geometric fallback after that same
static scorer, and `ctcGeometricFusionMetrics` measures their normalized union. These scored slates
intentionally reserve one of the 32 bounded scorer slots for the empty swipe raw placeholder,
matching the Android publication path.

The report explicitly marks itself diagnostic-only. Host timing is not Android performance
evidence, and the command does not
evaluate personal, language-lock, context, retained AOSP gesture suggestions, or complete final
fusion. Consequently, even a report that clears the numerical swipe thresholds does not satisfy
Phase 0. The vectorized Python geometry timing is diagnostic implementation timing, not a claim
about the pure-Kotlin decoder's device latency.

The Android CTC decoder uses its unconstrained greedy emissions to choose a bounded lexicon-length
window before prefix search. The geometric fallback instead estimates length from total path length
in live-key units; counting every crossed key substantially overestimates normal continuous swipes.
The bounded three-below/four-above window covers more than 99% of the pinned validation partition
without consulting held-out targets.

German candidates remain language-tagged while popup-only `ä`, `ö`, `ü`, and `ß` receive bounded
German-only gesture variants over their visible base keys. The original surface is retained, and the
same aliases are used by both Android decoders and the offline CTC evaluator.

## Full combined-decoder diagnostic, 2026-09-08

The 5,000-path run of the existing exported model measured:

| Decoder | Top-1 | Top-3 |
| --- | ---: | ---: |
| CTC beam | 85.24% | 90.98% |
| CTC with static scoring | 85.14% | 91.76% |
| Geometric with static scoring | 77.50% | 88.32% |
| CTC/geometric union with static scoring | 86.32% | 92.66% |

The union reduced top-1 error by 39.2% relative to geometry, but missed the absolute swipe gates.
Its very-sloppy top-1 was 71.4%; long-word top-1 was 72.93%. The corpus-derived vocabulary covered
95.52% of targets, leaving little margin for the 95% top-3 target even with improved ranking.
Further comparisons should measure the production dictionary path; do not add held-out targets to
the diagnostic lexicon. Model or decoder selection must use validation data, not tune this test set.

Evidence: `build/model-export/swipe-latin-v1/ctc-geometric-evaluation-report.json`, SHA-256
`3bab7fdf4b6f87706194d9829c7d11e761ff0315138e317eeb18c079d12ee816`, bound to model SHA-256
`1301f0d076f526fe0b67f2ea86a38abc8448f6de54d2c7191c002dcd4d05d38c`.
The run used the default sample/stratum counts and one inference thread. Host timing was collected
while context training was active and is not Android performance evidence. Phase 0 remains open.

The subsequent native dictionary check found a separate live-path defect: the bundled English
dictionary has 160,715 entries, and trie traversal places frequent words including `the`, `to`,
`of` and `with` after entry 100,000. The old index stopped at that position. Index construction now
scans up to 1,000,000 entries per language while retaining at most 100,000 distinct words ranked by
frequency, with deterministic ties. The scan still runs in the background. Instrumentation uses
the actual bundled binary dictionary and checks that these late-traversal words survive the bounded
index. This fixes vocabulary loss; it does not change the corpus-lexicon diagnostic above or establish
an Android quality/latency pass.

## Native dictionary validation diagnostic

The Android dictionary regression test can export the production collector's selected static words
with `-e exportStaticLexicon true`. It writes `files/static-swipe-lexicon.json` inside the debug app,
including the installed APK and bundled dictionary SHA-256 hashes. Manually install the debug and
instrumentation APKs before this command; Gradle's connected-test task uninstalls them afterwards.

```sh
adb shell am instrument -w -r \
  -e class helium314.keyboard.latin.engine.StaticDictionaryInstrumentedTest \
  -e exportStaticLexicon true \
  org.libreboard.keyboard.debug.test/androidx.test.runner.AndroidJUnitRunner
adb exec-out run-as org.libreboard.keyboard.debug cat files/static-swipe-lexicon.json \
  > build/device-evidence/static-swipe-lexicon.json
build/model-venv/bin/python tools/evaluate_swipe_ctc.py --split validation \
  --dictionary-lexicon build/device-evidence/static-swipe-lexicon.json \
  --dictionary-apk app/build/outputs/apk/debugNoMinify/LibreBoard_0.1.0-alpha01-debugNoMinify.apk \
  --output build/reports/swipe-native-validation.json
```

The evaluator rejects mismatching APK/asset hashes, duplicate normalized entries, and oversized
vocabulary exports. This diagnostic currently supports the bundled en-US dictionary, maps it to the
corpus's English language tag, and excludes possibly offensive words under the default policy.
It does not reconstruct vocabulary from evaluation targets. Reports retain split identity and hashes
of the input data, vocabulary export, originating APK, asset, and evaluator. Native vocabulary improves
coverage fidelity but still does not measure the complete Android publication path or device latency.

## Native-vocabulary validation, 2026-09-08

A separate 5,000-path **validation** run with at least 500 rows in every required stratum used the
APK-bound 100,000-word collector export (99,222 decodable words after the offensive-word filter).
Vocabulary coverage was 95.44%; the greedy-selected length window covered 99.98% of targets.

| Decoder | Top-1 | Top-3 |
| --- | ---: | ---: |
| CTC beam | 85.32% | 91.16% |
| CTC with static scoring | 87.10% | 91.98% |
| Geometric with static scoring | 67.74% | 82.72% |
| CTC/geometric union with static scoring | 86.68% | 92.38% |

The union's return-trip top-3 was 87.90%, also below its 90% gate. Short-word top-3 passed at 98.07%.
The union reduced top-1 error by 58.71% relative to geometry, but all three neural rankings still
missed the overall absolute thresholds. Static CTC had slightly better top-1 than the union; this
is a validation diagnostic to investigate, not a reason to claim the release quality problem solved.
Do not compare these numbers causally with the earlier corpus-vocabulary **test** run: both the
vocabulary and evaluated partition differ.

The retained report is `build/reports/swipe-native-validation.json`, SHA-256
`7e3f838b60898ac06487833c473fe3b2ece8d7df2a0369cea11c341dd664eb56`. It records the exact model,
vocabulary, APK, evaluator and split hashes. Full Android fusion and device budgets remain unmeasured.

A matched 1,000-path validation development comparison used 100 rows per required stratum. Widening
the beam from 64 to 256 changed union top-1 from 86.6% to 88.0% and top-3 from 92.7% to 92.8%, while
host prefix-decoding p95 increased from about 21 ms to 91 ms. Both runs remained below the absolute
quality gates. These host diagnostics do not justify changing the production beam or asserting an
Android performance result. The origin APK used for the native vocabulary is retained locally as
`build/device-evidence/static-vocabulary-origin.apk` so later APK builds cannot erase that provenance.
