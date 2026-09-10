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

A second matched 1,000-path development experiment added an unconstrained greedy CTC spelling only
when it was absent from the native vocabulary. It scored that spelling with the exact CTC forward
probability and zero dictionary frequency, without consulting evaluation targets. It added candidates
on 315 paths, but reduced union top-1/top-3 from 86.6%/92.7% to 84.2%/91.5%. Static CTC also declined
from 88.2%/92.2% to 87.7%/91.8%. The experiment was rejected; production still uses the existing
dictionary-constrained decoder. The local report is
`build/reports/swipe-native-validation-greedy-oov-1000.json` and includes the experimental script hash.

## Android decoder diagnostic

`tools/prepare_swipe_android_diagnostic.py` binds 100 validation paths (at least 10 per required
stratum) to the checked swipe export. The opt-in `SwipeRuntimeInstrumentedTest` reads the installed
native English dictionary, runs the source-built ONNX adapter and Kotlin CTC decoder, and records
the exact fixture, model, APK and dictionary hashes with every ranked slate and stage timings.
Its fixture/model belong in the debug app's private `files/swipe-runtime-diagnostic` directory.
Run with `-e libreboardRequireSwipeRuntime true -e swipeFixtureSha256 <printed-hash>`; the optional
`swipeMaximumRows` argument permits a smaller profiling run and is recorded in the report.

The diagnostic uses a generous 60-second per-path deadline so slow work is measured rather than
hidden behind production timeouts. It replays the already prepared 64-point paths through Android's
feature conversion; this is not original raw-sensor replay or an interchangeable Python inference
measurement. It excludes retained AOSP suggestions, final fusion, editor publication and personal
data. It cannot satisfy Phase 0 or the Android device matrix.

On the arm64 API 36 AOSP emulator, the 100-path unminified debug run scored 89% top-1 and 90% top-3.
Stage measurements found inference p95 around 4.4 ms, vocabulary lookup around 222 ms and the
remaining trie/beam work around 1.61 seconds. A software-clock profile identified spelling-path
allocation and garbage collection as major costs. Ordinary words now avoid per-character variant
expansion, ASCII labels use a direct class table, and the immutable vocabulary index no longer
repeats normalization/deduplication on every query. German alternatives retain their existing order
and bounds; Unicode and joiner regressions pass.

Every ranked slate was identical after these changes. Observed decoder p50/p95 changed from
1,132/1,813 ms to 250/362 ms; lookup p95 changed from 222 to 13 ms. These runs shared the host with
ongoing training and are not a release benchmark. The result remains above the 200 ms gate.
Retained local reports are `build/device-evidence/swipe-android-diagnostic/android-report-stages.json`
and `android-report-optimized.json`; the earlier cold run is `android-report-initial.json`.

A follow-up stores a trie's first child directly and allocates child maps/terminal lists only when
needed, while lowercase ASCII words bypass unnecessary Unicode normalization. All 100 ranked slates
again matched exactly; observed p50/p95 was 224/315 ms in
`android-report-sparse-trie.json`. The full host suite still passes 357 tests. This further reduces
temporary allocation, but does not establish the release latency or added-memory budgets.

The next pass materializes candidates only for the final or timed-out beam, computes each completed
frame's ranking score once before sorting, and inserts already-normalized lowercase ASCII words
without temporary spelling lists/emission arrays. A missing-key spelling is fully validated before
any trie node is added. Timeout and missing-key partial-branch regressions pass, as do all 359 tests
in the documented `runTests` variant. The broader debug variant still exposes the inherited,
explicitly named `insertLetterIntoWordHangulFails` case; no Hangul behavior was changed.

All 100 complete ranked slates again match the sparse-trie baseline. The final observed decoder
p50/p95 is 207/292 ms; intermediate passes measured 234/321 and 194/274 ms. Concurrent teacher
scoring and Linux builds changed host load, so these are diagnostics, not a controlled speed claim.
The final result still exceeds 200 ms. Reports are retained as `android-report-final-slate.json`,
`android-report-frame-score.json`, `android-report-ascii-trie.json` and the hash-bound
`decoder-allocation-comparison.json` in the same local evidence directory.

## Reference-prefix diagnostic preparation

`python3 tools/prepare_swipe_reference_context.py` verifies the pinned raw FUTO source bytes,
prepared swipe split bytes, and every source-to-prepared identity before publishing diagnostic
JSONL sidecars under `build/evaluation-data/swipe-reference-context-v1`. Existing output directories
are refused so earlier evidence is preserved. The committed manifest is
`docs/models/evidence/swipe-reference-context-v1.json`.

The publisher collected swipes against predefined sentence prompts. These sidecars contain the
prompt prefix before the indexed target, not captured prior editor input. They exclude the current
target and following words, preserve first-word empty prefixes, and bound context to 256 UTF-16
units. Sentence/index/target alignment must match exactly after the documented normalization;
publisher-invalid sentences are excluded without guessing repairs.

The verified test set has 53,551 aligned paths (48,255 nonempty prefixes and 5,296 empty prefixes),
with 285 publisher-invalid rows excluded. Validation has 47,292 aligned paths, with 63 excluded.
Every required test stratum retains at least 500 paths; the smallest is very-sloppy at 2,277.
These counts match the independent source screen. Five focused prefix tests and the complete
215-test Python suite passed (13 expected skips).

The sidecars preserve CTC split membership but do not independently qualify context-model holdout.
Use the shared-session audit before evaluating a context candidate. Prompt-prefix diagnostics do
not establish real editor-history performance, full runtime fusion, or release readiness.

## Retaining candidate scores for later context diagnostics

The optional `--slates-output /path/to/new-slates.jsonl` argument saves each selected row's identity,
session, language, target, strata, and ordered CTC, geometric, and merged candidates with their
spatial and frequency scores. The evaluation report records the completed file's SHA-256, size,
and row count. Output appears only after evaluation succeeds; existing paths are refused and
partial output is removed after failure. Use a separate path for the report.

Score retention does not change inference, sampling, ranking, or quality gates. In the verified
100-row native-vocabulary validation run, all four baseline metric tables were unchanged, and
replaying the saved merged scores reproduced every stratum's top-1/top-3 counts. The complete
219-test Python suite passed with 13 expected skips. Saved slates remain host diagnostics: adding
reference context later still requires a model/split audit and does not establish full IME fusion.

## Frozen 6,000-row validation baseline for context reranking

The native-vocabulary validation run retained complete candidate scores for 6,000 rows, with 600
minimum rows per stratum. Its report and replay checks are committed under `docs/models/evidence/`
as `swipe-validation-context-base-6000.json` and `context-swipes-slate-verification.json`. All four
metric tables were reproduced from the saved scores, and 5,997 rows joined to aligned prompt
prefixes; every required stratum retained at least 600 rows.

CTC alone scored 85.27% top-1 / 91.33% top-3; static fusion scored 87.10% / 92.17%; geometry scored
67.63% / 82.70%; CTC plus geometry scored 86.67% / 92.52%. The combined relative top-1 error
reduction over geometry was 58.81%, but the absolute quality gates still failed.

The candidate-recall audit (`swipe-validation-candidate-recall-bound.json`) exposes a further
constraint: the merged 32-candidate slate contains the target in only 5,670 of 6,000 rows (94.50%).
Even its untrimmed CTC/geometric union contains only 5,690 targets (94.83%). A perfect reranker
cannot attain 95% top-3 on these fixed slates. Further candidate-generation coverage work is needed;
context reranking alone cannot close that gate. This bound applies to the isolated diagnostic,
not the unmeasured complete IME path with retained AOSP and personal suggestions.

## Full dictionary coverage check

The opt-in `StaticDictionaryInstrumentedTest` argument `exportFullStaticLexicon=true` writes
`static-swipe-lexicon-full-diagnostic.json` with a separate 200,000-word bound and `diagnosticOnly`.
The ordinary production-bound export remains unchanged. The Android test passed, visited 160,715
native entries, retained 157,966 usable normalized entries, and reproduced the original 100,000
selected words exactly. The full export is deliberately rejected by the normal production-bound
host evaluator; it is for coverage analysis only.

Of the 330 target misses in the frozen 6,000-row merged slate, 268 are outside the bounded
vocabulary, 42 are in vocabulary but absent from both decoder slates, and 20 are lost at union
truncation. Applying the existing safety and emission rules to the full dictionary recovers only
10 of the 268 vocabulary misses: 253 targets are absent from the full dictionary and five remain
excluded by policy or emission constraints. Increasing usable vocabulary from 99,222 to 156,350
would raise membership coverage only from 5,732 to 5,742 of 6,000 rows. Production limits therefore
remain unchanged. Evidence is in `docs/models/evidence/full-dictionary-validation-coverage.json`;
APK and instrumented export provenance is in `full-static-vocabulary-provenance.json` there.

## Candidate-coverage experiments after the frozen baseline

A source-verified vocabulary analysis reads only the 919,337 training rows when constructing
supplements; validation targets are used only for membership measurement. Known offensive flags
from the bundled dictionary are excluded, but corpus additions still require a complete policy
review before adoption. The coverage report is `docs/models/evidence/swipe-training-vocabulary-coverage.json`.

| Minimum distinct training sessions | Additional decodable words | Newly covered validation rows |
| --- | ---: | ---: |
| 1 | 27,614 | 104 |
| 2 | 4,089 | 55 |
| 5 | 404 | 30 |
| 10 | 131 | 20 |

Separately, re-running only CTC inference on the exact 6,000 frozen paths produced unconstrained
ASCII spellings, excluding known offensive words. Appending a new spelling within either a 31- or
32-candidate bound recovered 41 targets without losing another target at truncation: candidate
recall rose from 5,670 to 5,711 of 6,000 (95.18%). Six known-offensive spellings were rejected.
The report is `docs/models/evidence/swipe-greedy-candidate-recall.json`.

These are coverage bounds, not improved ranking results. The earlier greedy-ranking experiment
regressed and remains rejected. Production vocabulary, scoring, and policy are unchanged; any
candidate expansion still needs actual ranking, latency, memory, and full-runtime validation.

A later diagnostic fitted an OLS map from unconstrained CTC forward log-probability onto
lexicon-constrained spatial scores on 512 training paths (targets not returned; no frequency
prior). Greedy OOV spellings then entered append-only reserved slots and competed in ranking
with frequency-free scores placed just below the lexicon top-3. On the matched 1,000-path
comparison two runs agreed: union top-1/top-3 86.6%/92.7%, competing-slate recall 952/1,000
versus merged 944/1,000 (the uncalibrated greedy-OOV ranking had been 84.2%/91.5%). Host
prefix-beam p95 stayed near 11 ms. Evidence:
`docs/models/evidence/swipe-oov-score-calibration.json` and
`docs/models/evidence/swipe-calibrated-greedy-oov-ranking.json`.

On the frozen 6,000-row slate (`slatesSha256` `fbbbcd0ef004f651a2e6c698ce6d834d4d2ca8d5f8195a6c9996e5ff2c61776e`),
candidate recall counts only the ranking-competing slate (merged union reserved). Every reserved
candidate in that bound is passed to `rank_static_fusion_with_reserved`. Reserved-slot n-best CTC
spellings, in-lexicon edit neighbors, stratum-adaptive CTC/geometry budgets, and truncated decoder
candidates raised competing-slate recall from 5,670 to 5,854 of 6,000 (97.57%), with `return_trip`
recall 94.39%. Static-fusion ranking on those slates was 87.28%/92.60% versus the frozen
86.67%/92.52% floor. Host reserved-decode p95 was 63.4 ms. This is a candidate membership bound,
not a Phase 0 quality pass. Production vocabulary, scoring, beam width, and safety policy stay
unchanged. Evidence: `docs/models/evidence/swipe-nbest-reserved-recall.json`,
`docs/models/evidence/swipe-stratum-adaptive-merge.json`, and
`docs/models/evidence/swipe-validation-candidate-recall-bound-calibrated.json`.

Beam 256 on the 1,156 long and double-letter frozen rows raised CTC-only candidate recall from
1,006 to 1,023 targets (long 605→618, double_letter 550→561). Isolated prefix-beam p95 moved
from 10.3 ms to 40.3 ms. The earlier matched 1,000-path union p95 of 91 ms is still treated as
unaffordable, so production beam width stays 64. This is a residual search gap, not a capacity
signal to retrain. Evidence: `docs/models/evidence/swipe-beam256-recall.json`.

## Published-31 membership (not the competing-slate 97.57%)

The 5,854/6,000 competing-slate figure counts membership in an unbounded ranking-input bag
(~184 extra spellings per path). Phase 0 `fused_swipe` top-3 ≥ 0.95 needs the target in the
published 31 (32 slots, one empty-swipe placeholder). A first-source ablation of the 184
recoveries on the frozen slates (`slatesSha256`
`fbbbcd0ef004f651a2e6c698ce6d834d4d2ca8d5f8195a6c9996e5ff2c61776e`) labeled:

| First source | Recovered rows |
| --- | ---: |
| greedy unconstrained CTC | 41 |
| per-frame 2nd/3rd-best greedy variant | 69 |
| prefix-beam n-best (n=2…32) | 33 |
| in-lexicon edit-1 / transposition / apostrophe neighbor | 36 |
| truncated CTC leftover | 4 |
| truncated geometry leftover | 1 |

Cumulative bag recoveries: greedy 41, greedy+alts 110, beam≤4 115, beam≤32 143,
+neighbors 179, +dropped 184. Construction never reads targets. Evidence:
`docs/models/evidence/swipe-184-source-ablation.json`.

Reserved merge always replaces the worst of the ranked 31 (at most 4 slots), matching the
greedy-41 31/32 existence proof. Slots fill greedy, then best-spatial 2nd/3rd-best and n-best
≤4, then in-lexicon neighbors. True OOV use the train-fit OLS map blended 50/50 toward the
25th-percentile spatial (a score fix after pure OLS dropped 6k top-3 by 1 and 1k top-3 by 3).
They are not pinned below top-3. In-lexicon reserved use real `log1p(frequency)`.

The official published-31 diagnostic (greedy + alts + n-best 4 + neighbors + truncated
leftovers, reserved budget 4, OLS map blend 0.5) measured:

| Quantity | Frozen 32-slot | Published-31 diagnostic | Gate |
| --- | ---: | ---: | ---: |
| Target in ranked[:31] | 5,670 | 5,744 | 5,822 at 97.9% conversion |
| Static-fusion top-3 | 5,551 (92.52%) | 5,551 (92.52%) | 5,700 (95%) |
| return_trip published-31 | 1,529 | 1,564 | 1,572 |
| return_trip top-3 | 1,498 | 1,498 | 1,539 (90%) |
| Host reserved-decode p95 | — | 24.1 ms | < 91 ms |

New reserved hits convert at 0/74 into top-3. Pure OLS reached the same 5,744 membership but
5,550 top-3. Matched 1,000-path union ranking twice was 86.6%/92.7%. `diagnosticOnly` /
`releaseEligible` false. Production vocabulary, scoring, beam 64, and `CtcSwipeDecoder.kt`
are unchanged.
Evidence: `docs/models/evidence/swipe-published-31-recall.json` and
`docs/models/evidence/swipe-published-31-1000-{1,2}.json`.

The 1,000-row ranking follow-up assigned frequency 1 to the 4,089 supplemental words. Adding them
to both CTC and geometry lowered combined top-1/top-3 from 86.6%/92.7% to 85.0%/91.9%. Replaying
only supplemented CTC with the original geometric candidates yielded 84.8%/92.3%. Both scoring
variants were rejected; coverage gains did not translate to ranking gains. The frequency prior was
experimental and uncalibrated. See `docs/models/evidence/swipe-supplement-ranking-comparison.json`.

### Reference-prefix context validation

`tools/evaluate_swipe_context.py` scores the frozen merged slates with a checked context export.
It verifies slate hashes/counts, joins publisher reference prefixes by row/session/target/language
and strata, and requires a corpus-bound joint-split audit with zero context-training overlap in the
validation pool. Missing aligned reference rows are counted rather than silently invented.
The script compares neural coefficients 0, 0.35, 0.7, 0.95 and 1.2 (the existing 0/25/50/75/100
strength mapping). It cannot evaluate the final test split or claim release qualification.

Supply `--slates-report`, `--slates`, `--prefix-manifest`, `--prefix-root`, `--joint-split-audit`,
`--distillation-manifest`, `--spec`, `--export-report` and a new `--output` path. Each manifest must
refer to the same prepared corpus/model chain; the smaller student requires its explicit candidate
specification. A bounded wiring smoke additionally uses `--development --maximum-examples 16`.

This measures only static/spatial/context ranking over existing candidates. Reference prefixes are
publisher prompts, not captured editor input; exact prompt overlap remains possible despite session
separation. The script omits retained AOSP suggestions, personalization, language locks, deadline
fallback, and full-IME latency. It does not change candidate recall or production fusion settings.
The small partial-training smoke passed, and an independent replay reproduced its zero-context
baseline exactly. No coefficient selection or quality claim follows from that smoke.

The rejected greedy-OOV experiment was rerun solely to retain scored slates for the context
comparison. Known-offensive entries from the full native dictionary were excluded. It again added
315 candidates across 1,000 validation paths and reproduced static/union top-1/top-3 of
87.7%/91.8% and 84.2%/91.5%, respectively. All four metric tables replay exactly from the saved slates;
999 rows have verified reference-prefix joins, including 100 very-sloppy rows. The
[baseline report](models/evidence/swipe-greedy-context-base-1000.json) and
[replay proof](models/evidence/greedy-context-swipes-slate-verification.json) bind the diagnostic.
This preserves a rejected baseline for later context scoring; it does not adopt the OOV strategy.
