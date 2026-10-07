# Current-work checkpoint

The Linux native ONNX Runtime A/B pair on the Debian-CMake image is complete and byte-identical. Canonical-model Android checks, human tap/swipe gates, devices, latency/memory, and signing remain open. LibreBoard is not release-qualified.

## Stock hardware Phase 0 leg, 2026-09-21

The stock physical Android environment — the matrix slot no attached device could previously fill — was
measured on a stock Pixel 7 Pro (cheetah, API 37, Play installed) under the artifact pin. Run
`stock-hw-shard0-1` replayed tap shard 0/3 (6,218 rows) and swipe shard 0/3 (17,946 rows) with the
pinned `8566382e…` context model, tokenizer, `1301f0d0…` swipe model and `54118ac8…` runtime AAR;
the driver verified the installed APK and injected bytes, all gated tests passed, and every pulled
row carries the run's id and environment. Bindings and findings are in
`build/device-evidence/stock-hw-shard0-1.evidence.json`; the evaluator parses the full dataset and
rejects it only for the two still-missing environments, as designed.

- The tap commit path now clears its direction on real hardware: +33.6% fused-vs-classic relative
  error reduction on tap errors (the 2026-09-12 diagnostic measured −5.26% before commit `712342dc`
  calibrated the head-to-head commit decision), 0% false corrections on valid-word keeps, and
  `fused_neural` p95 39.6 ms against the 80 ms budget.
- The neural context path still changes no tap commit at production strength 50 (`fused_neural`
  equals `fused` on every commit metric), but inference completes here (~1.5 ms p50 added), unlike
  the low-RAM circuit-breaker case.
- Swipe quality is transformed versus the superseded schema-3 diagnostics: CTC top-1/top-3
  83.97%/90.62%; `fused_swipe` 77.50%/88.26% — still below the 90%/95% gates.
- The swipe union fusion is now the dominant deficit: CTC alone beats the fused union overall and on
  every stratum except top-3 short; `long` top-1 falls 74.3% → 52.7% and `return_trip` 77.5% →
  63.4% when fused. `fused_swipe` p95 224.3 ms also exceeds the 200 ms budget; standalone geometric
  decoding still pins at its 1,500 ms deadline (p99 1,500.2 ms).
- The swipe-sidecar whole-process memory headline is 332 MiB added PSS (scope and attribution
  caveats recorded); isolated attribution remains open.

These are the stock-environment numbers a pooled evaluation will use, not pooled gate results.

## GrapheneOS hardware Phase 0 leg, 2026-09-21

The GrapheneOS environment was re-measured under the same pin and APK on a Play-free Pixel 9 Pro
XL (GrapheneOS 2026091901): run `grapheneos-hw-shard1-1` replayed tap shard 1/3 in full (6,218
rows) and 100 swipe rows of shard 1/3. Root cause found for this phone's chronic slowness:
**GrapheneOS disables the JIT system-wide** (`dalvik.vm.usejit=false`; stock ships `true`) and
AOSP policy caps debuggable packages to verify-only AOT, so the debug harness replays fully
interpreted — ~2.2× on tap paths, unbounded on the CTC beam loop. Release builds AOT-compile and
are unaffected; the condition bounds the harness, not the product. Findings:

- Tap decisions survive interpretation: +31.1% fused-vs-classic RER (stock: +33.6%), 0% false
  corrections, `fused_neural` p95 74.67 ms inside the 80 ms budget.
- Swipe rows are deadline-censored on this environment (CTC cannot finish interpreted beam search;
  the bail returns partial/empty candidates — ctc top-1 0.0). They must not be read as model
  quality; the same censor explains the superseded September diagnostics' signature.
- The swipe-lexicon build costs 130 s interpreted (22.7 s on stock), excluded from latencyMs but a
  real cold-start cost in this condition.
- Follow-ups: a non-debuggable testOnly measurement build for a future pin would restore
  representative latency here; the deadline bail returning garbage is a product defect worth
  fixing regardless (fall back to geometric candidates).

## GrapheneOS AOT measure swipe, 2026-09-28

The follow-up is implemented: build type `measure` is non-debuggable `testOnly`, the driver stages
corpora under `/data/local/tmp` and publishes outputs via `getExternalFilesDir`, and ART compiles
the app package to `speed` on this phone (`dalvik.vm.usejit` still false). Compiling the
androidTest package to `speed` aborted with an ART class-loader-context mismatch; leave it at
`verify`.

Run `grapheneos-measure-swipe-500-1` replayed 500 swipe rows of shard 1/3 on Pixel 9 Pro XL
(`47311FDAS00026`, GrapheneOS 2026091901) with APK `975362e4…` (not the pinned debug APK). CTC was
AVAILABLE on every row: top-1/top-3 81.0%/87.2%, p50/p95 99.8/152.6 ms, max 1301 ms, 0 timeouts.
Lexicon cold-start 8.9 s (interpreted debug: 130.3 s). `fused_swipe` 46.6%/50.4% and p95 226.4 ms
reproduce the stock fusion deficit and 200 ms overrun. Diagnostic only; evidence
`docs/models/evidence/grapheneos-measure-swipe-500-1.json`. The Phase 0 matrix still uses the
interpreted debug GrapheneOS tap leg plus the missing low-RAM environment.

## Swipe fusion, 2026-09-28

Production `fused_swipe` was ranking the native batch matcher and geometric scan over CTC, and
waiting out the leftover 125 ms proposal budget for geometric even after CTC had finished. CTC now
owns the fused swipe slate whenever it proposes; geometric fills only when CTC misses. Native
batch/trace matching is skipped when a swipe model is installed, and the proposal budget is 200 ms.

On a 20-row unsharded GrapheneOS AOT smoke, `fused_swipe` moved from 40%/40% with 6 empty slates to
75%/80%, matching standalone CTC. That sample does not hold at shard scale.

## GrapheneOS fusion swipe shard 1/3, 2026-09-29

Run `grapheneos-measure-swipe-shard1-1` replayed the full swipe shard 1/3 (17,945 rows) on Pixel 9
Pro XL (`47311FDAS00026`) with fusion measure APK `5b91a54f…` at ART `speed`. Evidence
`docs/models/evidence/grapheneos-measure-swipe-shard1-1.json`. Diagnostic only; not a Phase 0
matrix replacement.

- CTC: 84.16%/90.97% top-1/top-3, p50/p95 94.3/131.0 ms, AVAILABLE on every row. Still short of 90/95.
- `fused_swipe`: 68.48%/74.28%, p50/p95 213.8/229.6 ms. Up from the pre-fusion 500-row 46.6%/50.4%,
  still 15.7 pp behind CTC top-1, and 61.4% of rows exceed 200 ms (mode 210–220 ms).
- `fused_relaxed`: 86.43%/91.94%. Ranking works when both decoders finish; production is the gap
  (fused top-1 equals CTC top-1 on 71.0% of rows; 3,251 CTC hits that fused misses; 111 empty fused
  slates).
- Short fused top-3 96.66% clears 90%; return-trip fused top-3 56.72% does not. Geometric RER +41.9%.

## Production swipe budget, 2026-09-29

`fused_swipe` p50 sat at 214 ms (mode 210–220 ms) because `ParallelSwipeDecoder` still launched
geometric speculatively and treated CTC `TIMEOUT` partials as a miss, then waited out the 200 ms
proposal budget. CTC now runs first; geometric starts only when CTC publishes nothing. TIMEOUT
partials count as a CTC hit.

Same 50 shard-1 rows, fusion measure APK `4d4ed364…` at ART `speed`
(`grapheneos-fusion-budget-50-1`):

- Before: fused 50%/— top-1, p50/p95 214/262 ms, 40/50 over 200 ms, 6 empty, 27/50 fused top-1 = CTC.
- After: fused 86%/90% vs CTC 84%/90%, p50/p95 185/235 ms, 12/50 over 200 ms, 2 empty, 43/50 agree.
  Geometric ran on 1/50 rows. Stage split: decode p50 169 ms, fusion 9 ms, next-word 0 ms.

The 210–220 ms wait-out-budget mode is gone. Remaining p95 over 200 ms is the CTC decode itself
(p50 169 ms through `LiveSwipeModelSlot`) plus occasional 50 ms neural rescoring, not geometric.
Diagnostic only.

## GrapheneOS sequential CTC 500-row, 2026-09-29

Run `grapheneos-fusion-budget-500-1` replayed 500 swipe rows of shard 1/3 on the same sequential
CTC measure APK `4d4ed364…` at ART `speed`. Evidence
`docs/models/evidence/grapheneos-fusion-budget-500-1.json`. Diagnostic only.

- CTC: 84.6%/92.0% top-1/top-3, p50/p95 96.4/157.7 ms, AVAILABLE on every row.
- `fused_swipe`: 78.2%/84.0%, p50/p95 185.9/233.8 ms, 183/500 over 200 ms, 0 empty. Same IDs were
  47.0%/50.8% pre-fusion. `fused_relaxed` 86.6%/93.0%.
- The 50-row sequential check (86%/90%) overstated overall fused quality. Those same first 50 IDs
  scored 60%/66% here (33 CTC-empty in the fused path). When fused has CTC candidates (373/500),
  fused top-1 is 85.5% vs CTC 83.4%. Last 200 rows: fused 85.5%/92.5% vs CTC 84.5%/92.5%.
- 125 rows sit in 200–210 ms (decode p50 200.4 ms); the old 210–220 ms mode is 8 rows. 127 fused
  slates had zero CTC candidates and geometric ran on 114 of them. Standalone CTC on those rows is
  p50 89 ms, so the production 200 ms slot is emptying CTC.

## GrapheneOS LiveSwipeModelSlot partials, 2026-09-29

`LiveSwipeModelSlot` now waits 15 ms for a self-bounded TIMEOUT slate before interrupting, and will
not submit the next swipe while leftover ONNX still occupies the single decoder thread. Measure APK
`bc1dccb4…` at ART `speed`. Run `grapheneos-fusion-budget-500-2`, same 500 shard-1 rows. Evidence
`docs/models/evidence/grapheneos-fusion-budget-500-2.json`. Diagnostic only.

- CTC unchanged: 84.6%/92.0%, p50/p95 100.3/137.8 ms.
- `fused_swipe`: 83.0%/91.8% (was 78.2%/84.0%), p50/p95 179.5/221.4 ms, 99/500 over 200 ms (was 183),
  2 empty (was 127 CTC-empty; geometric in fused slate 114 → 0). `fused_relaxed` 86.6%/93.0%.
- First 50 IDs 86%/92% fused, matching the 50-row sequential check (they were 60%/66% on 500-1).
  Quintile 0–99 fused 87% with 1 CTC-empty (was 69% / 51).

Pooled status: stock + grapheneos rows (30,482) parse the full schema-4 contract; the evaluator now
rejects the metadata only for the missing low-RAM environment. The low-RAM leg (shard 2/3) is the
last matrix entry.

## CTC quality, 2026-09-29

On GrapheneOS swipe shard 1/3 (17,945 rows) CTC misses top-1 on 15.8% (84.16%/90.97%, in-slate
94.1%). Rank 2–3 is 43% of those misses, absent 37%, rank 4+ 20%. The greedy length window fails on
3 rows. Host greedy exact is 66.8%; the lexicon beam is the rest of the 84%.

Long (2,071): 73.9%/79.8%, 17.5% absent. Greedy exact 25.4%. Of the absences, 124 are in the 100k
en-US lexicon (logits/search: `massachusetts` → `mascara`) and 223 are OOV (names, hyphenated).
Return-trip (5,340): 77.4%/84.4% against the 90% top-3 gate; in-slate 88.1%, so coverage is the
limiter. Double-letter is the weakest ranking stratum (61.5% top-1): Viterbi says `still`, the
summed beam prefers `sill`.

`CtcSwipeDecoder` now hoists the unconstrained greedy spelling when it is already on the beam.
Simulated on this shard: +136 top-1 / −73, 84.16% → 84.92%. Long +0.19 pp, return-trip +0.30 pp.
`fused_relaxed` 86.43%/91.94% remains the both-decoder ceiling. Perfect ranking of the current CTC
slate would be 94.1% top-1/top-3 — clears 90% top-1, still short of 95% top-3. Next lever is a new
`swipe-latin-v1` candidate with stratum-weighted long / double-letter / return-trip training.

## Stratum-weighted candidate, host CTC, 2026-09-29

A `swipe-latin-v1` candidate was trained on Linux (CPU, 8 threads, 12 epochs) from
`models/swipe/model-spec-strata-candidate.json` at commit `5156cceb`: same 64×32 ABI and
architecture (821,121 parameters), with long / double-letter ×3 and return-trip ×2 sampling
(1,416,329 examples per epoch; final validation greedy exact 73.1%, CER 8.7%). The exported ONNX is
`f2ba9b7a…`; it is a candidate, not the pinned model, and is not committed.

A matched baseline was retrained from the unweighted `model-spec.json` on the same host, commit,
data manifest, toolchain and thread count (ONNX `46c3bece…`; final validation greedy exact 67.9%).
Host `evaluate_swipe_ctc.py`, default 5,000-row stratified sample, corpus lexicon (diagnostic only;
both models on identical rows), CTC top-1 / top-3:

| Test stratum | Rows | Matched baseline | Candidate | Δ |
|---|---|---|---|---|
| Overall | 5,000 | 85.28% / 91.58% | 86.88% / 92.02% | +1.60 / +0.44 pp |
| Long | 569 | 69.24% / 76.45% | 75.04% / 79.26% | +5.80 / +2.81 pp |
| Return-trip | 1,467 | 77.37% / 84.19% | 81.32% / 85.75% | +3.95 / +1.57 pp |
| Double-letter | 515 | 67.18% / 80.39% | 76.50% / 83.11% | +9.32 / +2.72 pp |
| Short | 1,844 | 95.34% / 98.32% | 95.17% / 98.16% | −0.16 / −0.16 pp |

Validation agrees: overall 85.94%/91.50% → 87.44%/92.48%, long +2.20/+2.03, return-trip
+2.50/+1.81, double-letter +8.33/+6.40 pp. The matched baseline is within 0.04 pp of the
documented `1301f0d0…` overall CTC top-1 (85.24%; its top-3 is 0.6 pp higher), so the gain is
attributable to the weighting rather than to retraining. The
CTC + geometric union moves less (test overall 86.34%/92.78% → 86.82%/92.84%; long top-1
+1.23 pp): the geometric fallback already recovered part of what the candidate now gets from CTC.

On in-vocabulary test rows the candidate's CTC is 91.0%/96.3% overall, long 90.9%/96.0%,
return-trip 90.8%/95.7%, double-letter 85.7%/93.0%. The remaining host top-3 gap on long and
return-trip is lexicon coverage (82.6% and 89.6%), not model ranking; double-letter still loses on
ranking. Device replay of the 500 shard-1 IDs on the measure APK is the next measurement; these host
numbers are not Phase 0 evidence.

## Completed

- Integrated the corrected shared-session corpus split, tokenizer binding, and teacher manifest from `codex/context-shared-splits`. Main retains the smaller explicit candidate profile and newer evaluation gates.
- Completed four epochs for both the 35.66M-parameter and 7.61M-parameter context students. Each has two independent Linux exports with all eight files byte-identical. All 32 file hashes across the four export directories were rechecked at this checkpoint.
- Large canonical INT4 model: validation top-one/top-three 68.40%/84.17%; held-out test 68.96%/85.81%. These are teacher candidate-ranking diagnostics, not swipe accuracy.
- Small canonical INT4 model: validation top-one/top-three 68.77%/84.63%. Prompt-disjoint validation: 68.97%/84.75% on 3,181 examples. Canonical Linux results exactly match the host export's validation metrics, despite differing model bytes. Float held-out test top-one is 67.56%; no canonical INT4 test claim is made.
- Frozen swipe validation: every tested nonzero small-context coefficient regressed the zero-context baseline on 5,997 aligned examples. The best greedy-OOV context variant also remained below the original bounded baseline on the same 999 examples. Neither strategy was adopted.
- Earlier pinned Linux core APK pair was byte-identical and passed packaging checks. This remains evidence for source `d421b3e345758068f213d8f3b62949270a273ec2`, not this final source checkpoint or a model-qualified release.

## Native runtime failure

Both native A attempts on image `sha256:57d61051fc53c6b8d9b21ca2d370a4408dbcb8e86836d3ffa66a81c154e90622` exited 1 with Docker reporting `OOMKilled=false`. Initial CMake dependency population failed with SIGSEGV while preparing Abseil. Archive integrity and isolated CMake download/extraction probes passed. The retry progressed to ONNX dependency population, then SIGSEGV occurred in its no-test stamp command.

Follow-up diagnosis reproduced that stamp command shape (`cmake -E echo_append && cmake -E touch`) SIGSEGVing under Colima qemu-user 7.0.0 when ninja spawned Android SDK CMake 3.31.6, a 21 MiB non-PIE linux-x86_64 binary. Isolated `cmake -E` from Python succeeded; `ldd` on the SDK cmake/ninja binaries exited 139; `/proc/cpuinfo` inside the amd64 container reports ARM features. The crash is intermittent, not an OOM, corrupt archive, or ONNX source defect. Evidence: [failure record](evidence/linux-runtime-pair-failure.json) and [qemu diagnosis](evidence/linux-runtime-cmake-qemu-diagnosis.json).

The Linux runtime image recipe now uses Debian bookworm-backports CMake 3.31.6 and Debian ninja-build as PATH host tools, while keeping Android SDK `cmake;3.31.6` installed for AGP. The rebuilt image is `sha256:34c6bfa469b1afb637cbe9e95c3a5ec2a5c5e4ad011953cce70d0d83073a1fe7`. Unprivileged smoke passed `--check-only`, 15/15 FetchContent stamp-command reproductions, and an empty-command ExternalProject no-test stamp.

Native A completed on the Debian-CMake image after the ninja-zombie resume: container `libreboard-runtime-repro-debian-a4` exited 0. Independent B failed once on Eigen FetchContent HTTP 503, was seeded from A's already-fetched zip, then `libreboard-runtime-repro-debian-b3` exited 0 after a host-reboot resume of the same B ninja cache. Both development AARs are 12,352,564 bytes, SHA-256 `54118ac8e37bc4833d32e2cb197ef6bd251f56e5aa9e4e51c3899fb616210d88`, with identical manifests and all four ABI `.so` pairs. Evidence: [A AAR](evidence/linux-runtime-debian-a-aar.json), [B AAR](evidence/linux-runtime-debian-b-aar.json), [pair](evidence/linux-runtime-debian-pair.json), [Eigen 503](evidence/linux-runtime-debian-b-eigen-503.json). This is pinned development-operator runtime reproducibility, not a model-qualified release.

Raw logs remain under `build/reports/linux-runtime-a-initial-failure.log` and `build/reports/linux-runtime-a.log`. Earlier Android timing results used a partially trained small model and an older large model; they do not qualify either newly completed canonical model.

## Verification and remaining gates

The final Python suite ran 232 tests successfully with 13 optional-toolchain skips. Export hashes were rechecked against the committed reproducibility reports. Source-release verification passed after updating its strict tokenizer pin to the verified shared-session tokenizer; results are recorded in [final checkpoint evidence](evidence/current-work-final-verification.json).

Tap corpus coverage is no longer a gate. The combined held-out corpus supplies 18,654 rows — 3,462 tap errors, 10,767 valid-word cases (6,132 corrections, 4,635 keeps), 509 spacing cases and 3,916 lexical cases across all three required kinds — which clears every `evaluate_engine.py` minimum. The earlier "887 missing spatial examples" and missing spacing/personal/compound counts are stale.

Release gates remain open: swipe absolute accuracy and return-trip quality; physical stock Android and low-RAM measurement (only GrapheneOS has been collected, and those rows are diagnostics, not qualifying evidence); full-IME latency and added peak-memory qualification; accepted signing and a model-qualified release artifact. No release was published. Training outputs and candidate worktrees are retained for a later explicitly requested continuation.

Three measurement defects are now fixed in the harness rather than in the evidence. Tap measurement scored slate rank one while the harness hoisted the typed word to the front, so every schema-3 tap result measured the harness; measurement schema 4 records production's own commit decision and the evaluator scores that. The harness also composed rows with `WordComposer.setComposingWord`, which marks the composition resumed, and production never auto-corrects a resumed word, so the correction rate was structurally 0% on every system including the classic baseline; rows are now replayed as key events. Separately, the GrapheneOS run measured the superseded independent-split context export, so [`docs/phase-0-artifact-pin.json`](../phase-0-artifact-pin.json) names the shared-session candidate, the driver fails by name on a rejected artifact, and metadata must state which candidate its evidence qualifies. All three require the device matrix to be re-measured; no previously collected tap number survives.

With the harness fixed, a 400-row emulator diagnostic against the superseded export shows the keyboard reaching the target in its top-3 slate 68.5% of the time while committing it almost never (0–1.7% top-1 on tap errors, 8.75–9.25% auto-correction overall). That is a real quality finding to investigate, not a harness artifact, and it is well short of the 20% tap relative-error-reduction gate.

Canonical model reports and reproducibility evidence are in [evidence](evidence/). The failed SDK-CMake runtime image `sha256:57d61051fc53c6b8d9b21ca2d370a4408dbcb8e86836d3ffa66a81c154e90622` is retained as `libreboard-runtime-build:sdk-cmake-57d61051`. Next runtime step is canonical-model Android checks with an explicit fixture, not another native AAR rebuild.
