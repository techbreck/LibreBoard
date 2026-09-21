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

These are the stock-environment numbers a pooled evaluation will use, not pooled gate results. The
GrapheneOS re-run (shard 1/3) and qualifying low-RAM run (shard 2/3) remain to be measured against
the same pin, from `4d1c990e`, whose tree is exactly the measured APK's content.

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
