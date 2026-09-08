# Current-work stopping checkpoint

The user requested completion of work already running, a progress report, and then a stop. Five model/evaluation job chains completed successfully. The sixth, the Linux native runtime pair, ended in a confirmed configuration failure after one diagnostic retry. No further experiments or retries are queued. LibreBoard is not release-qualified.

## Completed

- Integrated the corrected shared-session corpus split, tokenizer binding, and teacher manifest from `codex/context-shared-splits`. Main retains the smaller explicit candidate profile and newer evaluation gates.
- Completed four epochs for both the 35.66M-parameter and 7.61M-parameter context students. Each has two independent Linux exports with all eight files byte-identical. All 32 file hashes across the four export directories were rechecked at this checkpoint.
- Large canonical INT4 model: validation top-one/top-three 68.40%/84.17%; held-out test 68.96%/85.81%. These are teacher candidate-ranking diagnostics, not swipe accuracy.
- Small canonical INT4 model: validation top-one/top-three 68.77%/84.63%. Prompt-disjoint validation: 68.97%/84.75% on 3,181 examples. Canonical Linux results exactly match the host export's validation metrics, despite differing model bytes. Float held-out test top-one is 67.56%; no canonical INT4 test claim is made.
- Frozen swipe validation: every tested nonzero small-context coefficient regressed the zero-context baseline on 5,997 aligned examples. The best greedy-OOV context variant also remained below the original bounded baseline on the same 999 examples. Neither strategy was adopted.
- Earlier pinned Linux core APK pair was byte-identical and passed packaging checks. This remains evidence for source `d421b3e345758068f213d8f3b62949270a273ec2`, not this final source checkpoint or a model-qualified release.

## Native runtime failure

Both native A attempts exited 1 with Docker reporting `OOMKilled=false`. Initial CMake dependency population failed with SIGSEGV while preparing Abseil. Archive integrity and isolated CMake download/extraction probes passed. The retry progressed to ONNX dependency population, then SIGSEGV occurred in its no-test stamp command. The cause remains unconfirmed. Native compilation and the independent B build did not complete, so no Linux AAR reproducibility result exists. Final canonical-model Android checks using that AAR could not run.

Raw logs remain under `build/reports/linux-runtime-a-initial-failure.log` and `build/reports/linux-runtime-a.log`; container states and log hashes are preserved in [failure evidence](evidence/linux-runtime-pair-failure.json). Earlier Android timing results used a partially trained small model and an older large model; they do not qualify either newly completed canonical model.

## Verification and remaining gates

The final Python suite ran 232 tests successfully with 13 optional-toolchain skips. Export hashes were rechecked against the committed reproducibility reports. Source-release verification passed after updating its strict tokenizer pin to the verified shared-session tokenizer; results are recorded in [final checkpoint evidence](evidence/current-work-final-verification.json).

Release gates remain open: swipe absolute accuracy and return-trip quality; 887 missing human spatial tap examples, 500 spacing examples, and personal/compound coverage; physical stock Android and GrapheneOS testing; full-IME latency and added peak-memory qualification; accepted signing and a model-qualified release artifact. No release was published. Training outputs and candidate worktrees are retained for a later explicitly requested continuation.

Canonical model reports and reproducibility evidence are in [evidence](evidence/). No work should automatically resume from this stopping checkpoint.
