# Current-work checkpoint

The previous session stopped after five completed model/evaluation chains and a terminal Linux native runtime configuration failure. Work has resumed from that checkpoint. The runtime SIGSEGV is diagnosed; the image recipe now uses Debian CMake/Ninja as PATH host tools. The native A/B pair has not yet been rebuilt. LibreBoard is not release-qualified.

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

The Linux runtime image recipe now uses Debian bookworm-backports CMake 3.31.6 and Debian ninja-build as PATH host tools, while keeping Android SDK `cmake;3.31.6` installed for AGP. The rebuilt image is `sha256:34c6bfa469b1afb637cbe9e95c3a5ec2a5c5e4ad011953cce70d0d83073a1fe7`. Unprivileged smoke passed `--check-only`, 15/15 FetchContent stamp-command reproductions, and an empty-command ExternalProject no-test stamp. Native compilation and the independent B build have not yet completed on that image, so no Linux AAR reproducibility result exists. Final canonical-model Android checks using that AAR could not run.

Raw logs remain under `build/reports/linux-runtime-a-initial-failure.log` and `build/reports/linux-runtime-a.log`. Earlier Android timing results used a partially trained small model and an older large model; they do not qualify either newly completed canonical model.

## Verification and remaining gates

The final Python suite ran 232 tests successfully with 13 optional-toolchain skips. Export hashes were rechecked against the committed reproducibility reports. Source-release verification passed after updating its strict tokenizer pin to the verified shared-session tokenizer; results are recorded in [final checkpoint evidence](evidence/current-work-final-verification.json).

Release gates remain open: swipe absolute accuracy and return-trip quality; 887 missing human spatial tap examples, 500 spacing examples, and personal/compound coverage; physical stock Android and GrapheneOS testing; full-IME latency and added peak-memory qualification; accepted signing and a model-qualified release artifact. No release was published. Training outputs and candidate worktrees are retained for a later explicitly requested continuation.

Canonical model reports and reproducibility evidence are in [evidence](evidence/). The failed SDK-CMake runtime image `sha256:57d61051fc53c6b8d9b21ca2d370a4408dbcb8e86836d3ffa66a81c154e90622` is retained as `libreboard-runtime-build:sdk-cmake-57d61051`. Independent native A/B compilation on the Debian-CMake image is the next runtime step.
