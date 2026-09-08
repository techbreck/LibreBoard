# Small context student experiment

The separate `model-spec-small-candidate.json` specifies a 7,605,760-parameter student (width 256, four heads, four layers, feed-forward width 768). The default 35,662,848-parameter specification and release verifier remain unchanged. This candidate addresses the measured memory and latency failures; it is not a release selection.

Both candidates use the same pinned shared-session corpus, teacher scores, tokenizer, four training epochs, losses, and tensor/scoring ABI. Export validation derives the exact MatMulNBits node count from the layer count (28 for the small candidate, 56 for the default); other operator and tensor restrictions are unchanged. Pass `--spec models/context/model-spec-small-candidate.json` explicitly to training, export, evaluation, and runtime fixture generation.

Select using validation and Android measurements. Keep prompt-disjoint diagnostics, full held-out quality gates, joint memory and latency limits, physical-device requirements, reproducibility, licensing, and signing requirements. Synthetic kernel parity and host timings cannot qualify a release.

## Completed host diagnostics, 2026-09-08

The full four-epoch run at `f3d6220f1065ccddfe8bfe19fbea4d0943dbebc9` completed on the same
61,604 training examples. Final floating-point distillation validation/test top-one accuracy was
68.80%/67.56%. The host INT4 export is 4,436,097 bytes with SHA-256
`faa30394b789d1c3161f836783a520468bc5664626a9af8fd8ca1bd5c1157f32`.
INT4 validation on all 3,253 slates was 68.77% top-one and 84.63% top-three. The separate
prompt-disjoint diagnostic excludes 72 exact repeated prompts and reports 68.97%/84.75% on 3,181
slates. These distillation metrics do not establish correction quality on human tap errors.

The frozen swipe comparison gives no quality justification for adopting this scoring setup:

| Neural coefficient | Top-one, 5,997 aligned swipes | Top-three |
|---|---:|---:|
| 0 | 86.68% | 92.53% |
| 0.35 | 86.53% | 92.33% |
| 0.7 | 83.09% | 91.40% |
| 0.95 | 79.64% | 90.16% |
| 1.2 | 75.97% | 88.73% |

The zero-context result independently replays for every stratum. The separate greedy-OOV comparison
also remains rejected: its best tested context coefficient (0.35) yields 855/919 top-one/top-three
hits on 999 aligned rows, versus 866/927 for the original bounded candidate baseline on those same
rows. No production fusion setting or model selection was changed.

The [training report](../../docs/models/evidence/context-small-full-training.json),
[host export](../../docs/models/evidence/context-small-host-export.json),
[INT4 validation](../../docs/models/evidence/context-small-host-validation.json),
[swipe comparison](../../docs/models/evidence/context-small-swipe-validation.json), and
[independent baseline replay](../../docs/models/evidence/context-small-swipe-replay.json) retain the
bindings. A training/export report's `releaseEligible` field indicates complete provenance at that
stage; it does not qualify the model or application for release. Linux reproducibility, final-artifact
Android measurements, physical devices, human-data gates, and accepted signing remain separate.
