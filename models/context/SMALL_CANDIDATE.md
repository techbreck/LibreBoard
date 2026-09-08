# Small context student experiment

The separate `model-spec-small-candidate.json` specifies a 7,605,760-parameter student (width 256, four heads, four layers, feed-forward width 768). The default 35,662,848-parameter specification and release verifier remain unchanged. This candidate addresses the measured memory and latency failures; it is not a release selection.

Both candidates use the same pinned shared-session corpus, teacher scores, tokenizer, four training epochs, losses, and tensor/scoring ABI. Export validation derives the exact MatMulNBits node count from the layer count (28 for the small candidate, 56 for the default); other operator and tensor restrictions are unchanged. Pass `--spec models/context/model-spec-small-candidate.json` explicitly to training, export, evaluation, and runtime fixture generation.

Select using validation and Android measurements. Keep prompt-disjoint diagnostics, full held-out quality gates, joint memory and latency limits, physical-device requirements, reproducibility, licensing, and signing requirements. Synthetic kernel parity and host timings cannot qualify a release.
