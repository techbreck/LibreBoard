# Swipe CTC offline evaluation

`tools/evaluate_swipe_ctc.py` runs the exported `swipe-latin-v1` ONNX model over a deterministic,
stratified held-out sample using the same CTC blank/repeat rules and lexicon-constrained prefix beam
semantics as the Android decoder. It also mirrors the pure-Kotlin geometric template cost, including
live-geometry trace edits and turn evidence, then normalizes the two decoder slates independently
before their bounded union. Release-mode diagnostics require at least 5,000 test gestures and 500
examples from every length, sloppiness, double-letter, and return-trip stratum.

The lexicon is derived only from the prepared train and validation targets. Test targets never enter
lexicon construction. The command verifies the pinned prepared-corpus hashes, exported-model hash,
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
evidence, the corpus lexicon is not the production AOSP dictionary, and the command does not
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
