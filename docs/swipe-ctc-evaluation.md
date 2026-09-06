# Swipe CTC offline evaluation

`tools/evaluate_swipe_ctc.py` runs the exported `swipe-latin-v1` ONNX model over a deterministic,
stratified held-out sample using the same CTC blank/repeat rules and lexicon-constrained prefix beam
semantics as the Android decoder. Release-mode diagnostics require at least 5,000 test gestures and
500 examples from every length, sloppiness, double-letter, and return-trip stratum.

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
0.65 static-frequency weight. The latter intentionally reserves one of the 32 bounded scorer slots
for the empty swipe raw placeholder, matching the Android publication path.

The report explicitly marks itself diagnostic-only. Host timing is not Android performance
evidence, the corpus lexicon is not the production AOSP dictionary, and the command does not
evaluate geometric, personal, language-lock, context, or complete final fusion. Consequently, even
a report that clears the numerical swipe thresholds does not satisfy Phase 0.

The Android CTC decoder uses its unconstrained greedy emissions to choose a bounded lexicon-length
window before prefix search. The geometric fallback instead estimates length from total path length
in live-key units; counting every crossed key substantially overestimates normal continuous swipes.
The bounded three-below/four-above window covers more than 99% of the pinned validation partition
without consulting held-out targets.

German candidates remain language-tagged while popup-only `ä`, `ö`, `ü`, and `ß` receive bounded
German-only gesture variants over their visible base keys. The original surface is retained, and the
same aliases are used by both Android decoders and the offline CTC evaluator.
