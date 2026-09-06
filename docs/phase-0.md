# Phase 0 evaluation gate

Phase 0 is an empirical release gate, not a documentation checkbox. Session-separated held-out data
must contain at least 3,000 tap errors, 1,000 valid-word confusions, 500 split/join cases, 500
contraction/personal/compound cases, and 5,000 swipe paths stratified by length and path difficulty.

The harness must report stock HeliBoard v4.1, fused static/spatial, fused plus personal, and fused plus
neural separately. Swipe reports geometric and CTC results separately and together. A proprietary blob
may be measured only outside the repository as a black-box reference.

`tools/evaluate_engine.py` implements the fail-closed metric and threshold checks over prediction
JSONL. Its versioned input contract is documented in `docs/phase-0-dataset-schema.md`, and its own
tests run in CI. Reports are cryptographically bound by SHA-256 to the APK and both models and must
identify stock hardware, physical GrapheneOS hardware without sandboxed Google Play, and a low-RAM
emulator. Every measured test row is bound to one declared device run, with minimum tap/swipe
coverage and separate latency gates per environment. The swipe source, split, training, and ONNX export contracts are executable and tested, and
the exact million-gesture corpus manifest is committed. The context student architecture, tokenizer
contract, INT4 exporter and hash-locked toolchain are also executable. Its pinned 68,748-sentence
English/German corpus uses whole-session 90/5/5 splits and produces a deterministic 16,384-token BPE
after runtime-parity checks. Its offline teacher-scoring pipeline completed all 68,748 records with no
rejection, and the committed distillation manifest binds the exact data, teacher, tokenizer, tools,
toolchain, output hashes and reconciled metrics. Its vectorized student trainer has deterministic
mid-epoch model/optimizer/RNG checkpoints, but no full trained student has been accepted. The real
teacher-to-student development smoke reaches the checked INT4 ONNX Runtime path; it does not count as
quality evidence. No complete trained candidate or held-out device measurement exists yet. Synthetic
smoke results never count as Phase 0 evidence.

Phase 1 remains blocked until all original gates pass: tap relative error reduction, context-sensitive
and valid-word gains, false-correction ceiling, swipe top-1/top-3 strata, 80/200 ms p95 budgets, 64 MiB
added peak memory, reproducibility, licensing, and the physical GrapheneOS matrix.

The repository may ship core keyboard improvements before then, but releases must describe the neural
components as unavailable and must keep the classic/geometric fallback fully usable.
