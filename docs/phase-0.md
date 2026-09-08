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
coverage and separate latency gates per environment. Context-dependent corrections and unchanged
valid words have independent minimums so easy keep cases cannot dilute the neural quality gate. The
tap-corpus preparation contract is documented in `docs/phase-0-tap-corpus.md`; it hashes collection
sessions, binds source licensing and consent metadata, rejects synthetic spatial rows, and cannot
label an undersized corpus release-eligible. The
swipe source, split, training, and ONNX export contracts are executable and tested, and
the exact million-gesture corpus manifest is committed. The context student architecture, tokenizer
contract, INT4 exporter and hash-locked toolchain are also executable. Its pinned 68,748-sentence
English/German corpus uses whole-session 90/5/5 splits and produces a deterministic 16,384-token BPE
after runtime-parity checks. Its offline teacher-scoring pipeline completed all 68,748 records with no
rejection, and the committed distillation manifest binds the exact data, teacher, tokenizer, tools,
toolchain, output hashes and reconciled metrics. Its vectorized student trainer has deterministic
mid-epoch model/optimizer/RNG checkpoints, but no full trained student has been accepted. The real
teacher-to-student development smoke reaches the checked INT4 ONNX Runtime path; it does not count as
quality evidence. A full context candidate has now completed training and INT4 export, and its
synthetic kernel-parity check passes on the arm64 Android emulator. No accepted full-quality or
physical-device measurement exists yet. Synthetic smoke results never count as Phase 0 evidence.

Phase 1 remains blocked until all original gates pass: tap relative error reduction, context-sensitive
and valid-word gains, false-correction ceiling, swipe top-1/top-3 strata, 80/200 ms p95 budgets, 64 MiB
added peak memory, reproducibility, licensing, and the physical GrapheneOS matrix.

The repository may ship core keyboard improvements before then, but releases must describe the neural
components as unavailable and must keep the classic/geometric fallback fully usable.

## Current evidence and remaining gates, 2026-09-08

The completed context training/export/diagnostic work is recorded in the
[context model card](../models/context/MODEL_CARD.md). Its two Linux exports are byte-identical,
and the exact Linux artifact has full distillation validation/test results and Android kernel
parity. Those results do not establish correction gains on human tap errors.

| Required evidence | Current state |
|---|---|
| 3,000 held-out human spatial tap errors | 2,113; another 887 required |
| Valid-word correction and keep coverage | 6,132 corrections and 4,635 keeps; count minimums met, quality not measured |
| 500 split/join cases | None in the combined corpus |
| Contraction/personal/compound coverage | 3,244 contraction cases; personal and compound coverage missing |
| Swipe absolute quality and difficult strata | Offline native-lexicon union remains below absolute top-1/top-3 gates; full live fusion is not qualified |
| End-to-end latency and combined added peak memory | Not qualified; diagnostic snapshots and generous-deadline replays are insufficient |
| Required device matrix | Forced-low-RAM emulator checks exist; stock physical Android and physical GrapheneOS evidence are absent |
| Model/runtime release reproducibility | Both fixed model exports repeat on Linux; two clean Linux runtime AARs and signed model-pack builds are not established |
| Signing and publication | No accepted production key or model-qualified release; artifacts remain local and unsigned/unaccepted |

The [tap corpus report](phase-0-tap-corpus.md), [swipe evaluation](swipe-ctc-evaluation.md),
[device tests](testing.md), and [APK build evidence](release/abi-packaging.md) retain exact scope
and limitations. Missing human cases cannot be supplied by duplicating examples, manufacturing
touches, or changing session splits to increase test counts. Physical-device checks cannot be
replaced by emulator fingerprints. Phase 0 remains open.
