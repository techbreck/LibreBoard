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
mid-epoch model/optimizer/RNG checkpoints. A shared-session four-epoch student has completed
training and two byte-identical Linux INT4 exports; distillation ranking diagnostics are not human
correction quality. The real teacher-to-student development smoke reaches the checked INT4 ONNX
Runtime path and does not count as quality evidence. Earlier Android kernel-parity timings used
other weights and do not qualify the canonical export. No accepted full-quality or physical-device
measurement exists yet. Synthetic smoke results never count as Phase 0 evidence.

Phase 1 remains blocked until all original gates pass: tap relative error reduction, context-sensitive
and valid-word gains, false-correction ceiling, swipe top-1/top-3 strata, 80/200 ms p95 budgets, 64 MiB
added peak memory, reproducibility, licensing, and the physical GrapheneOS matrix.

The repository may ship core keyboard improvements before then, but releases must describe the neural
components as unavailable and must keep the classic/geometric fallback fully usable.

## Current evidence and remaining gates, 2026-09-08

The completed context training/export/diagnostic work is recorded in the
[context model card](../models/context/MODEL_CARD.md). Its two Linux exports are byte-identical,
and the exact Linux artifact has full distillation validation/test results. Those results do not
establish correction gains on human tap errors. Android kernel parity was measured against the
earlier independent-split weights, not against the shared-session candidate
`8566382e57ea5a9af76ddf600e189572d62fcc8b227de8415ce88ad0e172022b`; the model card records that
warning, and the parity fixture must be repeated against the pinned hash.

| Required evidence | Current state |
|---|---|
| 3,000 held-out human spatial tap errors | 3,462 (2,113 phone + 1,349 smartwatch); corpus minimum met |
| Valid-word correction and keep coverage | 6,132 corrections and 4,635 keeps; count minimums met, quality not measured |
| 500 split/join cases | 509 ITE letter-regroup cases; minimum met |
| Contraction/personal/compound coverage | 3,244 human contractions plus 360/312 project-authored personal/compound held out; all kinds represented |
| Swipe absolute quality and difficult strata | Offline native-lexicon union remains below absolute top-1/top-3 gates; full live fusion is not qualified |
| End-to-end latency and combined added peak memory | Not qualified; diagnostic snapshots and generous-deadline replays are insufficient |
| Required device matrix | Physical GrapheneOS tap and swipe rows were collected on 2026-09-11, but they are diagnostics: they measured the superseded independent-split context export, and their tap results came from the defective schema-3 harness. Stock physical Android and a qualifying low-RAM run are absent. No environment currently holds qualifying evidence |
| Model/runtime release reproducibility | Both fixed model exports repeat on Linux; two independent Linux development-operator runtime AARs are byte-identical. That is not a model-qualified runtime. Signed model-pack builds are not established |
| Signing and publication | No accepted production key or model-qualified release; artifacts remain local and unsigned/unaccepted |

Three measurement defects are fixed in the harness, and each invalidates previously collected tap
evidence rather than being repairable by rescoring it. First, the harness hoisted the typed word to
the front of every slate while the evaluator scored rank one, so all four tap systems reported an
identical 25.265% top-1 and 0% on tap errors — a measurement of the harness, not the keyboard.
Measurement schema 4 now records production's own commit decision per system and the evaluator scores
that; see [the schema](phase-0-dataset-schema.md). Second, the harness built each row with
`WordComposer.setComposingWord`, which marks the composition resumed, and production never
auto-corrects a resumed word; every measured row was therefore structurally uncorrectable, which is
why the correction rate stayed at 0% even on the classic baseline until rows were replayed as key
events. Third,
[`phase-0-artifact-pin.json`](phase-0-artifact-pin.json) pins the shared-session candidate, and
`tools/run_phase0_measurement.py --artifact-pin` fails by name when a run injects a rejected export,
verifies the APK installed on the device, requires every gated instrumented test to pass, and checks
that every pulled row belongs to the run before saving it. The physical release checklist stays
blocked until the exact release APK completes it.

### Planning the qualifying device matrix

The three environments share one dataset, so their row budgets add rather than repeat. A 500-row cap
per device yields 1,500 swipes and cannot satisfy the 5,000-swipe minimum, nor the 500-per-stratum
minimums for `short`, `medium`, `long`, `clean`, `sloppy`, `very_sloppy`, `double_letter` and
`return_trip`; plan disjoint shards sized to the totals, not to the per-device floor of 100. Example
ids must stay unique across the whole matrix — the evaluator rejects duplicates, so a full emulator
corpus cannot be pooled with a hardware shard drawn from the same rows. The low-RAM slot needs a
device that actually reports `isLowRamDevice: true` with at most 2,048 MiB; the 12,500-row emulator
rehearsal reports false and carries no model or APK hashes, so it cannot fill that slot.

The [tap corpus report](phase-0-tap-corpus.md), [swipe evaluation](swipe-ctc-evaluation.md),
[device tests](testing.md), and [APK build evidence](release/abi-packaging.md) retain exact scope
and limitations. Missing human cases cannot be supplied by duplicating examples, manufacturing
touches, or changing session splits to increase test counts. Physical-device checks cannot be
replaced by emulator fingerprints. Phase 0 remains open.
