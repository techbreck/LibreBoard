# LibreBoard status audit — 2026-09-12

LibreBoard remains in Phase 0. The core implementation, model tooling, corpus preparation and reproducible development runtime have advanced, but the neural keyboard is not release-qualified and Phase 1 acceptance remains blocked. The last-session update accurately describes collected GrapheneOS data, but it overstates how close those measurements are to usable release evidence.

Audited checkout: `main`, `bf23d8ef03457c5824de15f35ee8533b56acbdf0`. The pre-existing uncommitted change in `Phase0MeasurementInstrumentedTest.kt` raises lexicon readiness from 120 to 600 seconds and adds elapsed-time logging. It was preserved. This audit changes no implementation or device settings. Only `emulator-5554` was connected; the physical phone's current configuration could not be rechecked.

| Area | Verified state |
|---|---|
| Core keyboard and engine | Implemented; 359 JVM tests pass, including policy, fusion, decoders and storage contracts. This does not replace physical IME behavior checks. |
| Tap corpus | 18,654 held-out rows: 3,462 tap errors, 10,767 valid-word rows, 509 spacing rows and 3,916 lexical rows. The previously missing corpus counts are supplied. |
| Context candidates | Current committed evidence records completed large and small shared-session students and repeatable Linux exports. Neither is accepted for release; the small student remains experimental. |
| Development ONNX Runtime | Rehashed both Linux A/B output AARs: both match `54118ac8e37bc4833d32e2cb197ef6bd251f56e5aa9e4e51c3899fb616210d88`. This is development-operator reproducibility. |
| GrapheneOS measurements | 6,218 tap rows and 500 swipe rows exist, are schema-valid, and exactly match tap shard 0/3 and the first 500 rows of swipe shard 0/3. |
| Reference device matrix | Incomplete. Saved metadata contains only GrapheneOS. No stock-device Phase 0 result was located. The large emulator rehearsal explicitly reports `isLowRamDevice: false`. |
| Release | No accepted production model/key or completed model-qualified release evidence. The physical GrapheneOS release checklist remains unfilled. |

The saved owner-profile inventory records user 0, Pixel 9 Pro XL, API 37, build `2026091001`, and no installed Google Play packages. Its “LibreBoard not installed” blocker concerns the release package; these measurements used the debug package. This confirms the recorded posture, not the phone's present state. The final build number is `2026091001`, despite an earlier `2026090701` reference in the pasted update. User-profile cleanup and present membership of users 10/11/12 were not reverified.

Recomputed GrapheneOS results use the evaluator's nearest-rank percentile definition:

| Measurement | Result | Interpretation |
|---|---:|---|
| Tap `fused_neural` p50 / p95 / p99 | 41.09 / 71.14 / 119.70 ms | Suggestion-call p95 is below 80 ms for this workload; does not establish complete IME latency or successful neural inference. |
| Swipe `fused_swipe` p50 / p95 / p99 | 216.39 / 317.05 / 349.65 ms | Exceeds the 200 ms p95 threshold. |
| Swipe fused top-1 / top-3 | 3.8% / 5.0% | Far below 90% / 95% thresholds on this slice. |
| Standalone geometric p95 / availability | 23,800.71 ms / 500 of 500 TIMEOUT | Serious deadline overrun in the measured replay. |
| Standalone CTC p95 / availability | 16,148.90 ms / 500 of 500 TIMEOUT | No AVAILABLE CTC result in this slice. |
| Saved memory metric | 208.70 MiB | Exceeds 64 MiB; metric implementation has attribution/sampling limitations below. |

The update's approximately 318 ms swipe p95 is directionally correct; the repository evaluator calculates 317.05 ms from the saved file. The 87.8 MiB tap-only figure cannot be independently reconstructed from the consolidated metadata, which retains the later 208.70 MiB value.

The audit found four evidence blockers beyond simply finishing two more devices:

1. **Tap correction quality is mismeasured.** `Phase0MeasurementInstrumentedTest.kt:219` prepends the raw input before every fused prediction slate; the baseline does the same around line 313. `evaluate_engine.py:236` scores rank one as the prediction. Every one of the 6,218 GrapheneOS rows has raw input first in all four systems. Consequently all four show exactly 25.265% overall top-1 and 0% top-1 on the 1,134 tap errors. This is a harness defect, not evidence that the keyboard never corrects anything. Production `InputLogic.java:660` uses `mWillAutoCorrect` and the correction index to select the actual committed word. Fix the measurement contract around actual correction/keep behavior and candidate ranking; merely removing one `add(raw)` is insufficient because the production suggestion list also includes the typed word. Existing JSONL does not retain that commit decision, so tap acceptance requires a rerun. The full 18,654-row emulator rehearsal has the same defect.

2. **The GrapheneOS run used the historical context artifact.** Its metadata and local model bytes match `b6dde70259d75790d8685884a50b9939f88befd55405fa3e78f6d8edebc033df`, the older macOS independent-split export documented in `models/context/MODEL_CARD.md:151`. Current shared-session large-model evidence names `8566382e57ea5a9af76ddf600e189572d62fcc8b227de8415ce88ad0e172022b`; the experimental small model is `9f09d7b37ff4c7f1bc0a7c7e5918ea860eb26e1a809c9758207ade52f22ed0f5`. The model card explicitly rejects independent splits for joint evaluation. `docs/models/joint-session-splits.md` records exposure of 47,931 CTC test paths to old context training and 2,226 to validation. These runs cannot qualify the corrected candidate. Standalone CTC/geometric results remain useful component diagnostics. The measured APK and CTC hashes do match retained local files.

3. **Schema-valid rows are not complete or sufficiently bound acceptance evidence.** `validate_metadata` rejects the saved metadata with “metadata requires exactly the three reference environments.” The 12,500-row emulator swipe rehearsal cannot fill the low-RAM slot: its own API result is false, and its metadata lacks model/APK hashes. Reusing the full emulator corpus alongside a hardware shard also requires removing duplicate example IDs and preserving valid artifact/run bindings. A 500-row cap on each of three devices would yield only 1,500 swipes, below the 5,000 minimum; difficult-stratum counts must also pass. The driver accepts artifact hashes from arguments rather than verifying the installed/injected bytes and still relies on a blocking `am instrument -w`. It does not parse test-success status or verify pulled run IDs/counts before saving. Harden completion, reconnect recovery and artifact binding before more expensive runs.

4. **Latency and memory need better diagnostic attribution.** Standalone geometric and CTC decoding run concurrently with 1,500 ms deadlines, yet the physical results take many seconds. This proves overruns in that replay, not a controlled universal hardware-versus-emulator speed ratio. The synchronous lexicon cache/rebuild path and deadline coverage warrant profiling, but no single root cause is established here. Tap latency times `getSuggestedWords`, excluding editor commit/UI work and fixture readiness. The memory field is the largest sparsely sampled whole-process PSS increase after fixture setup, including non-neural work; it is not an isolated neural peak and may miss transient peaks. These limitations do not establish a memory pass. The 600-second readiness change helps the fixture wait but does not optimize startup or typing.

Several status documents need reconciliation. `README.md` still lists missing corpus coverage; `docs/models/current-work-progress.md:30` still lists 887 missing spatial examples and missing spacing/personal/compound coverage. Those counts are stale. `docs/phase-0.md:57` should now distinguish collected GrapheneOS diagnostics from missing qualifying release evidence. Its claim about canonical-model Android parity also needs alignment with the model card's warning that older Android measurements used different weights. Keep the physical release checklist blocked until the exact release APK completes it.

The next coherent work sequence is: repair and test the tap measurement semantics; pin the intended shared-session model plus tokenizer, runtime, APK and test APK; add run completion/provenance checks; then use short diagnostic runs to address swipe deadline overruns and memory before repeating the full disjoint device matrix. Finish signing, reproducible model-qualified packaging and the complete physical GrapheneOS behavior checklist only when their prerequisites pass. Preserve the current raw runs as diagnostic evidence.

Validation performed during this audit:

- Source release checks: pass.
- Retained measured debug APK release-verifier checks: pass. Its SHA-256 matches the GrapheneOS metadata; verifier success is packaging validation, not product acceptance.
- Python suite: 324 tests run, 13 skipped, no failures (311 executed successfully).
- JVM suite: 359 tests, no failures/errors/skips. The first offline attempt lacked one cached dependency; the normal retry passed without loosening dependency verification.
- Both GrapheneOS and both full emulator JSONL files parse through the current evaluator. Exact hardware shard membership was checked against the source corpora.
- `git diff --check`: pass.

Recomputed metrics and raw-file hashes are saved locally in `build/reports/status-audit-2026-09-12.json`. Raw measurements and consolidated metadata remain under `build/reports/` and are ignored by Git; preserve an immutable evidence bundle before cleaning that directory or moving computers. This audit report is the only new source-controlled-path file; no commit or publication was performed.

## Resolution, 2026-09-12 (later session)

Every audit finding that could be fixed in the repository is fixed. Nothing above is retracted; the
tap-measurement finding turned out to understate the problem, and one blocker is confirmed to need an
artifact this checkout does not contain.

### The tap measurement had four defects, not one

The audit's finding 1 is correct and is fixed: measurement schema 4 records, per system, the commit
decision production would apply when the word is terminated — `Suggest.commitDecisionOf` for the
fused systems and the new `Suggest.classicCommitDecision` for the baseline, both reading production's
own `shouldBeAutoCorrected` rather than a harness copy. `evaluate_engine.py` scores tap top-1 from
that decision, takes top-3 as the committed surface plus the first three strip entries, and counts a
false correction only when the committed surface differs from `raw`. The harness no longer hoists
`raw`; the exact surface must still appear in the slate, but production already seats it at index 0.

Repeating the measurement with only that fix still produced 0% auto-correction on all four systems,
including the classic baseline, which is not a plausible product result. The cause is a second
defect: the harness composed each row with `WordComposer.setComposingWord`, which ends by setting
`mIsResumed`, and production never auto-corrects a resumed word — that is the "user tapped back into
an existing word" case. Every measured row was structurally uncorrectable. Rows are now replayed as
key events with their real touch coordinates, as typing does, and the harness reloads the
auto-correction threshold the way `LatinIME.loadSettings` does.

Repeating the measurement after those two fixes still reported 0% top-1 on tap errors, and the
reason was a third defect. Of 58 tap errors, 55 had the target somewhere in the slate and 47 had it
at index 1 — the auto-correction slot — yet only one row committed a correction. The fixture built
its `EditorInfo` as a bare `TYPE_CLASS_TEXT`, which requests neither auto-correction nor multi-line
input. With `PREF_MORE_AUTO_CORRECTION` defaulting false that makes `mAutoCorrectEnabled` false, and
`SettingsValues` then sets the auto-correction threshold to `Float.MAX_VALUE`. Only whitelist entries
could ever be committed, which is exactly what the data showed: all 35 corrections in that run were
whitelist expansions such as `cant` to `can't` and `im` to `I'm`, and every ordinary spatial
correction like `qemt` to `went` was blocked. The fixture now models a composing field that requests
auto-correction, and asserts `mAutoCorrectEnabled` at fixture build and after every configuration
change, so this cannot regress silently.

The fourth defect was in the evaluator and was introduced by this session's own work: the
false-correction metric was rewritten to compare the committed surface to `raw` exactly, where the
original compared them normalized. Accuracy is normalized, so auto-capitalizing `thursday` to
`Thursday` scores as a hit; counting the same commit as a false correction penalized the keyboard for
behaviour the accuracy gate rewards. It inflated the measured false-correction rate to 33%, and every
one of those rows was a proper-noun capitalization. The comparison is normalized again.

Measured on 400 held-out rows against the superseded context export, diagnostic only. Physical
GrapheneOS (`grapheneos-tap-ac`) and the forced-low-RAM emulator (`lowram-tap-ac`) agree exactly on
every quality figure, which is what a deterministic replay of one corpus should do:

| System | top-1 | top-3 | top-1 on tap errors | auto-corrected | false correction | p95 hardware | p95 low-RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| heliboard | 51.75% | 68.50% | 67.24% | 47.25% | 0.00% | 62.0 ms | 21.1 ms |
| fused | 51.50% | 68.50% | 65.52% | 46.75% | 0.00% | 47.5 ms | 21.0 ms |
| fused_personal | 51.50% | 69.25% | 65.52% | 46.75% | 0.00% | 72.6 ms | 24.8 ms |
| fused_neural | 51.50% | 69.25% | 65.52% | 46.75% | 0.00% | 72.9 ms | 25.4 ms |

Two results matter more than the totals. The tap relative-error-reduction gate compares `fused`
against `heliboard` on tap errors and requires at least +20%; the measured value is **-5.26%**, so the
fusion layer currently commits slightly *worse* tap corrections than classic HeliBoard. And
`fused_neural` is identical to `fused` to the digit on every quality metric, so the neural path
contributes nothing at neural strength 50. Both are measured against the superseded export, so they
diagnose the fusion layer rather than the pinned candidate, and both are the opposite of what the
schema-3 numbers implied. The false-correction increase gate passes at +0.00%. The result is also a real product finding rather
than an artifact: the target reaches the top-3 slate 68.5% of the time but is almost never committed,
so the 20% tap relative-error-reduction gate is far from passing and the neural path currently
changes neither the commit rate nor accuracy. That is a measurement worth acting on; it is not a
release result, and the 400-row diagnostic is far below the release-size minimums.

### The superseded model is now refused by name

`docs/phase-0-artifact-pin.json` pins the shared-session candidate
`8566382e57ea5a9af76ddf600e189572d62fcc8b227de8415ce88ad0e172022b` with its tokenizer, the swipe
graph and the runtime AAR, and names each rejected export with the reason it is rejected.
`run_phase0_measurement.py --artifact-pin` hashes the injected bytes on the host and again on the
device and fails by name before touching the device; this was verified against the real superseded
export. Metadata now carries an `artifactPin` naming the candidate the evidence claims to qualify,
the evaluator refuses metadata without one, and refuses a pin whose hashes disagree with the
measurement's own. A passing report therefore cannot omit which candidate it qualified.

**The pinned bytes are not in this checkout.** Every retained context export was hashed; only the
rejected `b6dde702…` and `53e1a1d2…` artifacts are present. The shared-session candidate must be
re-exported in the Linux container before any qualifying device run. Both diagnostic runs below
deliberately ran without the pin against `b6dde702…` and are labelled accordingly.

### Run completion and provenance

The driver no longer trusts its own arguments or `am instrument`'s exit status. It verifies the APK
actually installed on the device against the hash bound into metadata, hashes every injected model
both locally and on the device, clears the previous run's output files before starting, parses the
runner's per-test status codes so a crash, a failure, or a *requested* test skipping is rejected
while the unrequested replay test may skip, waits for the device before pulling, pulls bytes rather
than lossily decoded text, and checks that every pulled row carries this run's id, environment and
schema before anything is written. Both behaviours were exercised for real during this session: a run
whose ONNX session was unavailable and a run during which the phone dropped off USB were both
rejected with nothing saved.

Losing two hardware runs to a dropped cable also showed that a blocking `adb shell am instrument -w`
ties a multi-hour run to one USB connection: when the link dies the shell dies and takes the
instrumentation with it. The driver now writes the instrumentation command to a script on the device,
launches it detached under `nohup`, and polls for a completion marker, tolerating and waiting out a
disconnect rather than discarding the run. The successful GrapheneOS rerun was collected this way.

### Memory and latency attribution

`peakAddedNeuralMemoryMiB` is unchanged in definition but is no longer the only number recorded. The
sidecar now carries a breakdown and an explicit statement of what the figure does and does not
include. On GrapheneOS hardware the headline fell from 208.70 MiB to 81.36 MiB after the fixture
baseline was taken correctly, and the breakdown shows why the headline overstates the neural path:
whole-process PSS was 228,867 KiB after fixture setup, peaked at 309,721 KiB with the neural path
disabled, and at 312,181 KiB with it enabled. The neural path accounts for roughly 2.4 MiB over the
classic peak; the rest is classic suggestion work. This still does not establish the 64 MiB gate —
it is a sparse whole-process sample, not an isolated neural peak, and the sidecar says so. The
sidecar also records fixture readiness separately and states that `latencyMs` times the suggestion
call only, excluding editor commit, UI and fixture work. That separation immediately produced the
profiling number the audit asked for: the one-time swipe-lexicon build takes 83,687 ms on GrapheneOS
hardware against 15,942 ms on the emulator, while main-dictionary initialisation takes 25 ms. The
hardware figure is what the uncommitted 600-second readiness deadline exists for, and it is a
startup cost no suggestion-call latency measurement can see.

The hardware rerun also repeated the memory measurement on the corrected harness: 73.63 MiB headline,
from a 224,324 KiB fixture baseline to a 298,173 KiB peak with the neural path disabled and 299,721
KiB with it enabled — about 1.5 MiB attributable to the neural path, consistent with the 2.4 MiB seen
in the first hardware run.

### Documents reconciled

`README.md` and `docs/models/current-work-progress.md` no longer claim missing corpus coverage; the
held-out corpus supplies 18,654 rows and clears every minimum, which was verified directly from
`build/evaluation-data/combined-tap-v2/test.jsonl`. `docs/phase-0.md` distinguishes collected
GrapheneOS diagnostics from qualifying evidence, aligns the Android-parity claim with the model
card's warning, and adds the matrix-planning constraints the audit identified. The model card
cross-references the pin. `docs/phase-0-dataset-schema.md` documents schema 4 and the pin.

### Verification

359 → 366 JVM tests pass with no failures, including new tests for both commit-decision entry points
and for the resumed-composition trap. 311 → 333 Python tests execute with 13 optional-toolchain
skips and no failures, including the commit-contract, run-completion and artifact-pin suites. Source
release checks pass. `git diff --check` passes.

### The low-RAM slot is filled; stock hardware and the pinned model are not

The `LibreBoard_API_36_AOSP_LowRAM` AVD was booted with a writable system image and
`ro.config.low_ram=true`, and now reports `isLowRamDevice: true` at 1,974 MiB on API 36 — inside the
evaluator's 2,048 MiB ceiling and the first valid environment record that slot has had. Setting the
property is required: RAM size alone does not make `ActivityManager.isLowRamDevice()` true, and the
emulator's `-prop` flag does not take for a read-only build property.

Stock physical Android hardware cannot be produced from this machine; only the GrapheneOS Pixel and
emulators are attachable.

The pinned context model is further out of reach than "re-export the bytes". Every retained artifact
belongs to the rejected independent-split run: the only full context weights are
`95384df3…`, no training report matching the shared-session run `df174d41…` exists, and the retained
teacher-scored corpus counts 61,620 train / 3,364 validation / 3,764 test, which the model card
identifies as the independent split — the shared-session corpus is 61,604 / 3,253 / 3,637. Reaching
`8566382e…` therefore requires regenerating the shared-session split, re-scoring it with the retained
teacher, re-running the four-epoch distillation, and only then the Linux export. The export image
itself is retained, so that final step still reproduces.

### Still blocked

Phase 1 acceptance remains blocked, and the device matrix is emptier than the audit recorded, because
the previously collected tap rows are now known to be invalid for two independent reasons. Remaining
work, in order: re-export the pinned shared-session model; investigate why the keyboard reaches the
right candidate but does not commit it; then run the full disjoint matrix across stock hardware,
GrapheneOS and a genuine low-RAM device under the pin. The physical GrapheneOS release checklist
stays blocked until the exact release APK completes it.
