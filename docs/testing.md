# Testing and release evidence

LibreBoard uses layered tests because no single test environment can prove an IME is private,
correct, fast, and compatible with GrapheneOS.

## Host tests

`./gradlew testRunTestsUnitTest` exercises engine contracts, field policy, candidate fusion,
language lock, geometric swipe, CTC feature tensors and lexicon beam decoding, model-manifest and
ONNX validation, bounded context tensors, persistence, deadlines, and stale-result handling.
Clipboard-policy tests require the platform sensitive-content marker, restricted fields, and
incognito mode to veto history capture independently. Context tests enforce field-policy gates,
candidate/language score identity, masks, field classes, tokenizer
bounds, strict BPE schema/merge/special-token validation, Unicode normalization, and malformed/runtime
fallback. They also verify bounded prefix-token reuse, privacy resets, and generation-safe handling
when a reset races in-flight tokenization. Runtime adapter tests lock exact tensor names/types/shapes, dynamic batch dimensions,
direct-buffer invocation, output bounds, native-wrapper cleanup, core-only absence, and hard deadline
fallback. Official-model policy tests lock the independent swipe/context size and parameter ceilings,
the exact reduced-operator allowlist, bounded RSA key reads, and the 3072-bit/exponent-65537 trust
contract. Live-path tests prove bounded immutable request and language-lock propagation, including
hard spacebar/language-key selections that survive word boundaries while automatic editor-locale
switches do not create hard locks,
0–100 neural weighting, calibrated commit selection, classic ranking on timeout, cooperative and
non-cooperative timeout circuit breaking, delayed native-owner cleanup, invalid preference fallback,
restricted-field bypass, and safety-veto precedence. Swipe live-path tests additionally prove
revision-safe static lexicon caching, CE-only personal vocabulary gates, negative-score multilingual
locking, CTC/geometric provenance union, missing-model fallback, non-cooperative timeout circuit
breaking, delayed native-owner cleanup, and batch single-commit metadata. Registry tests prove
swipe/context activation and rollback slots cannot collide, and explicit runtime rejection restores
the last-known-good model. Runtime-bootstrap tests also require core-only manual import to fail before
opening an untrusted URI, low-RAM devices to reject context models through both automatic and manual
activation paths, and bundled activation to occur only for a missing model or app update.
Retained input-logic tests exercise
terminal policy through the editor adapter: tap input creates no composing span, swipe commits
directly, and duplicate asynchronous tail delivery cannot commit the same gesture twice. Lexical
tests require contractions,
split/join hypotheses, and
German compound evidence to stay within one explicitly tagged dictionary; they also cover casing,
language locks, deadline exits, generated provenance, raw-word vetoes, and multilingual source-order
invariance. CTC tests cover
double letters, return-trip words, canonical contractions, language locks, malformed tensors,
German-only popup-letter gesture aliases, runtime failure, and deadline fallback. Geometric tests
exercise the same German surface preservation and reject crossed-key count as a word-length proxy.
Backup tests additionally exercise ZIP path traversal,
canonical aliases, entry and expanded-size limits, strict typed-settings parsing, preservation of excluded private
state, and process-death recovery on both sides of the restore commit point. Learned-data tests
require idempotent removal without deleting clipboard, model, or explicit dictionary fixtures.
`python3 -m unittest discover -s tools/tests` exercises the Phase 0 evaluator and release-evidence
verifier. It also locks the ONNX Runtime source-build schema, ABI set, SDK levels, CPU-only flags,
strict reduced-operator configuration parser, immutable model-source manifest/fetch behavior,
immutable GitHub evaluation-source fetches, the Google TSI word-replay adapter, session-separated
swipe and tap-evaluation preparation, split/hash reproducibility, bounded and
licensed tap-source provenance, rejection of synthetic spatial evidence, model architecture/parameter count,
CTC utilities, production-parity prefix-beam semantics, deterministic held-out swipe sampling,
path-based length estimation, and ONNX operator-file ordering. These dependency-free tests run for every pull
request.

After a full swipe export, `build/model-venv/bin/python tools/evaluate_swipe_ctc.py` measures the
real ONNX graph with the production CTC prefix-beam rules on 5,000 deterministically selected
held-out gestures, including at least 500 from each required stratum. Its corpus-derived lexicon is
built only from train and validation targets. The report preserves the raw prefix-beam ranking,
measures the production geometric cost and applies the production scorer's z-normalized
spatial/static-frequency weighting to the isolated and normalized-union slates. The resulting host
report is a model diagnostic, not Phase 0 evidence; device runtime, production dictionaries,
complete scorer fusion, memory, and latency remain part of the bound Android measurements.

The Phase 0 evaluator binds each test row to a unique metadata-declared device run. Release-sized
evidence requires at least 100 tap and 100 swipe latency samples from each stock Android,
GrapheneOS-without-Play, and low-RAM run, and enforces the p95 budgets independently on all three.
It also requires 500 true context corrections and 500 unchanged valid words, evaluating neural gain
and false-correction behavior on the appropriate strata instead of a blended easy-case score.
Added neural memory is recorded for each environment and the 64 MiB gate uses the maximum.
Its input parser also enforces bounded UTF-8 JSONL, path-specific exact system inventories,
normalization-distinct slates, unique strata, and category-scoped correction labels.

The source release gate also validates the Phase 1 Fastlane metadata. Only reviewed English and
German copy may ship initially; titles and length bounds are checked, the zero-network and geometric-
fallback contract must remain explicit, and inherited instructions for proprietary swipe libraries
are rejected. Obsolete upstream changelogs and screenshots are intentionally not redistributed as
LibreBoard release material.

Gradle dependency verification is fail-closed through `gradle/verification-metadata.xml`. The
committed inventory pins SHA-256 checksums for build plugins and every artifact resolved by the host
and Android test graphs; `tools/verify_release.py --source` rejects a missing or implausibly small
inventory, trust bypasses, malformed checksums, and omitted direct dependencies. To change a
dependency, regenerate the inventory from the real verification graph with
`./gradlew --write-verification-metadata sha256 testRunTestsUnitTest assembleDebugNoMinifyAndroidTest
--max-workers=1`, review the coordinate and checksum diff, and then rerun the normal online and
offline release commands. Generated checksums establish artifact integrity and reproducibility, not
publisher identity, so unexplained coordinate or digest changes must not be accepted automatically.

A second CI job installs the hash-locked Python 3.11 CPU model toolchain and runs the same suite with
its optional checks enabled. It instantiates the real 821,121-parameter model, verifies finite
`[1,32,65]` output, prepares a synthetic three-split corpus, trains one deterministic CTC step, stores
safetensors, exports FP16-stored ONNX, runs full ONNX type/shape checking, verifies exact tensor names
and dimensions, derives required operators, and enforces the model-size and manifest hash contracts.
The result is deliberately development-only and is deleted with the test workspace.

The optional context-model package is built separately and receives the same fail-closed release
inspection. `tools/tests/test_verify_release.py` covers both model identities, their bounded archive
schemas, deterministic ZIP metadata, payload hashes, and RSA policy; a release candidate must
additionally run `tools/verify_release.py --model-pack-apk --model-public-key` against the built APK.
The APK check proves that the sidecar requests no permissions, exposes only its fixed read-only
provider, stores exactly one uncompressed `.lbmodel` asset, contains no native code, and carries a
signature trusted by the supplied project RSA key. Core-APK inspection likewise verifies the signed
swipe archive and key together and rejects that pair unless every ABI carries the complete source-
built ONNX Runtime native pair.

## Android instrumentation

`./gradlew assembleDebugNoMinifyAndroidTest --max-workers=1` compiles the on-device suite in CI.
`./gradlew connectedDebugNoMinifyAndroidTest --max-workers=1` runs it on an attached unlocked
Android device or emulator. Serial Gradle workers avoid races in the inherited multi-ABI `ndk-build`
archive tasks.

The Phase 0 low-RAM emulator must use a system image that actually reports Android low-RAM mode.
The emulator's `-lowram` option permits a smaller RAM allocation; it does not by itself prove
`ActivityManager.isLowRamDevice()`. Cold-boot the configured AVD, then check the platform property
before running or labeling low-RAM evidence:

```sh
$ANDROID_SDK_ROOT/emulator/emulator -avd LibreBoard_API_36_AOSP_LowRAM \
  -lowram -no-snapshot -no-window -no-audio -no-boot-anim
adb shell getprop ro.config.low_ram
python3 tools/capture_android_device.py --serial emulator-5554 \
  --output build/device-evidence/api36-low-ram-emulator.json
./gradlew connectedDebugNoMinifyAndroidTest --max-workers=1 \
  -Pandroid.testInstrumentationRunnerArguments.libreboardRequireLowRam=true
```

For product-configured low-RAM evidence, the property must print exactly `true`. Do not override
the read-only property to make a check pass. A debug build can alternatively exercise Android
low-RAM behavior with the supported switch below; always label that configuration explicitly.
The capture also records `/proc/meminfo` as rounded-up
MiB. The opt-in instrumentation assertion independently checks `ActivityManager.isLowRamDevice()`
and proves that the real device policy disables the context model while retaining CTC swipe. A
renamed AVD or a RAM setting alone does not establish low-RAM mode and must not be submitted as
low-RAM evidence.
The emulator's RAM-floor behavior is implemented in AOSP's
[`main-common.c`](https://android.googlesource.com/platform/external/qemu/+/emu-master-dev/android/android-emu/android/main-common.c).
During the 2026-09-08 recheck, the available API 36 AVD booted with `-lowram -memory 1536` but
reported no low-RAM property; the required instrumentation assertion failed as intended. The normal
suite passed on that boot, including the three real editor-connection tests.

For a debuggable AOSP emulator (`ro.debuggable=1`), AOSP's
[`ActivityManager`](https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/app/ActivityManager.java)
also honors `debug.force_low_ram`. Restart Android services after setting it because the framework
caches the value at class initialization:

```sh
adb root
adb shell setprop debug.force_low_ram true
adb shell stop
adb shell start
# Wait for Android services, then run the suite with libreboardRequireLowRam=true.
```

On 2026-09-08, the API 36 emulator passed all eight privacy/storage tests with the low-RAM
assertion enabled. The launch requested 1,536 MiB without `-lowram`; the emulator applied its normal
RAM floor and the guest reported 1,975 MiB. The platform API returned true; real `NeuralDevicePolicy`
disabled context rescoring while keeping CTC eligible. The local raw output is
`build/reports/forced-lowram-instrumentation.log`, and its read-only device capture is
`build/device-evidence/api36-forced-lowram.json`. Capture records all three properties and labels
this mode `debug-forced`; its property-derived boolean still requires the in-app API assertion.
This establishes fallback-policy behavior on a forced low-RAM emulator, not the full Phase 0
latency/memory/quality matrix or a product-configured Android Go image. Clear the debug property
and restart services or cold-boot without a snapshot before an ordinary run.

The suite checks the installed package rather than only source XML: merged permissions, backup and
cleartext flags, the IME service permission/direct-boot flag, the non-exported clipboard-search
activity, the Android `EditorInfo` policy matrix, and real credential-encrypted SQLite behavior for
personalization plus clipboard search, storage limits, pruning, and stale-id handling. It also
verifies that Android's sensitive-clipboard marker reaches the capture-policy veto.
The device suite also verifies that an explicit wipe clears real credential-encrypted personal rows
and legacy rejection files while leaving installed-model and clipboard fixtures intact. It passes a
device-protected context through both private-store entry points after unlock and verifies that the
clipboard and personal databases are still created only in credential-encrypted storage. A rebooted,
locked-device run remains required to prove the complete Direct Boot boundary.

`EditorConnectionInstrumentedTest` exercises the production `RichInputConnection` against Android's
real `EditText` connection on the main thread. It checks repeated composing updates and a single
commit, selection replacement and cursor-cache reconciliation, and a context-access transition that
must permit typing while issuing zero surrounding-text reads. These framework tests do not drive
the installed IME through another application's window and do not establish WebView or Termux
compatibility.

`StaticDictionaryInstrumentedTest` reads the bundled English binary through the native dictionary
iterator. It exercises the 100,000-entry boundary and requires frequent words beyond that boundary
to survive the bounded frequency-ranked swipe index. Host tests cover input-order invariance,
deterministic ties, normalized duplicate replacement and rejection of unusable words.

The connected suite must expand with external-app editor fixtures for composing reconciliation, cursor movement,
correction rejection, WebView, terminal single-commit behavior, model failure, clipboard expiry,
the complete SAF backup/restore UI flow, the clipboard-search interaction flow, language lock, and latency collection as those
paths land.

## Physical GrapheneOS acceptance

Emulators, stock Android devices, and automated instrumentation do **not** certify GrapheneOS. A
Phase 1 release additionally requires the exact reproducible release APK to pass the complete matrix
in [grapheneos.md](grapheneos.md) on currently supported physical Pixel hardware running current
stable GrapheneOS without sandboxed Google Play or weakened compatibility/exploit-protection
settings.

The operator records results using
[`release/grapheneos-evidence-schema.md`](release/grapheneos-evidence-schema.md). The strict release
verifier binds this evidence, raw instrumentation output, Phase 0 results, device identity, model
hashes, and both byte-identical APK builds. Missing or mismatched evidence blocks release.

## Release commands

Universal GitHub and one-ABI F-Droid artifacts use different, fail-closed version-code and native
packaging rules. The build commands, mapping, and matching `--expected-abi` verifier invocation are
documented in [ABI packaging](release/abi-packaging.md).

```sh
./gradlew testRunTestsUnitTest assembleDebugNoMinifyAndroidTest --max-workers=1
python3 -m unittest discover -s tools/tests
python3 tools/verify_release.py --source
```

The full artifact/evidence command is documented by `python3 tools/verify_release.py --help` and is
run only when the physical and reproducible-build inputs exist. Checked-in templates remain marked
blocked until a real release candidate passes.

## Real IME window test

`LiveImeInstrumentedTest` is opt-in because it requires the debug keyboard to be selected as the
system IME. Its non-exported `EditorFixtureActivity` exists only in `debugNoMinify`; release builds
do not include the fixture. After the editor obtains window focus and the restarted IME reconnects,
the test waits for stable keyboard geometry, obtains actual key hitboxes and injects touchscreen
DOWN/UP events through Android. It checks each character, exactly one word/space commit, and
replacement of a selected word with the trailing space preserved. It does not call the input logic
directly or inject hardware text events that bypass the IME.

On a disposable emulator, record the current default IME and `show_ime_with_hard_keyboard` value,
install the debug and test APKs, then run:

```sh
adb shell ime enable org.libreboard.keyboard.debug/helium314.keyboard.latin.LatinIME
adb shell ime set org.libreboard.keyboard.debug/helium314.keyboard.latin.LatinIME
adb shell settings put secure show_ime_with_hard_keyboard 1
adb shell am instrument -w -r -e libreboardRequireLiveIme true \
  org.libreboard.keyboard.debug.test/androidx.test.runner.AndroidJUnitRunner
```

Restore the previous IME and hardware-keyboard setting afterwards. The 2026-09-08 API 36 run passed
12 device tests with the dedicated low-RAM test skipped; the live-window test passed in isolation
and with the complete suite. Raw output is retained at `build/reports/live-ime-full-device-suite.log`.
The corresponding 354 host tests and APK source/privacy/native checks passed. The fixture explicitly
accounts for instrumentation restarting the selected IME and for its opening animation; elapsed
fixture startup time is not a keyboard latency measurement. This same-package editor test does not
replace cross-app WebView/terminal, Direct Boot, or physical GrapheneOS acceptance.

The live fixture also exposes a terminal-marked `EditText` whose real input connection counts
composition calls. On 2026-09-08, this caught a cursor-update path that resumed locally typed
terminal text as a composing word. `restartSuggestionsOnWordTouchedByCursor` now respects the
composition policy. A separate restart check caught cached field attributes surviving changes to
IME privacy/terminal options with an unchanged input type; the cache check now compares the captured
IME options and resolved policy. Both focused host regressions failed before their fixes.

The three live scenarios pass for ordinary typing/selection replacement, terminal typing without
composition, and switching the same editor to terminal mode through `restartInput`. The full host
suite passes 356 tests. These are actual IME-window checks against a debug fixture; they still do
not establish compatibility with a separate terminal application or cold-boot/Direct-Boot behavior.

The API 36 fixture also exposed a binding race after instrumentation restarted the selected
IME. A captured platform dump showed a served editor connection but a null current IME method
and `mBoundToMethod=false`; repeated show requests alone could not reconnect it. The fixture
uses one explicit show path after the editor is active, and the test restarts the editor connection
while waiting for the selected service to create its keyboard view. This setup is specific to
instrumentation killing the same process that hosts the selected IME. Force-stopping the package
also makes Android select its system keyboard; reselect LibreBoard before running this test.
These checks do not establish production cold-start or latency qualification.

## Source-built Android context runtime smoke

`tools/prepare_context_runtime_smoke.py` checks the provenance of a context export and generates
hash-bound synthetic kernel-parity cases for batches of 1, 8 and 32 candidates. Build the debug
APK with the verified runtime AAR, manifest and operator configuration described in
`runtime/onnxruntime/README.md`; no signing key or installed model pack is needed for this isolated
instrumentation fixture. Copy `context.onnx` and `fixture.json` into the debug app's private
`files/context-runtime-smoke` directory, then run:

```sh
build/model-venv/bin/python tools/prepare_context_runtime_smoke.py \
  --export-report build/model-export/context-en-de-v1/export-report.json \
  --output-root build/device-evidence/context-runtime-smoke
# Set CONTEXT_FIXTURE_SHA256 to the printed hash and copy the two generated files before instrumentation.
adb shell am instrument -w -r \
  -e class helium314.keyboard.latin.engine.ContextRuntimeInstrumentedTest \
  -e libreboardRequireContextRuntime true \
  -e contextFixtureSha256 "$CONTEXT_FIXTURE_SHA256" \
  org.libreboard.keyboard.debug.test/androidx.test.runner.AndroidJUnitRunner
adb exec-out run-as org.libreboard.keyboard.debug \
  cat files/context-runtime-smoke/android-report.json
```

The test requires the exact host-generated fixture hash, checks the model hash, loads the actual
packaged Java/native runtime through LibreBoard's production adapter, and compares every finite
score within an absolute tolerance of 0.001. It records the installed APK hash and Android build
fingerprint. It does not register or activate the fixture as a typing model. Remove the private
fixture directory after collecting the report.

The 2026-09-08 arm64 API 36 AOSP emulator run passed all three shapes for the development INT4 model
`6789ab54a0607c9731387d7ac09d818a4f04f509167b94b7a6a28e77d1f14e08`. Single-run adapter times were
65.1, 94.4 and 242.2 ms respectively under concurrent host work. These are neither p95 measurements
nor physical-device evidence, and do not establish the 35 ms context dispatch budget. The retained
local report and originating APK are under `build/device-evidence/context-runtime-smoke/`.

The full trained context graph also passes the same Android kernel-parity test. The test now records
whole-process PSS, allocated native heap and used Java heap before model open, after open, after each
inference and after close. These bounded snapshots expose loading/inference growth without claiming
to capture isolated peak model memory. Full-candidate results and limitations are recorded in
[`models/context/MODEL_CARD.md`](../models/context/MODEL_CARD.md).

## Simultaneous model kernel snapshots

The opt-in `ContextRuntimeInstrumentedTest` also accepts
`-e libreboardRequireCombinedRuntime true -e swipeModelSha256 <exact-hash>` alongside its existing
context-runtime and fixture-hash arguments. Put the checked swipe graph at
`files/context-runtime-smoke/swipe.onnx`. The test validates both graph hashes, keeps both sessions
open, checks context batches 1/8/32 against the host reference, and runs fixed-size synthetic swipe
tensors between those batches. Both sessions close on failure as well as success. Ordinary test runs
still skip all fixture injection.

The report marks `combinedModelSnapshots`, binds both model hashes, and records process PSS,
Java-heap and native-heap snapshots before opening, after each model opens, after each inference and
after both sessions close. These are snapshots, not isolated added peak memory. No static vocabulary,
CTC trie, candidate fusion, editor workload or physical-device qualification is included.

The 2026-09-08 arm64 emulator run with the preserved independent-split context candidate and canonical
Linux swipe graph passed both kernel checks. Process PSS rose from 96,404 KiB to an observed maximum
of 162,073 KiB (64.13 MiB difference), leaving no demonstrated headroom under the 64 MiB combined
budget. This is a diagnostic concern, not a formal peak-memory verdict or a release pass. The exact
report, APK and source/test-APK hash sidecar are retained under
`build/device-evidence/context-swipe-combined-kernel-v1`. Repeat the diagnostic for the corrected
shared-session candidate and measure the complete workload on the required device matrix.

Combined mode now also samples process PSS and native/Java heap counters during loading, inference
and closing, with a requested 5 ms delay between samples. It stops and joins the sampler even when
an assertion fails. The reported maxima can occur at different times and must not be added together;
sampling can miss short peaks and adds overhead, so the result remains diagnostic-only.

The default-runtime sampled run observed a 64.05 MiB process-PSS increase across 126 samples. A local
experiment disabling ORT CPU arenas observed 70.54 MiB across 124 samples, despite a smaller native
heap counter. That experiment was rejected and production allocator settings are unchanged. The
baseline and experiment reports, APKs, instrumentation APKs and source hashes are retained in
`build/device-evidence/combined-sampled-default` and `combined-sampled-no-arena`; the rejected source
patch is `build/reports/rejected-no-arena-runtime.patch`. These measurements still use the preserved
independent-split context candidate and do not qualify the corrected candidate or complete workload.

Two further local runtime experiments also failed to improve the sampled combined-model memory
footprint. A fresh default run observed 64.13 MiB added PSS; enabling
`session.use_device_allocator_for_initializers` observed 64.55 MiB, and disabling prepacking observed
67.60 MiB. All three passed synthetic kernel parity. Both experimental settings were rejected and
the default runtime factory restored. The restored APK has identical ZIP payload entries to the
saved default APK, although the APK container hashes differ; this is restoration evidence, not
byte-for-byte reproducibility. Hash-bound measurements are in
[`runtime-initializer-experiments.json`](models/evidence/runtime-initializer-experiments.json).
These emulator samples use the original context model and do not qualify release memory or latency.
