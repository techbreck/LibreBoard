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

The Phase 0 low-RAM emulator must be cold-booted with the emulator's supported low-RAM mode rather
than an injected read-only property. For the checked-in API 36 AOSP AVD, use:

```sh
$ANDROID_SDK_ROOT/emulator/emulator -avd LibreBoard_API_36_AOSP_LowRAM \
  -lowram -no-snapshot -no-window -no-audio -no-boot-anim
adb shell getprop ro.config.low_ram
python3 tools/capture_android_device.py --serial emulator-5554 \
  --output build/device-evidence/api36-low-ram-emulator.json
./gradlew connectedDebugNoMinifyAndroidTest --max-workers=1 \
  -Pandroid.testInstrumentationRunnerArguments.libreboardRequireLowRam=true
```

The property command must print exactly `true`; the capture also records `/proc/meminfo` as rounded-up
MiB. The opt-in instrumentation assertion independently checks `ActivityManager.isLowRamDevice()`
and proves that the real device policy disables the context model while retaining CTC swipe. A
renamed AVD or a RAM setting alone does not establish low-RAM mode and must not be submitted as
low-RAM evidence.
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

The connected suite must expand with editor fixtures for composing reconciliation, cursor movement,
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

```sh
./gradlew testRunTestsUnitTest assembleDebugNoMinifyAndroidTest --max-workers=1
python3 -m unittest discover -s tools/tests
python3 tools/verify_release.py --source
```

The full artifact/evidence command is documented by `python3 tools/verify_release.py --help` and is
run only when the physical and reproducible-build inputs exist. Checked-in templates remain marked
blocked until a real release candidate passes.
