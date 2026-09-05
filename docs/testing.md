# Testing and release evidence

LibreBoard uses layered tests because no single test environment can prove an IME is private,
correct, fast, and compatible with GrapheneOS.

## Host tests

`./gradlew testRunTestsUnitTest` exercises engine contracts, field policy, candidate fusion,
language lock, geometric swipe, model-manifest and ONNX validation, persistence, deadlines, and
stale-result handling. `python3 -m unittest discover -s tools/tests` exercises the Phase 0 evaluator
and release-evidence verifier. These tests run for every pull request.

## Android instrumentation

`./gradlew assembleDebugNoMinifyAndroidTest --max-workers=1` compiles the on-device suite in CI.
`./gradlew connectedDebugNoMinifyAndroidTest --max-workers=1` runs it on an attached unlocked
Android device or emulator. Serial Gradle workers avoid races in the inherited multi-ABI `ndk-build`
archive tasks.
The suite checks the installed package rather than only source XML: merged permissions, backup and
cleartext flags, the IME service permission/direct-boot flag, the Android `EditorInfo` policy matrix,
and real credential-encrypted SQLite personalization behavior.

The connected suite must expand with editor fixtures for composing reconciliation, cursor movement,
correction rejection, WebView, terminal single-commit behavior, model failure, clipboard expiry,
backup/wipe, language lock, and latency collection as those paths land.

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
