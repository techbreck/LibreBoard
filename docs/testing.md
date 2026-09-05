# Testing and release evidence

LibreBoard uses layered tests because no single test environment can prove an IME is private,
correct, fast, and compatible with GrapheneOS.

## Host tests

`./gradlew testRunTestsUnitTest` exercises engine contracts, field policy, candidate fusion,
language lock, geometric swipe, CTC feature tensors and lexicon beam decoding, model-manifest and
ONNX validation, bounded context tensors, persistence, deadlines, and stale-result handling. Context
tests enforce field-policy gates, candidate/language score identity, masks, field classes, tokenizer
bounds, strict BPE schema/merge/special-token validation, Unicode normalization, and malformed/runtime
fallback. Runtime adapter tests lock exact tensor names/types/shapes, dynamic batch dimensions,
direct-buffer invocation, output bounds, native-wrapper cleanup, core-only absence, and hard deadline
fallback. Official-model policy tests lock the independent swipe/context size and parameter ceilings,
the exact reduced-operator allowlist, bounded RSA key reads, and the 3072-bit/exponent-65537 trust
contract. Live-path tests prove bounded immutable request and language-lock propagation,
0–100 neural weighting, calibrated commit selection, classic ranking on timeout, cooperative and
non-cooperative timeout circuit breaking, delayed native-owner cleanup, invalid preference fallback,
restricted-field bypass, and safety-veto precedence. Swipe live-path tests additionally prove
revision-safe static lexicon caching, CE-only personal vocabulary gates, negative-score multilingual
locking, CTC/geometric provenance union, missing-model fallback, non-cooperative timeout circuit
breaking, delayed native-owner cleanup, and batch single-commit metadata. Registry tests prove
swipe/context activation and rollback slots cannot collide, and explicit runtime rejection restores
the last-known-good model. Runtime-bootstrap tests also require core-only manual import to fail before
opening an untrusted URI and bundled activation to occur only for a missing model or app update.
Retained input-logic tests exercise
terminal policy through the editor adapter: tap input creates no composing span, swipe commits
directly, and duplicate asynchronous tail delivery cannot commit the same gesture twice. Lexical
tests require contractions,
split/join hypotheses, and
German compound evidence to stay within one explicitly tagged dictionary; they also cover casing,
language locks, deadline exits, generated provenance, raw-word vetoes, and multilingual source-order
invariance. CTC tests cover
double letters, return-trip words, canonical contractions, language locks, malformed tensors,
runtime failure, and deadline fallback. Backup tests additionally exercise ZIP path traversal,
canonical aliases, entry and expanded-size limits, strict typed-settings parsing, preservation of excluded private
state, and process-death recovery on both sides of the restore commit point. Learned-data tests
require idempotent removal without deleting clipboard, model, or explicit dictionary fixtures.
`python3 -m unittest discover -s tools/tests` exercises the Phase 0 evaluator and release-evidence
verifier. It also locks the ONNX Runtime source-build schema, ABI set, SDK levels, CPU-only flags,
strict reduced-operator configuration parser, immutable model-source manifest/fetch behavior,
session-separated swipe preparation, split/hash reproducibility, model architecture/parameter count,
CTC utilities, and ONNX operator-file ordering. These dependency-free tests run for every pull
request.

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
The suite checks the installed package rather than only source XML: merged permissions, backup and
cleartext flags, the IME service permission/direct-boot flag, the non-exported clipboard-search
activity, the Android `EditorInfo` policy matrix, and real credential-encrypted SQLite behavior for
personalization plus clipboard search, storage limits, pruning, and stale-id handling.
The device suite also verifies that an explicit wipe clears real credential-encrypted personal rows
and legacy rejection files while leaving installed-model and clipboard fixtures intact.

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
