# ABI packaging

LibreBoard has one source release sequence and two deterministic APK layouts:

| Artifact | Gradle property | APK version code for base `1` | Intended channel |
|---|---|---:|---|
| `armeabi-v7a` | `libreboardTargetAbi=armeabi-v7a` | 101 | F-Droid |
| `arm64-v8a` | `libreboardTargetAbi=arm64-v8a` | 102 | F-Droid |
| `x86` | `libreboardTargetAbi=x86` | 103 | F-Droid |
| `x86_64` | `libreboardTargetAbi=x86_64` | 104 | F-Droid |
| Universal | property omitted | 199 | GitHub mirror and cross-ABI recovery |

The source `defaultConfig.versionCode` remains the unencoded base value so F-Droid can discover it.
Each packaged output uses `100 * base + ABI offset`; the universal artifact uses offset 99. This is
the same version-code shape supported by F-Droid's documented `VercodeOperation` mechanism. It makes
all ABI artifacts unique, lets a same-release ABI install upgrade to the universal artifact, and
lets every next-release ABI artifact upgrade the prior universal artifact.

An unsupported or empty `libreboardTargetAbi` value fails during Gradle configuration. The property
narrows both NDK compilation and APK packaging to one ABI; it does not create a misleading split
that still contains native libraries for other architectures.

Build the ordinary universal release with:

```sh
./gradlew :app:assembleRelease --max-workers=1
```

Build each F-Droid artifact in a separate clean build invocation:

```sh
./gradlew :app:assembleRelease -PlibreboardTargetAbi=armeabi-v7a --max-workers=1
./gradlew :app:assembleRelease -PlibreboardTargetAbi=arm64-v8a --max-workers=1
./gradlew :app:assembleRelease -PlibreboardTargetAbi=x86 --max-workers=1
./gradlew :app:assembleRelease -PlibreboardTargetAbi=x86_64 --max-workers=1
```

Model-qualified release builds must also supply the pinned runtime manifest/operator configuration,
runtime AAR, signed swipe archive, and public key described in
[`runtime/onnxruntime/README.md`](../../runtime/onnxruntime/README.md). Omitting those inputs produces
the deliberate core-only fallback build, not a model-qualified release candidate.

Verify a universal APK without an ABI argument. Verify a one-ABI APK with the matching explicit
argument:

```sh
python3 tools/verify_release.py --apk app/build/outputs/apk/release/LibreBoard_0.1.0-alpha01-release.apk
python3 tools/verify_release.py --apk app/build/outputs/apk/release/LibreBoard_0.1.0-alpha01-release-arm64-v8a.apk --expected-abi arm64-v8a
```

The verifier checks the encoded version code, exact native ABI set, complete ONNX Runtime pair when
present, native-library allowlist, 16 KiB ELF/ZIP alignment, permissions, components, and signed-model
coupling. Passing `--expected-abi` cannot make a universal or wrong-ABI APK pass.

An F-Droid metadata submission should use four build entries with matching `gradleprops` and the
documented operations `100 * %c + 1` through `100 * %c + 4`. The final fdroiddata recipe remains a
release artifact: it must bind the accepted commit, source-built ONNX Runtime recipe, accepted signed
swipe model, and the exact output for each entry rather than referring to local development paths.

## Local core reproducibility check, 2026-09-08

Two fresh source-archive directories built the universal core-only APK from commit `b8283126`
with independent compilation outputs, Gradle build/configuration caches disabled, JDK 17.0.19,
NDK 28.0.13004108 and strict dependency verification. Both 21,623,288-byte unsigned APKs have SHA-256
`2ab4104531b9e503006c32cbe5f348e6b8343f5549a4687d937447faeec4167b`. The local evidence file is
`build/apk-reproducibility/core-reproducibility-b8283126.json`. The source archives do not carry Git
metadata; the report binds the source commit separately.

The first release attempt exposed 26 missing release-lint dependency hashes. Every added JAR/POM
was independently downloaded from Google Maven or Maven Central and matched before the verification
metadata was committed. CI now builds and verifies the core release APK as well as the debug tests.

This comparison uses one macOS host/toolchain. It does not replace an independent Linux rebuild,
signing-key acceptance, a model-qualified APK comparison, or the Phase 0/device release gates.
