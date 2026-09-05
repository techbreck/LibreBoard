# LibreBoard

LibreBoard is a fully free, offline-first Android keyboard aimed at high-quality tap correction,
next-word prediction, multilingual typing, and swipe input without an account, telemetry, Google
Play Services, or network access.

The application is an in-progress fork of [HeliBoard v4.1](https://github.com/HeliBorg/HeliBoard/tree/v4.1)
at commit `9f5bb635c2e8609dcd95dc7506c0c58fba82a52c`. Internal HeliBoard/AOSP package names are preserved
temporarily to keep upstream review practical; the Android application ID is `org.libreboard.keyboard`.

## Current implementation

- Android 8/API 26 minimum; compile and target API 36.
- No `INTERNET` or `ACCESS_NETWORK_STATE` permission.
- No imported, proprietary, or system Google gesture library.
- Pure-Kotlin geometric swipe fallback using live keyboard geometry.
- Immutable engine contracts for candidate fusion, language lock, deadlines, model validation,
  neural availability, and stale-result sequencing.
- Credential-encrypted clipboard and personal stores with Direct-Boot-safe degradation.
- Central field policy for passwords, PINs, email/URI fields, terminals,
  `IME_FLAG_NO_PERSONALIZED_LEARNING`, and `TYPE_TEXT_FLAG_NO_SUGGESTIONS`.
- Enforced `InputConnection` context boundary: restricted fields and incognito never query
  surrounding editor text, even during cache refresh or cursor reconciliation.
- Local unigram, n-gram, phrase, and correction-rejection persistence with decay, preferred word
  casing, live personal completions/predictions, and atomic wipe.

The context and CTC model artifacts are not represented as complete until their held-out quality,
latency, provenance, reproducibility, and GrapheneOS gates pass. See [Phase 0](docs/phase-0.md).

## Build

Install Android SDK 36, NDK `28.0.13004108`, JDK 17, and set `sdk.dir` in `local.properties`.

```sh
./gradlew testRunTestsUnitTest
./gradlew assembleDebugNoMinifyAndroidTest --max-workers=1
./gradlew assembleDebug
python3 tools/verify_release.py --source --apk app/build/outputs/apk/debug/LibreBoard_0.1.0-alpha01-debug.apk
```

## Architecture and release gates

- [Architecture decision](docs/architecture/0001-fork-and-engine.md)
- [Privacy and storage contract](docs/privacy.md)
- [GrapheneOS compatibility gate](docs/grapheneos.md)
- [GrapheneOS physical-device record](docs/release/grapheneos-device-record.md)
- [GrapheneOS machine-verifiable evidence](docs/release/grapheneos-evidence-schema.md)
- [Testing and release evidence](docs/testing.md)
- [Model and runtime policy](docs/models/README.md)
- [Phase 0 evaluation gate](docs/phase-0.md)
- [Phase 0 measurement schema](docs/phase-0-dataset-schema.md)

## License and ancestry

LibreBoard application changes are licensed under GPL-3.0-only. Files inherited from HeliBoard,
OpenBoard, and AOSP retain their original copyright and SPDX notices. Permissively licensed native,
dictionary, and future model assets retain their own notices.

The retained Git history is the authoritative attribution record. Principal upstreams:

- [HeliBoard](https://github.com/HeliBorg/HeliBoard)
- [OpenBoard](https://github.com/openboard-team/openboard)
- [AOSP LatinIME](https://android.googlesource.com/platform/packages/inputmethods/LatinIME/)

See [LICENSE](LICENSE), [LICENSE-Apache-2.0](LICENSE-Apache-2.0), and
[LICENSE-CC-BY-SA-4.0](LICENSE-CC-BY-SA-4.0).
