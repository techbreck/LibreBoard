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
- Live batch input unions the retained AOSP matcher with pure-Kotlin geometric and optional CTC
  candidates over one revision-cached static/personal lexicon. Missing, late, incompatible, and
  circuit-open CTC inference preserves the classic/geometric slate.
- Immutable engine contracts for candidate fusion, language lock, deadlines, model validation,
  neural availability, and stale-result sequencing.
- Live tap and next-word fusion builds immutable, bounded requests from current geometry and coarse
  field class. A registry-loaded context model is hard-limited to 35 ms; disabled, absent, late,
  incompatible, and circuit-open results preserve the classic slate and raw-word guarantee.
- User-facing 0–100 context strength and cautious/balanced/aggressive calibrated commit controls;
  neither setting can bypass field, personal-word, rejection, digit, capitalization, or compound
  safety vetoes.
- Bounded, searchable credential-encrypted clipboard and personal stores with Direct-Boot-safe degradation.
- Central field policy for passwords, PINs, email/URI fields, terminals,
  `IME_FLAG_NO_PERSONALIZED_LEARNING`, and `TYPE_TEXT_FLAG_NO_SUGGESTIONS`.
- Enforced `InputConnection` context boundary: restricted fields and incognito never query
  surrounding editor text, even during cache refresh or cursor reconciliation.
- Local unigram, n-gram, phrase, and correction-rejection persistence with decay, preferred word
  casing, live personal completions/predictions, and atomic wipe.

The production context-model loader is not activated until an accepted signing key and model pass
the release gates. The context and CTC model artifacts are not represented as complete until their
held-out quality, latency, provenance, reproducibility, and GrapheneOS gates pass. See
[Phase 0](docs/phase-0.md).

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
