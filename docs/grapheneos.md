# GrapheneOS compatibility gate

GrapheneOS is a first-class reference platform for LibreBoard. A release is blocked unless the same
APK is fully usable on supported GrapheneOS hardware without installing sandboxed Google Play.

## Build-time requirements

- No `INTERNET` or `ACCESS_NETWORK_STATE` permission in any merged manifest or APK.
- No Play Services, ML Kit, Firebase, proprietary delegate, downloaded executable, imported `.so`,
  or system-Google gesture dependency.
- Only allowlisted FOSS native libraries; project-owned native code is source-built, dependencies are
  version-pinned, and all ELF load segments and APK entries meet 16 KiB alignment requirements.
- Models and model packs are data-only, bounded, hashed, signed, operator-allowlisted, and optional.
- The keyboard remains functional when JIT, native inference, a model provider, or CE storage is
  unavailable. It must not ask users to weaken exploit protection or compatibility settings.
- GrapheneOS's per-app [Exploit protection compatibility mode](https://grapheneos.org/usage#bugs-uncovered-by-security-features)
  remains disabled. This preserves hardened_malloc, the extended address space, memory tagging where
  applicable, and the normal native-debugging restriction described by GrapheneOS.

## Required device matrix

Run on a currently supported Pixel with the current stable GrapheneOS release and no Google Play:

Before filling the release record, capture the connected device's read-only prerequisites:

```sh
python3 tools/capture_android_device.py \
  --adb "$ANDROID_SDK_ROOT/platform-tools/adb" \
  --output build/reports/grapheneos-device-inventory.json
```

The collector rejects ambiguous/offline targets, records bounded build identity, detects qemu and
the three sandboxed Google Play packages for the active Android user, and confirms LibreBoard is
installed. Its output is intentionally marked `isReleaseEvidence: false`: the operator must still
confirm current stable GrapheneOS, keep exploit protection compatibility mode disabled, and execute
the full matrix below. The final release verifier accepts only the separately completed schema-2
record.

1. Install the reproducible release APK and verify it appears in the system keyboard picker.
2. Reboot and type before first unlock using static dictionaries/layouts. Confirm clipboard,
   personalization, models, and their files remain unavailable.
3. After unlock, exercise Messages-compatible plain text, Chromium URL/email/password fields,
   Vanadium WebView fields, and a terminal. Verify the field-policy table in `docs/privacy.md`.
4. Toggle incognito while a composition is active. Confirm the composition/context cache is cleared,
   the strip is suppressed, and private-store row counts do not change.
5. Corrupt/remove/slow each optional model. Tap input and geometric swipe must remain responsive,
   with no crash, stale publication, duplicate terminal commit, or empty normal-field strip.
6. Exercise clipboard pin/search/expiry, manual backup round trip, learned-data wipe, language lock,
   German compounds, correction rejection, one-handed/split layouts, rotation, and IME animation.
7. Capture p50/p95/p99 tap/swipe latency, cold/warm start, peak RSS, circuit-breaker activation, and
   crash-free results. Attach the device build fingerprint and GrapheneOS version to the release.

## Representative-latency measurement (debug vs testOnly)

GrapheneOS disables the JIT system-wide (`dalvik.vm.usejit=false`). Debuggable packages are
capped to verify-only AOT, so `debugNoMinify` instrumented replays run fully interpreted and
cannot be read as product latency — CTC swipe in particular misses its 1500 ms deadline.

For representative Phase 0 latency on this platform, install the non-debuggable `measure`
variant (`android:testOnly="true"`) and AOT-compile it before the run:

```sh
./gradlew :app:assembleMeasure :app:assembleMeasureAndroidTest --max-workers=1 \
  -PlibreboardTestBuildType=measure \
  -PlibreboardOnnxRuntimeAar=build/linux-runtime-debian-a/output/onnxruntime-mobile-1.26.0.aar \
  -PlibreboardOnnxRuntimeManifest=build/linux-runtime-debian-a/output/onnxruntime-mobile-1.26.0.build.json \
  -PlibreboardOnnxRuntimeOperators=build/model-export/onnxruntime-development-types/required_operators-development.config
adb install -t -r -d app/build/outputs/apk/measure/LibreBoard_0.1.0-alpha01-measure.apk
adb install -t -r -d app/build/outputs/apk/androidTest/measure/app-measure-androidTest.apk
python3 tools/run_phase0_measurement.py --serial "$SERIAL" --external-staging --compile-filter speed ...
```

The driver stages corpora under `/data/local/tmp` because `run-as` is unavailable on a
non-debuggable package. This is a measurement-harness path, not the release APK.

Automated unit/emulator tests do not satisfy this gate. The release evidence must contain a completed
physical-device record and the matching machine-readable JSON described in
`docs/release/grapheneos-evidence-schema.md`; an absent or verifier-rejected record means “not
released,” never “probably compatible.”

The checked-in record template is `docs/release/grapheneos-device-record.md`. Its blocked status is
intentional until the exact release APK passes on supported physical hardware.
