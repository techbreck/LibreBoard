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

## Required device matrix

Run on a currently supported Pixel with the current stable GrapheneOS release and no Google Play:

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

Automated unit/emulator tests do not satisfy this gate. The release evidence must contain a completed
physical-device record and the matching machine-readable JSON described in
`docs/release/grapheneos-evidence-schema.md`; an absent or verifier-rejected record means “not
released,” never “probably compatible.”

The checked-in record template is `docs/release/grapheneos-device-record.md`. Its blocked status is
intentional until the exact release APK passes on supported physical hardware.
