# Android 16 16 KB smoke-test record

Date: 2026-09-05

Status: PASS for the checks below
GrapheneOS release status: **not certified by this record**

This record covers a disposable AOSP-derived Android emulator. It is evidence for Android 16,
16 KB native compatibility, field-policy behavior, and Direct Boot. The separate physical
GrapheneOS device matrix in `grapheneos-device-record.md` remains mandatory before release.

## Environment

- Host: Apple arm64
- Emulator: Android Emulator 36.4.10.0 (build 15004761)
- System image: `system-images;android-36;google_apis_ps16k;arm64-v8a`
- Guest: Android 16 / API 36
- `getconf PAGE_SIZE`: `16384`
- Artifact: `LibreBoard_0.1.0-alpha01-debug.apk`
- SHA-256: `4968ed9dc5040f31b03011b4200acce0e056c1a68e485bad67373a846c0d36da`

## Results

- The signed debug APK installed successfully under `org.libreboard.keyboard.debug`.
- Android discovered, enabled, selected, and bound
  `org.libreboard.keyboard.debug/helium314.keyboard.latin.LatinIME`.
- `libjni_latinime.so` loaded successfully on the 16 KB ARM64 guest.
- The LibreBoard settings activity started and remained healthy.
- In Chrome's URL field, literal input worked while the keyboard suggestion strip stayed empty,
  as required by `EMAIL_URI` field policy.
- In a normal text field, typing `thsi` displayed the literal `thsi` in the left suggestion slot
  and `this` in the center slot.
- `READ_CONTACTS` remained denied; the keyboard still initialized and typed normally.
- After setting a device credential and rebooting, Android reported user 0 as `RUNNING_LOCKED`.
  LibreBoard remained the selected IME, was bound as the current IME on the credential screen,
  and `mInputShown=true` before credential-encrypted storage was unlocked.
- Entering the test credential transitioned user 0 to `RUNNING_UNLOCKED`.
- No fatal Java exception, native-link error, SIGSEGV, or fatal native signal was observed during
  installation, normal typing, restricted-field typing, IME rebinding, reboot, or unlock.

## Release limitation

This does not substitute for testing a release-signed APK on supported GrapheneOS hardware.
Vanadium/WebView, Termux, password managers, GrapheneOS permission controls, reboot-before-unlock,
and upgrade behavior must still be recorded in the physical-device release matrix.
