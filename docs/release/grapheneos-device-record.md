# GrapheneOS physical-device release record

Status: **BLOCKED — no qualifying device run has been recorded**

Complete this file for the exact release APK. Do not replace missing evidence with emulator results.
The completed run must also be serialized using `grapheneos-evidence-schema.md` and pass the strict
`tools/verify_release.py` evidence command documented there.

## Artifact and device

- LibreBoard version/commit:
- APK filename:
- APK SHA-256:
- Reproducible Linux build report:
- Device model:
- GrapheneOS build number:
- Android security patch level:
- Build fingerprint:
- Sandboxed Google Play installed: no
- Exploit protection compatibility mode enabled: no
- Tester/date:

## Required results

- [ ] APK passes `tools/verify_release.py --source --apk ...`.
- [ ] Keyboard can be enabled, selected, and used with GrapheneOS's per-app Exploit protection
      compatibility mode disabled.
- [ ] After reboot and before first unlock, static tap typing works while personal/model/clipboard
      stores remain unavailable.
- [ ] Plain text tap correction, raw-word recovery, next-word suggestions, and correction revert pass.
- [ ] Chromium/Vanadium URL and email fields have no context read, suggestion, clipboard, or learning.
- [ ] Password and PIN fields have no suggestion, composing, clipboard, context, or persistence leak.
- [ ] Explicit incognito clears active composition/context caches and causes zero private-store writes.
- [ ] `NO_SUGGESTIONS` is hard-off; its dedicated override affects no other policy.
- [ ] Terminal tap characters are not duplicated and each swipe commits exactly once.
- [ ] English/German switches occur only at word boundaries; German compounds remain unsplit.
- [ ] Missing, corrupt, disabled, and deadline-exceeding models preserve classic tap and geometric swipe.
- [ ] Clipboard search/pin/expiry, backup/restore, and atomic learned-data wipe pass.
- [ ] Rotation, configuration changes, predictive back, edge-to-edge insets, and IME animation pass.
- [ ] p50/p95/p99 tap and swipe latency, cold/warm start, peak RSS, and timeout counts are attached.
- [ ] No crash, ANR, stale candidate, empty normal-field strip, or security-setting exception occurred.

## Measurements and observations

Attach raw instrumentation output and the Phase 0 JSON report here or link immutable release assets.
