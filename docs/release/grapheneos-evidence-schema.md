# GrapheneOS release-evidence schema

The physical-device procedure in `grapheneos-device-record.md` produces a JSON object with
`schemaVersion: 1` and `status: "PASS"`. The release verifier rejects a partial or stale record.
Copy `grapheneos-evidence.template.json` for the run; the checked-in template is intentionally
`BLOCKED` with empty identifiers and false checks, so it cannot be mistaken for release evidence.

Identity fields bind the run to the exact release: `appCommit`, `apkFilename`, `apkSha256`,
`swipeModelSha256`, `contextModelSha256`, `phase0ReportSha256`, and
`instrumentationOutputSha256`. Device fields are `deviceModel`, `grapheneOsBuildNumber`, `apiLevel`,
`securityPatchLevel`, `buildFingerprint`, `physicalDevice: true`,
`sandboxedGooglePlayInstalled: false`, and `compatibilityChangesEnabled: false`. The record also
contains `testerId`, UTC `testedAtUtc`, and the `phase0TestRunId` from the matching GrapheneOS
environment in the Phase 0 report.

`checks` contains exactly these boolean keys, all `true`:

- `apk_verified`
- `keyboard_enable_select`
- `direct_boot`
- `plain_text_correction`
- `url_email_policy`
- `password_pin_policy`
- `incognito_policy`
- `no_suggestions_policy`
- `terminal_commit`
- `english_german_lock`
- `model_fallbacks`
- `clipboard_backup_wipe`
- `window_and_animation`
- `performance_budgets`
- `crash_anr_stale_result_free`

`measurements` contains `tapLatencyMs` and `swipeLatencyMs`, each with ordered, non-negative `p50`,
`p95`, and `p99`, plus non-negative `coldStartMs`, `warmStartMs`, `peakRssMiB`,
`neuralTimeoutCount`, and `circuitBreakerActivationCount`. Tap p95 must be at most 80 ms and swipe
p95 at most 200 ms.

Run the strict evidence gate with two independently clean-built copies of the same APK:

```sh
python3 tools/verify_release.py --source \
  --apk build/first/LibreBoard_release.apk \
  --rebuilt-apk build/second/LibreBoard_release.apk \
  --phase0-report build/reports/phase-0.json \
  --grapheneos-evidence build/reports/grapheneos.json \
  --instrumentation-output build/reports/grapheneos-instrumentation.txt
```

This gate supplements APK inspection; it does not manufacture device evidence. Until a qualifying
physical run exists, the correct result is a blocked release.
