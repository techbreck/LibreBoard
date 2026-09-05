# Privacy and storage contract

LibreBoard requests no network permission and contains no analytics, crash-reporting, advertising,
or account SDK. Typed text and gesture traces are never logged or exported.

## Field policy

`FieldPolicyResolver` is the sole decision point for restricted editors.

| Policy | Context | Suggestions | Composing | Autocorrect | Clipboard capture | Persistence |
|---|---:|---:|---:|---:|---:|---:|
| Normal | yes | yes | yes | yes | yes | yes |
| No learning | no | no | yes | no | no | no |
| No suggestions | no | no | yes | no | no | no |
| Sensitive/password/PIN | no | no | no | no | no | no |
| Email/URI | no | no | yes | no | no | no |
| Terminal | no | keyboard-local only | no | no | no | no |

The setting override for `TYPE_TEXT_FLAG_NO_SUGGESTIONS`, when exposed, must affect only that flag.
It must never override password, PIN, email, URI, or no-personalized-learning policies.

## Storage

- Layouts and safe settings needed at boot explicitly use device-protected storage.
- Clipboard files/database, personalization, rejection history, models, and caches use the normal
  credential-protected application context.
- Private stores return unavailable before first unlock; the static keyboard still types.
- `android:allowBackup` is false. Explicit SAF export excludes clipboard unless the user selects it.
- Personal facts are normalized tokens, 1–4-token n-grams, counts, timestamps, language tags, and
  one-way context fingerprints. Surrounding paragraphs and application package names are forbidden.

Incognito clears process-local rejection/model context, suppresses suggestions and clipboard capture,
and prevents all personal writes. Existing learned data is not silently erased. The learned-data wipe
uses one database transaction and also clears process caches.

## Automated guard

`python3 tools/verify_release.py --source` rejects network permissions, dynamic native loading,
known telemetry/Play dependencies, a backup-enabled manifest, and reintroduction of gesture-data
collection source files. APK verification additionally checks merged permissions, native allowlists,
ZIP alignment, and 16 KiB ELF segment alignment.
