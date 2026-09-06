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

`RichInputConnection` enforces the context column at the editor boundary. When reads are disabled it
clears previously cached text, refuses `getTextBeforeCursor`, `getTextAfterCursor`, selected-text,
and extracted-text IPC calls, and exposes only characters LibreBoard itself subsequently composes or
commits. Cursor updates and editing continue without importing pre-existing field contents. Unit
tests count every underlying surrounding-text call and require zero reads for each restricted policy.

## Storage

- Layouts and safe settings needed at boot explicitly use device-protected storage.
- Clipboard files/database, personalization, rejection history, models, and caches use the normal
  credential-protected application context.
- Clipboard text entries are capped at 100,000 UTF-16 characters, history at 100 unpinned and 200
  total entries, and attachment metadata is strictly bounded. Oldest unpinned entries are pruned
  first; an all-pinned full store rejects new captures. Attachment filenames must be safe leaf names.
- Clipboard search uses a bounded in-memory index. Its internal editor is non-exported and receives
  the `NO_LEARNING` field policy, so query text is neither persisted nor exposed to suggestion,
  context, capture, or personalization paths.
- Private stores return unavailable before first unlock; the static keyboard still types.
- Manual `.lbmodel` imports use the system document picker, copy through a bounded temporary file in
  credential-encrypted cache, and accept only project-signed data matching the fixed model and ONNX
  operator contracts. Runtime-incompatible imports restore the preceding active model. URI contents,
  model files, and inference results are never sent to another app or network service.
- `android:allowBackup` is false. Explicit SAF export currently excludes clipboard by design.
- Backup archives have strict entry-count, per-entry, and total expanded-size limits. Restore rejects
  traversal, absolute paths, duplicate targets, malformed typed settings, and personal data without
  a LibreBoard schema manifest. A crash journal rolls allowlisted file replacement back before
  commit or finishes deleting obsolete rollback copies after commit. Clipboard attachments, model
  packs, and other files outside the documented archive allowlist survive a restore.
- Personal facts are normalized tokens, preferred token casing, 1–4-token n-grams, counts,
  timestamps, language tags, and one-way context fingerprints. Surrounding paragraphs and
  application package names are forbidden. Personal candidates join the live tap/prediction slate
  under a 15 ms deadline, but do not replace the retained dictionary's calibrated winner; an exact
  personal match vetoes autocorrection.

Incognito clears process-local rejection/model context, suppresses suggestions and clipboard capture,
and prevents all personal writes. Existing learned data is not silently erased. The explicit,
searchable learned-data action clears personal tables in one database transaction, removes legacy
history, rejection files, dedicated learning caches/adapters, and learned files inside interrupted
restore copies, then notifies the live IME through a non-exported receiver so in-memory dictionaries
and next-word caches are cleared. Manually managed personal dictionaries, clipboard history, and
installed static models are outside that wipe boundary.

## Automated guard

`python3 tools/verify_release.py --source` rejects network permissions, dynamic native loading,
known telemetry/Play dependencies, a backup-enabled manifest, reintroduction of gesture-data
collection source files, and Java/Kotlin log calls containing typed text, candidates, key events,
or gesture state. LatinIME's native debug and profiling modes are fail-closed at build time because
their inherited diagnostics can contain candidate text and reconstructable gesture traces. APK
verification additionally checks merged permissions, native allowlists, ZIP alignment, and 16 KiB
ELF segment alignment.
