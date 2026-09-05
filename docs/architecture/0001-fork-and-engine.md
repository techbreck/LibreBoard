# ADR 0001: Fork HeliBoard and isolate the LibreBoard engine

Status: accepted

## Decision

LibreBoard forks HeliBoard v4.1 at `9f5bb635c2e8609dcd95dc7506c0c58fba82a52c`, retains its Git
history, and keeps internal package names during the first release cycle. The application ID is
`org.libreboard.keyboard`; application changes are GPL-3.0-only.

The retained IME shell owns editor lifecycle, composing, layouts, themes, toolbar, dictionaries,
and pointer collection. New behaviour enters through Android-independent contracts under
`helium314.keyboard.latin.engine`:

- `TypingRequest` is an immutable, bounded snapshot with a sequence ID.
- `FieldPolicy` makes data access and commit behaviour explicit.
- `CandidatePipeline` returns bounded, language-tagged `SuggestionBatch` values.
- `NeuralRescorer` and `SwipeDecoder` return explicit unavailable/timeout states.
- `PersonalStore` observes bounded token facts and correction rejections, never prose or app IDs.
- `ModelManifestValidator` rejects incompatible, oversized, unlicensed, or unsupported models.

All published results must still match the active sequence. The raw tap input occupies the first
suggestion slot. Autocorrection is a separately gated decision and cannot be forced by a model.

## Gesture decision

Every HeliBoard route for importing a `.so`, probing `jni_latinimegoogle`, or loading a user-selected
native library is removed. The retained native library is the source-built AOSP dictionary engine.
LibreBoard swipe input always has a pure-Kotlin, live-geometry fallback; an ONNX CTC decoder may add
candidates but cannot replace that fallback.

## Storage decision

The application context remains credential protected. Static settings that must exist before first
unlock explicitly use HeliBoard's device-protected helper. Clipboard, personalization, rejection,
model registry, and caches remain credential protected and fail closed while locked. Android cloud
backup is disabled; explicit SAF backup is the only supported transfer mechanism.

## Consequences

Upstream security/platform fixes can be reviewed from the `upstream` remote without rebasing the
fork point. The engine can be unit-tested without an `InputMethodService`. Model absence and native
inference failure remain normal runtime states. A release cannot claim neural quality merely because
the integration interfaces exist: Phase 0 measurements remain a blocking gate.
