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
The retained suggestion path now fuses bounded static/spatial, personal, lexical, language, and
optional context scores. It snapshots current geometry, coarse field class, enabled languages,
bounded context, and sequence ID before dispatch. Context inference runs through one process-scoped
session behind a hard deadline and three-overrun circuit breaker. Only an available result receives
nonzero neural weight; every other result preserves the classic ranking. Production activation and
release calibration remain gated on accepted signed artifacts and Phase 0 measurements.

## Gesture decision

Every HeliBoard route for importing a `.so`, probing `jni_latinimegoogle`, or loading a user-selected
native library is removed. The retained native library is the source-built AOSP dictionary engine.
LibreBoard swipe input always has a pure-Kotlin, live-geometry fallback. It runs beside the optional
ONNX CTC decoder against one language-tagged static/personal lexicon, then sends their provenance-
preserving union through the same language lock and context scorer used for taps. CTC execution has
a hard deadline and three-overrun circuit breaker; absent, incompatible, late, or disabled models
cannot replace or suppress the retained AOSP/geometric fallback.

## Storage decision

The application context remains credential protected. Static settings that must exist before first
unlock explicitly use HeliBoard's device-protected helper. Clipboard, personalization, rejection,
model registry, and caches remain credential protected and fail closed while locked. Android cloud
backup is disabled; explicit SAF backup is the only supported transfer mechanism.

Field policy is also enforced at the `InputConnection` boundary. Disabling context access clears
the IME's text mirrors and prevents all surrounding/selected/extracted-text IPC, so a later caller
cannot accidentally bypass policy by invoking a retained HeliBoard helper.

## Consequences

Upstream security/platform fixes can be reviewed from the `upstream` remote without rebasing the
fork point. The engine can be unit-tested without an `InputMethodService`. Model absence and native
inference failure remain normal runtime states. A release cannot claim neural quality merely because
the integration interfaces exist: Phase 0 measurements remain a blocking gate.
