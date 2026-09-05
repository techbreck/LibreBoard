# Model and native-runtime policy

The core APK currently has no accepted neural artifact. This is intentional: model filenames are not
treated as implementation until training provenance, held-out quality, latency, memory, reproducible
export, and GrapheneOS execution all pass.

The swipe build path is now implemented without committing or pretending to accept untrained bytes:

- `models/sources/v1.json` pins the MIT FUTO gesture corpus and Apache-2.0 Hanse2 teacher to immutable
  repository commits, exact sizes, licenses, and SHA-256 values.
- `tools/model_sources.py` verifies offline by default. Its explicit `fetch` operation stages each
  transfer, enforces the declared byte ceiling while streaming, checks SHA-256, and atomically
  publishes only an exact match. Files over 64 MiB require `--accept-large-downloads`.
- `tools/prepare_swipe_dataset.py` reassigns whole collection sessions with a committed salted hash,
  hashes session IDs, strips timestamps/context, resamples paths to the Android 64-point convention,
  uses top-left-to-bottom-right key-slot order, deduplicates records, and emits a hash-bound split
  manifest with all rejection and difficulty-stratum counts.
- `models/swipe/corpus-manifest.json` pins the exact reviewed million-gesture corpus. A
  release-eligible training run refuses any prepared split whose manifest is not byte-identical.
- `models/swipe/model-spec.json` fixes the 821,121-parameter layout-conditioned architecture and
  tensor ABI. `tools/train_swipe_model.py` uses deterministic CTC training and writes hash-bound
  safetensors, with atomic epoch checkpoints that restore model, optimizer, and RNG state only when
  every execution binding still matches. `tools/export_swipe_model.py` stores learned initializers as FP16, retains float32
  external tensors/CPU compute, runs the full ONNX checker, rejects external data/custom domains,
  and derives the reduced-operator file from the exact graph.
- `models/training/requirements-linux-x86_64.lock` pins and hashes the build-only CPU toolchain. These
  packages are not Android dependencies and never enter an APK.

The context-model architecture/export path is also executable without treating synthetic bytes as a
release artifact:

- `models/context/model-spec.json` fixes the 35,662,848-parameter en/de candidate-only Transformer,
  32-row dynamic tensor ABI, 24-token shared prefix, eight-token candidate suffix, and 24 MiB limit.
- `tools/context_tokenizer_contract.py` mirrors the Android tokenizer's strict schema and adds the
  release requirement for exactly 16,384 dense token IDs.
- `tools/export_context_model.py` binds the pinned Hanse2 teacher, corpus manifest, tokenizer,
  safetensors and app commit; converts the source graph to blockwise INT4; verifies the exact custom
  operator counts; and runs CPU inference at batch sizes 1 and 32.
- `models/training/requirements-context-linux-x86_64.lock` hashes the complete build-only distillation
  and export environment, including Transformers and ONNX Runtime 1.26 tooling.

Synthetic end-to-end smoke tests prove corpus preparation, one-step training, FP16 ONNX export,
operator enumeration, the 2.5 MiB model ceiling, and manifest/hash binding. A smoke artifact retains
`development` in every relevant filename and is release-ineligible by construction.

The accepted v1 design reserves engine ABI 1 for:

- `swipe-latin-v1.onnx`: layout-conditioned CTC model, Apache-2.0 project weights, at most 3 MiB.
- `context-en-de-v1.onnx`: bounded candidate rescorer, Apache-2.0 project weights, at most 24 MiB,
  delivered through a data-only model pack or signed SAF import.
- ONNX Runtime Mobile 1.26.0 source at commit
  `8c546c37b43caaca1fa25db430dab94b901cf277`, reduced to committed model operators.

`swipe-latin-v1.onnx` has a fixed, test-covered tensor ABI. It receives `path_coordinates`
`float32[1,64,2]`, `key_centers` `float32[1,64,2]`, and `key_mask` `float32[1,64]`. Coordinates are
arc-length-resampled and normalized against the live letter-key bounds; unused or disabled key slots
are masked. It returns finite `logits` `float32[1,32,65]`, where class zero is CTC blank and classes
1–64 correspond to the supplied key-slot order. Output validation, CTC blank/repeat handling,
lexicon-constrained prefix beam search, language filtering, and candidate construction live in pure
Kotlin rather than inside the model or runtime bridge. Apostrophes and hyphens may be absent from a
gesture emission while the lexicon retains the canonical surface form.

`context-en-de-v1.onnx` also has a fixed candidate-scoring ABI. It receives `input_ids`
`int64[N,32]`, `attention_mask` `int64[N,32]`, `candidate_mask` `float32[N,32]`, and `field_class`
`int64[N]`, then returns one finite `candidate_log_likelihood` per row. `N` is capped at 32 and each
candidate at eight wordpieces; the BOS/language prefix and bounded preceding context occupy the
remaining positions. Context scoring is keyed by normalized surface **and language**, so identical
English and German spellings cannot exchange scores. Restricted field policy is checked before
tokenization or native inference. A candidate requiring more than eight wordpieces is omitted from
neural scoring instead of being ranked from a misleading prefix; the rest of the candidate slate is
still scored normally.

The context `tokenizer.json` is schema 1 `NFKC_LOWER` BPE data: a dense release vocabulary of exactly 16,384
IDs, ranked two-symbol merge pairs, distinct padding/BOS/unknown tokens, and explicit language-token
mappings. LibreBoard parses it with unknown fields disabled, caps it at 2 MiB, validates merge outputs
and special IDs, applies Unicode NFKC deterministically, and makes start-versus-end truncation
explicit. No tokenizer code is loaded from a model pack.

No stock Maven AAR, ORT Extensions, NNAPI/Play delegate, GGUF runtime, downloaded executable, or
arbitrary model signature is permitted. `ModelManifestValidator` is fail-closed on schema, engine ABI,
model kind, exact tensor ABI, app version, size, hashes, required tokenizer, operator allowlist,
locale list, license, and provenance. A signed context model therefore cannot be loaded through the
swipe tensor interface, or vice versa.

Operator validation is independent of the signed declaration. Before native inference, LibreBoard
maps the bounded ONNX protobuf, enumerates nodes in the inference graph, graph-valued attributes,
repeated graph attributes, and model-local function bodies, and requires that discovered set to
exactly match the manifest. Non-standard domains use `domain::Operator`; training graphs, malformed
wire data, undeclared operators, declared-but-absent operators, and every form of ONNX external
tensor data are rejected. The runtime therefore cannot follow a model-supplied filesystem path.

Manual imports use a strict `.lbmodel` ZIP container with exactly `manifest.json`, `model.onnx`,
optional `tokenizer.json`, and `signature.der`. `ModelRegistry` streams the blob into CE staging,
enforces size and entry allowlists, verifies SHA-256 and the injected project ECDSA key, atomically
activates it, and retains one last-known-good version for corruption rollback. Swipe and context
models own separate CE activation/rollback slots, so installing or wiping one cannot replace the
other. No ZIP entry is ever loaded as code. A production signing public key and accepted model still
need to pass the Phase 0 gate.

The official English/German context archive is distributed through the opt-in `modelpack-en-de`
Android module. The resulting `org.libreboard.model.en_de` APK has no permissions, activity,
service, receiver, native library, or networking dependency. Its sole exported component is a
read-only provider for `content://org.libreboard.model.en_de/model.lbmodel`; the core app copies that
stream through the same untrusted `.lbmodel` verifier used for manual imports. It is unavailable
before first unlock and therefore cannot weaken Direct Boot behavior. Build it only with a real,
signed archive:

```sh
./gradlew \
  -PlibreboardIncludeModelPack=true \
  -PlibreboardContextModelArchive=/absolute/path/context-en-de-v1.lbmodel \
  :modelpack-en-de:assembleRelease

python3 tools/verify_release.py \
  --model-pack-apk modelpack-en-de/build/outputs/apk/release/modelpack-en-de-release-unsigned.apk
```

The verifier checks the merged zero-permission/component manifest, fixed authority, uncompressed
single asset, absence of native code, bounded archive entries, hashes, en/de model contract, license,
and provenance. The project signature is still rechecked by `ModelRegistry` on the device.

When the verified source-built runtime is packaged, LibreBoard opens the model by local filesystem
path and validates the runtime-reported input/output names, element types, and static or required
dynamic dimensions against the contracts above before the first inference. Inputs use bounded direct
native-order buffers; outputs are shape-checked and copied into bounded Kotlin arrays before every
native result/tensor wrapper is closed. Session options and the single session are retained and closed
together. Missing classes/native libraries, malformed metadata, inference failure, and deadline expiry
remain distinct degradation states. A non-cooperative context call runs behind a deadline-bound worker,
so it cannot delay classic suggestion publication past the request budget.

An accepted model directory must contain the ONNX blob, `model.json`, tokenizer where applicable,
LICENSE, model card, deterministic export command, dataset revision manifest, split hashes, evaluation
report, and reduced-operator configuration. Never commit placeholder bytes under a release filename.

The current swipe model card and exact reproduction commands are in
`models/swipe/MODEL_CARD.md`; the context architecture and export contract are in
`models/context/MODEL_CARD.md`. The pinned gesture inputs total roughly 5.7 GB, so fetching them remains
an explicit operator action rather than a side effect of Gradle or tests. The context source and
exporter are pinned, but its bilingual corpus preparation/distillation training and both accepted
trained candidates still remain outstanding.
