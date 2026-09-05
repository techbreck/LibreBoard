# Model and native-runtime policy

The core APK currently has no accepted neural artifact. This is intentional: model filenames are not
treated as implementation until training provenance, held-out quality, latency, memory, reproducible
export, and GrapheneOS execution all pass.

The accepted v1 design reserves engine ABI 1 for:

- `swipe-latin-v1.onnx`: layout-conditioned CTC model, Apache-2.0 project weights, at most 3 MiB.
- `context-en-de-v1.onnx`: bounded candidate rescorer, Apache-2.0 project weights, at most 24 MiB,
  delivered through a data-only model pack or signed SAF import.
- ONNX Runtime Mobile 1.26.0 source at commit
  `8c546c37b43caaca1fa25db430dab94b901cf277`, reduced to committed model operators.

No stock Maven AAR, ORT Extensions, NNAPI/Play delegate, GGUF runtime, downloaded executable, or
arbitrary model signature is permitted. `ModelManifestValidator` is fail-closed on schema, engine ABI,
app version, size, hashes, operator allowlist, locale list, license, and provenance.

Operator validation is independent of the signed declaration. Before native inference, LibreBoard
maps the bounded ONNX protobuf, enumerates nodes in the inference graph, graph-valued attributes,
repeated graph attributes, and model-local function bodies, and requires that discovered set to
exactly match the manifest. Non-standard domains use `domain::Operator`; training graphs, malformed
wire data, undeclared operators, and declared-but-absent operators are rejected.

Manual imports use a strict `.lbmodel` ZIP container with exactly `manifest.json`, `model.onnx`,
optional `tokenizer.json`, and `signature.der`. `ModelRegistry` streams the blob into CE staging,
enforces size and entry allowlists, verifies SHA-256 and the injected project ECDSA key, atomically
activates it, and retains one last-known-good version for corruption rollback. No ZIP entry is ever
loaded as code. A production signing public key and accepted model still need to pass the Phase 0 gate.

An accepted model directory must contain the ONNX blob, `model.json`, tokenizer where applicable,
LICENSE, model card, deterministic export command, dataset revision manifest, split hashes, evaluation
report, and reduced-operator configuration. Never commit placeholder bytes under a release filename.
