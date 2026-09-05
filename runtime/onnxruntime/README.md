# Source-built ONNX Runtime Mobile

LibreBoard pins ONNX Runtime 1.26.0 as the `third_party/onnxruntime` Git submodule at commit
`8c546c37b43caaca1fa25db430dab94b901cf277`. The committed settings build only the CPU execution
provider for API 26 and all four application ABIs. NNAPI, XNNPACK, WebGPU, Play Services, Runtime
Extensions, training APIs, and the Maven ONNX Runtime AAR are not used.

The build is intentionally fail-closed until the accepted ONNX models and their generated reduced
operator configurations exist. `tools/assemble_runtime_operator_config.py` verifies each exported
model, export report, signed manifest and per-model operator file before producing their deterministic
union. After those artifacts pass Phase 0, initialize all source dependencies and build the local AAR
with:

```sh
uv venv --python 3.11 build/onnxruntime-venv
uv pip install --python build/onnxruntime-venv/bin/python \
  --require-hashes -r models/training/requirements-onnxruntime-build-linux-x86_64.lock
git submodule update --init --recursive third_party/onnxruntime
python3 tools/assemble_runtime_operator_config.py
build/onnxruntime-venv/bin/python tools/build_onnxruntime_android.py \
  --ops-config build/model-export/onnxruntime/required_operators.config \
  --jobs 4
```

`--jobs` bounds native compilation parallelism without changing the audited build configuration or
artifact contract. Lower it on a machine concurrently running model training.

The builder verifies the source commit, clean source tree, initialized nested submodules, exact build
settings, exact NDK 28.0.13004108, reduced-operator configuration, native size ceiling, and 16 KiB ELF
alignment. It sets stable locale/time/path inputs, rewrites the AAR with canonical order, timestamps,
metadata and compression, and writes a schema-2 hash/toolchain manifest beside it under
`build/onnxruntime/output`. Generated binaries are never committed as source.

For a model-qualified integration build, pass the verified AAR, build manifest, and exact composed
operator configuration to Gradle:

```sh
./gradlew assembleRelease \
  -PlibreboardOnnxRuntimeAar=build/onnxruntime/output/onnxruntime-mobile-1.26.0.aar \
  -PlibreboardOnnxRuntimeManifest=build/onnxruntime/output/onnxruntime-mobile-1.26.0.build.json \
  -PlibreboardOnnxRuntimeOperators=build/model-export/onnxruntime/required_operators.config
```

Gradle rejects an incomplete artifact set, an altered AAR, a non-pinned source/NDK/toolchain, a
different builder or settings file, or an AAR reduced for a different operator configuration.
Ordinary development builds omit all three properties and exercise the always-available
classic/geometric path.

The application refers to the optional AAR through fixed internal class names rather than a Maven
compile dependency, allowing the same source tree to build the core-only fallback. The adapter never
accepts a class or library path from a model. At session creation it compares ORT's model metadata to
the committed `swipe-latin-v1` or `context-en-de-v1` tensor contract and rejects any extra, missing,
mistyped, or reshaped input/output before inference.

Two independent clean Linux invocations must produce byte-identical AARs before the runtime is
enabled in a release APK. The final APK verifier remains authoritative for packaged permissions,
native-library allowlists, 16 KiB ELF segments, and ZIP alignment.
