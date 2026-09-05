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
git submodule update --init --recursive third_party/onnxruntime
python3 tools/assemble_runtime_operator_config.py
python3 tools/build_onnxruntime_android.py \
  --ops-config build/model-export/onnxruntime/required_operators.config
```

The builder verifies the source commit, clean source tree, initialized nested submodules, exact build
settings, NDK r28, reduced-operator configuration, native size ceiling, and 16 KiB ELF alignment. It
sets stable locale/time/path inputs and writes a hash manifest beside the generated AAR under
`build/onnxruntime/output`. Generated binaries are never committed as source.

For a model-qualified integration build, pass both verified outputs to Gradle:

```sh
./gradlew assembleRelease \
  -PlibreboardOnnxRuntimeAar=build/onnxruntime/output/onnxruntime-mobile-1.26.0.aar \
  -PlibreboardOnnxRuntimeManifest=build/onnxruntime/output/onnxruntime-mobile-1.26.0.build.json
```

Gradle rejects a lone artifact, an altered AAR, a non-pinned source revision, or a build manifest
produced from settings other than the audited file in this directory. Ordinary development builds
omit both properties and exercise the always-available classic/geometric path.

The application refers to the optional AAR through fixed internal class names rather than a Maven
compile dependency, allowing the same source tree to build the core-only fallback. The adapter never
accepts a class or library path from a model. At session creation it compares ORT's model metadata to
the committed `swipe-latin-v1` or `context-en-de-v1` tensor contract and rejects any extra, missing,
mistyped, or reshaped input/output before inference.

Two independent clean Linux invocations must produce byte-identical AARs before the runtime is
enabled in a release APK. The final APK verifier remains authoritative for packaged permissions,
native-library allowlists, 16 KiB ELF segments, and ZIP alignment.
