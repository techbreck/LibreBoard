# Source-built ONNX Runtime Mobile

LibreBoard pins ONNX Runtime 1.26.0 as the `third_party/onnxruntime` Git submodule at commit
`8c546c37b43caaca1fa25db430dab94b901cf277`. The committed settings build only the CPU execution
provider for API 26 and all four application ABIs. NNAPI, XNNPACK, WebGPU, Play Services, Runtime
Extensions, generation operators, RTTI, unused float4/float8/optional/sparse/string tensor support,
training APIs, upstream unit-test/example-plugin targets, and the Maven ONNX Runtime AAR are not
used. Release compilation uses Clang's size optimization and disables external tensor initializers,
which the signed self-contained model format already rejects. Full ONNX model loading remains
enabled; LibreBoard does not substitute ORT-format-only minimal builds for its signed `.onnx` model
contract.

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
build/onnxruntime-venv/bin/python tools/assemble_runtime_operator_config.py --type-reduction
build/onnxruntime-venv/bin/python tools/build_onnxruntime_android.py \
  --ops-config build/model-export/onnxruntime/required_operators.config \
  --jobs 4
```

`--jobs` bounds native compilation parallelism without changing the audited build configuration or
artifact contract. Lower it on a machine concurrently running model training.

The assembler converts copies of both exact ONNX graphs at disabled and full optimization levels to
temporary ORT graphs solely to discover the complete operator/type kernel inventory. It unions that
inventory with the raw ONNX operators, writes a canonical type-specialized configuration, and deletes
the temporary graphs. The signed and shipped model artifacts remain the original `.onnx` files.

The wrapper resumes an ABI build or AAR packaging through a bounded retry only when a process launch
reports `Resource temporarily unavailable` together with `posix_spawn`, `fork`, or errno 35.
Compiler, linker, configuration, verification, and all other build failures remain terminal.

The builder verifies the source commit, clean source tree, initialized nested submodules, exact build
settings, exact NDK 28.0.13004108, reduced-operator configuration, native size ceiling, and 16 KiB ELF
alignment. After that validation, each upstream ABI build skips the redundant submodule sync so it
cannot fetch or rewrite source while compiling. The wrapper sets stable locale/time/path inputs,
rewrites the AAR with canonical order, timestamps, metadata and compression, and writes a schema-2
hash/toolchain manifest beside it under `build/onnxruntime/output`. Generated binaries are never
committed as source.

For a model-qualified integration build, pass the verified AAR, build manifest, and exact composed
operator configuration to Gradle:

```sh
./gradlew assembleRelease \
  -PlibreboardOnnxRuntimeAar=build/onnxruntime/output/onnxruntime-mobile-1.26.0.aar \
  -PlibreboardOnnxRuntimeManifest=build/onnxruntime/output/onnxruntime-mobile-1.26.0.build.json \
  -PlibreboardOnnxRuntimeOperators=build/model-export/onnxruntime/required_operators.config \
  -PlibreboardSwipeModelArchive=/absolute/path/swipe-latin-v1.lbmodel \
  -PlibreboardModelPublicKey=/absolute/path/libreboard-model-signing-public.der
```

Gradle rejects an incomplete artifact set, an altered AAR, a non-pinned source/NDK/toolchain, a
different builder or settings file, an AAR reduced for a different operator configuration, a
partial model/key pair, or a signed model without the verified runtime. The generated model assets
are deleted before every packaging pass so a core-only build cannot inherit files from an earlier
model-qualified build. Ordinary development builds omit all five properties and exercise the
always-available classic/geometric path.

The application refers to the optional AAR through fixed internal class names rather than a Maven
compile dependency, allowing the same source tree to build the core-only fallback. The adapter never
accepts a class or library path from a model. At session creation it compares ORT's model metadata to
the committed `swipe-latin-v1` or `context-en-de-v1` tensor contract and rejects any extra, missing,
mistyped, or reshaped input/output before inference.

Two independent clean Linux invocations must produce byte-identical AARs before the runtime is
enabled in a release APK. The final APK verifier remains authoritative for packaged permissions,
native-library allowlists, 16 KiB ELF segments, and ZIP alignment.

## Linux build environment

Build the core toolchain image using `docs/release/linux-core-build.md`, then prepare the runtime
image with the committed Python dependency lock:

```sh
mkdir -p build/linux-runtime-context
cp models/training/requirements-onnxruntime-build-linux-x86_64.lock build/linux-runtime-context/requirements.lock
docker build --platform linux/amd64 -f runtime/onnxruntime/Dockerfile.linux \
  -t libreboard-runtime-build:local build/linux-runtime-context
docker image inspect libreboard-core-release:local libreboard-runtime-build:local --format '{{.Id}}'
```

The image installs Debian bookworm-backports CMake 3.31.6 and Debian `ninja-build` as the PATH
host tools, plus native compiler utilities and the API 34 / Build Tools 30.0.3 inputs required by
the pinned upstream Android library project. Android SDK `cmake;3.31.6` remains in the SDK tree
so AGP can locate that package; `cmake` and `ninja` on PATH must resolve to `/usr/bin`. It reuses
the core image's previously accepted Android SDK licenses. Record both resolved image IDs: a local
tag alone does not pin the environment, and package repository contents can change between image
builds.

The SDK CMake 3.31.6 linux-x86_64 binary is a 21 MiB non-PIE `ET_EXEC` with control-flow and
stack-clash hardening. On this Apple Silicon host it runs through Colima qemu-user 7.0.0. That
binary intermittently SIGSEGVs during FetchContent `cmake -E` stamp steps such as
`cmake -E echo_append && cmake -E touch`. Isolated download/extract probes and `OOMKilled=false`
do not contradict this: the archives were intact, and Docker did not report memory killing.
Diagnosis is recorded in [`docs/models/evidence/linux-runtime-cmake-qemu-diagnosis.json`](../../docs/models/evidence/linux-runtime-cmake-qemu-diagnosis.json).
Do not prepend the SDK `cmake/3.31.6/bin` directory to `PATH`.

The first prepared runtime image resolved to
`sha256:f6e9c2f240620458f548daca3dfc759974c5fad51edd6ba2ce50f8d7ba6518ce`, based on core image
`sha256:425b1a57386654a8124fab1d5c02c182a4e9aa26fe0f4d15ed1c84f461c0187e`.
Its offline `tools/build_onnxruntime_android.py --check-only` invocation passed with the
source mounted read-only and the development operator inventory. This verifies build inputs;
it does not establish a successful native build or byte-identical AARs. Those remain pending.
Use separate empty build roots mounted at the same container path for the two native builds,
and serialize heavy builds on memory-constrained hosts.

Run native compilation as an unprivileged container user: the pinned upstream build script rejects
root execution. Match the output directory owner's UID/GID, set a writable `HOME` and
`GRADLE_USER_HOME`, and grant that user access only to its output and Gradle cache. When mounting
source read-only, provide writable temporary mounts at
`/source/third_party/onnxruntime/java/.gradle` and
`/source/third_party/onnxruntime/java/build`. Upstream `--build_java` runs `gradlew clean jar`
in that tree; a read-only `java/build` fails after native compilation has already started. Keep
the source, operator configuration, and build scripts read-only. The input-only check does not
exercise the upstream root-user guard or these Gradle output paths; the builder probes both
directories for writability before compiling.

The first packaging dry run passed after upstream Gradle installed Platform-Tools 37.0.1 into
its container. The recipe now installs `platform-tools` when building the image, so an unprivileged
native build does not need to modify the SDK during configuration. That updated image resolved to
`sha256:57d61051fc53c6b8d9b21ca2d370a4408dbcb8e86836d3ffa66a81c154e90622`.
The SDK package name is mutable; retain the resolved image ID and package revision with evidence.
A Gradle dry run verifies configuration and task selection, not successful compilation or packaging.

The updated image also passed `assembleRelease --dry-run --offline` with networking disabled,
UID/GID `501:20`, source mounted read-only, and writable output/cache mounts. Upstream Gradle
reported an analytics home-directory warning for the numeric UID, but configuration completed.
This confirms the packaging configuration needs no further SDK downloads; native compilation and
AAR comparison remain separate checks.

The Debian-CMake host-tool image resolved to
`sha256:34c6bfa469b1afb637cbe9e95c3a5ec2a5c5e4ad011953cce70d0d83073a1fe7`, still based on core
`sha256:425b1a57386654a8124fab1d5c02c182a4e9aa26fe0f4d15ed1c84f461c0187e`. Image build verified
`cmake`/`ninja` resolve to `/usr/bin` (CMake 3.31.6, ninja 1.11.1). An unprivileged smoke then
passed `--check-only`, 15/15 reproductions of the FetchContent `echo_append && touch` stamp
command, and an empty-command ExternalProject including its no-test stamp. That is not a native
AAR pair. The previous SDK-CMake image remains tagged `libreboard-runtime-build:sdk-cmake-57d61051`.
