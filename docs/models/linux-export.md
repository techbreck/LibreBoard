# Clean Linux model export verification

On 2026-09-08, two fresh Linux amd64 containers exported the same completed swipe training
checkpoint with networking disabled and the repository mounted read-only. All nine output files
were byte-identical, including the model, reports, manifests, operator config, license and model card.
The 1,748,153-byte ONNX file also matched the earlier macOS arm64 export:
`1301f0d076f526fe0b67f2ea86a38abc8448f6de54d2c7191c002dcd4d05d38c`.

The minimal Linux environment exposed a missing dependency: `safetensors.torch` imports `packaging`
without declaring it in the base safetensors wheel. `requirements.in` and the minimal swipe lock now
pin `packaging==26.3`; the context lock already pinned the same version. CI exercises swipe training
and export tests before installing the context extras so this class of omission cannot be hidden.
The clean environment passed five training and three export tests, with one expected default-
environment test skipped in each suite.

The retained local report is `build/linux-exports/swipe-reproducibility.json`, SHA-256
`abe58471151ca321edaf1875e165ddf6742a04c6615751b8571699c16ad40f20`. It records the hashes of every
output file, the lockfile, Dockerfile and image identity. The host used amd64 emulation on arm64;
this is an export-byte comparison, not latency evidence. The image ID was
`sha256:4e91e5435f0d6daface37b9179474a68db45f7b90edb37fd25fdea675531b9f1`.

## Reproduction

Stage only the small build context, not the repository's downloaded training data:

```sh
mkdir -p build/linux-export-context
cp models/training/Dockerfile.swipe-export build/linux-export-context/Dockerfile
cp models/training/requirements-linux-x86_64.lock build/linux-export-context/requirements.lock
docker build --platform linux/amd64 -t libreboard-model-export:local build/linux-export-context
mkdir -p build/linux-exports/swipe-a build/linux-exports/swipe-b
```

Run this command once with `swipe-a` and again with `swipe-b`. Each invocation creates a fresh
container. Set `LIBREBOARD_ROOT` to the absolute checkout path before running it.

```sh
docker run --rm --platform linux/amd64 --network none --cpus 1 --memory 2g \
  -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/source \
  --mount "type=bind,src=$LIBREBOARD_ROOT,dst=/source,readonly" \
  --mount "type=bind,src=$LIBREBOARD_ROOT/build/linux-exports/swipe-a,dst=/output" \
  libreboard-model-export:local python tools/export_swipe_model.py --output-root /output/export
diff -rq build/linux-exports/swipe-a/export build/linux-exports/swipe-b/export
```

The Dockerfile pins its Python/Debian base by digest and installs the hash-locked Python packages.
OS package installation uses Debian's signed repositories; this procedure does not claim byte-
identical container-image rebuilding. It proves repeated model export from a fixed trained checkpoint.
It does not prove deterministic retraining, full context-model export, independently rebuilt Android APKs,
or the still-required Phase 0 quality, device, signing and GrapheneOS gates.

## Development context export

Two fresh Linux amd64 containers also exported a new, provenance-valid 16-example development
context checkpoint using the unchanged `requirements-context-linux-x86_64.lock`. All eight output
files matched byte-for-byte. The INT4 ONNX SHA-256 was
`6789ab54a0607c9731387d7ac09d818a4f04f509167b94b7a6a28e77d1f14e08`.
The exported graph passed the exporter's ABI, operator and runtime checks, then ran through
`evaluate_context_model.py` on 16 validation slates. This is a runtime smoke only; its tiny training
run and host inference times do not qualify model quality or Android performance.

Use the Dockerfile above with the context lock copied as `requirements.lock`, allocate `--memory 4g`,
and replace the export command with:

```sh
python tools/export_context_model.py --development \
  --training-report build/model-training/context-linux-export-smoke/training-report-development.json \
  --output-root /output/export
```

The retained comparison is `build/linux-exports/context-smoke-reproducibility.json`, SHA-256
`781b174af9a6f982d3e4b2d7d4c4bb680eac2213e11984b9afe17ba74327af51`. It binds all output files,
the pinned lock, image identity, training report and runtime smoke report. Full-candidate evidence
is recorded separately below.

## Full context candidate

The completed four-epoch student has two byte-identical Linux exports across all eight files. Use
the same context environment with the full report and omit `--development`:

```sh
python tools/export_context_model.py \
  --training-report build/model-training/context-en-de-v1/training-report.json \
  --output-root /output/export
```

The canonical model SHA-256 is
`53e1a1d29566efaa4ca3fe1b396206c10d211cdb2394d231dc0413b146dc17a2`. The retained comparison is
`build/linux-exports/context-full-reproducibility.json`, SHA-256
`525aba62ad5bd6725f3b32f5264a525f38ce611adf6795808a0f85685822ca04`.
The macOS export differs slightly in quantized and constant tensor bytes, so validation, test and
Android kernel parity were repeated against the canonical Linux artifact. Their metrics and the
remaining acceptance gates are in the [context model card](../../models/context/MODEL_CARD.md).
This closes repeatable export of this checkpoint; it is not deterministic retraining or release
qualification.
