# LibreBoard Context English/German v1

## Status

No release weights are accepted yet. This card defines the fixed architecture, tokenizer and export
contracts that a trained `context-en-de-v1.onnx` candidate must satisfy before it can enter the
official data-only model pack.

## Purpose and limits

The model ranks at most 32 candidates supplied by LibreBoard from static dictionaries, touch
geometry and the personal store. It receives only a bounded preceding prefix, a coarse non-sensitive
field class, one language token and each candidate's wordpieces. It returns one finite score per
candidate. It is not exposed as a general text generator, and sensitive, email, URI, incognito or
otherwise restricted requests never reach tokenization or inference.

The student is an eight-layer decoder-only Transformer with width 512, eight attention heads,
SwiGLU width 1,536, RoPE, RMSNorm and tied token embeddings. Its exact trainable parameter count is
35,662,848. Candidate rows for one language share the 24-token prefix computation and use a fixed
eight-token suffix. English and German candidates are inferred in separate groups so identical
surface forms cannot exchange language evidence.

## Tokenizer

`tokenizer.json` is data-only schema 1 BPE with Unicode `NFKC_LOWER` normalization. It must have
exactly 16,384 dense IDs, the word-start symbol `▁`, ranked non-duplicate merges, distinct
padding/BOS/unknown tokens and exact `en`/`de` language-token mappings. Both the Python exporter and
the Kotlin runtime enforce the same 2 MiB ceiling and structural rules. No tokenizer implementation
is imported from the teacher or model pack.

## Training provenance

The only teacher approved for v1 is Apache-2.0
[`Evicka/Hanse2-100M-Base`](https://huggingface.co/Evicka/Hanse2-100M-Base) at commit
`0a834967424be0ec471f2846646d1c75906a9a9c`. Its exact source files and hashes are pinned in
`models/sources/v1.json`. The teacher is a build-only input and is never packaged. Release weights
must be LibreBoard-trained Apache-2.0 weights, and the training report must bind the student spec,
tokenizer, data manifest and exact teacher hash.

The bilingual, session-separated candidate corpus and its deterministic preparation/distillation
commands are not accepted yet. Development/random weights and synthetic reports must remain under
`build/`, carry development filenames and cannot produce a release export.

## Reproducible export contract

The build-only x86-64 Linux toolchain is hash-locked in
`models/training/requirements-context-linux-x86_64.lock`. `tools/export_context_model.py` loads only
hash-bound safetensors and a structurally valid tokenizer, exports an opset-18 source graph, applies
ONNX Runtime 1.26 blockwise asymmetric INT4 quantization, and verifies the final opset-21 graph. The
fixed quantized graph contains 56 `com.microsoft::MatMulNBits` nodes and four
`com.microsoft::GatherBlockQuantized` nodes. It must pass the full ONNX checker, exact dynamic tensor
ABI, no-external-data rule, 24 MiB ceiling and CPU runtime smoke at batch sizes 1 and 32.

A production export will use:

```sh
uv venv --python 3.11 build/context-model-venv
uv pip install --python build/context-model-venv/bin/python \
  --require-hashes --torch-backend cpu \
  -r models/training/requirements-context-linux-x86_64.lock

build/context-model-venv/bin/python tools/export_context_model.py
```

The first command that prepares and trains the accepted corpus will be added here with its immutable
data manifest; until then this is deliberately not a complete reproduction recipe.

## Required acceptance evidence

The candidate must improve context-sensitive error by at least 15% relative to dictionary/spatial
fusion, improve valid-word accuracy by at least five percentage points, and add no more than 0.5
percentage points of false correction. Tap p95 remains below 80 ms end to end and total added neural
memory remains at most 64 MiB. The model pack, core-only fallback and timeout/circuit-breaker behavior
must pass on supported physical GrapheneOS Pixel hardware without sandboxed Google Play. Two clean
Linux exports and model-pack builds must be byte-identical before signing.
