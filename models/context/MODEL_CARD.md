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

The pinned corpus deterministically produces 16,384 tokens and 16,341 merges in 530,571 bytes, with
SHA-256 `1395e285927bfbfa5888dc7c83e4f57dfcfbfeb54f29a3cf2437f5db3d71d6a2`.
The builder compares 256 English, German and Unicode probes against the dependency-free runtime BPE
algorithm before publishing the tokenizer.

## Training provenance

The only teacher approved for v1 is Apache-2.0
[`Evicka/Hanse2-100M-Base`](https://huggingface.co/Evicka/Hanse2-100M-Base) at commit
`0a834967424be0ec471f2846646d1c75906a9a9c`. Its exact source files and hashes are pinned in
`models/sources/v1.json`. The teacher is a build-only input and is never packaged. Release weights
must be LibreBoard-trained Apache-2.0 weights, and the training report must bind the student spec,
tokenizer, data manifest and exact teacher hash.

The committed `corpus-manifest.json` binds a deterministic 90/5/5 whole-session split to the source
manifest, data policy, preparation tool and project-authored German data. It contains 68,748 accepted
sentences: 49,872 English US and 18,876 German. The held-out split contains 2,650 English and 1,114
German sentences. The FUTO source contributes only its public sentence field after strict filtering;
raw gesture paths, timestamps and source session identifiers are discarded. The German portion is
expanded from reviewed Apache-2.0 templates covering ordinary context, messages/search, contractions,
confusion pairs, privacy vocabulary and compounds. `tools/score_context_teacher.py` then constructs
bounded same-language slates and computes the actual mean conditional log probability of every
candidate with the verified offline Hanse2 model. Its emitted records retain only bounded student
prefix IDs, candidate surfaces/IDs, provenance classes and teacher scores; source session IDs and
teacher-prefix text are not copied. Scoring is resumable through atomic chunks whose bindings include
the app commit, tool, policy, corpus, tokenizer, teacher and toolchain. The full teacher-scored corpus
and student training remain outstanding. Development/random weights and synthetic reports must remain
under `build/`, carry development filenames and cannot produce a release export.

Student training uses 128 candidate rows per optimizer step, implemented as 16 vectorized but
prefix-isolated eight-candidate groups. Its committed objective combines temperature-scaled teacher
cross-entropy (weight 1.0), observed-candidate cross-entropy (0.5), and a 0.2 observed-versus-negative
margin (weight 0.25). The order is deterministically buffer-shuffled, and atomic checkpoints every
4,096 examples retain model, AdamW, RNG and exact mid-epoch progress. Validation is measured after
every epoch, while the untouched test split is evaluated once for the final training report. Bounded
development smoke data has passed teacher scoring, student forward/backward, safetensors publication,
INT4 export, exact operator validation and ONNX Runtime CPU inference; this is not model-quality evidence.

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

python3 tools/model_sources.py verify
build/context-model-venv/bin/python tools/prepare_context_dataset.py
build/context-model-venv/bin/python tools/build_context_tokenizer.py
build/context-model-venv/bin/python tools/score_context_teacher.py --threads 4
build/context-model-venv/bin/python tools/train_context_model.py --threads 4
build/context-model-venv/bin/python tools/export_context_model.py
```

The teacher-scoring and student-training commands will be added with their immutable reports; until
then this is deliberately not a complete model reproduction recipe.

## Required acceptance evidence

The candidate must improve context-sensitive error by at least 15% relative to dictionary/spatial
fusion, improve valid-word accuracy by at least five percentage points, and add no more than 0.5
percentage points of false correction. Tap p95 remains below 80 ms end to end and total added neural
memory remains at most 64 MiB. The model pack, core-only fallback and timeout/circuit-breaker behavior
must pass on supported physical GrapheneOS Pixel hardware without sandboxed Google Play. Two clean
Linux exports and model-pack builds must be byte-identical before signing.
