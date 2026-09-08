#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare hash-bound reference-prefix sidecars for swipe diagnostics, never release evidence."""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import unicodedata

import audit_joint_model_splits as audit
import model_sources
import prepare_context_dataset as context
import prepare_swipe_dataset as swipe

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ("validation", "test")
MAXIMUM_SENTENCE_CODEPOINTS = 4096
MAXIMUM_CONTEXT_UTF16_UNITS = 256
DEFAULT_OUTPUT = ROOT / "build/evaluation-data/swipe-reference-context-v1"


def reference_prefix(raw, target):
    if raw.get("potentially_invalid_sentence") is not False:
        return None, "invalid_sentence_flag"
    sentence, index = raw.get("sentence"), raw.get("word_idx")
    if not isinstance(sentence, str) or len(sentence) > MAXIMUM_SENTENCE_CODEPOINTS:
        return None, "invalid_sentence"
    sentence = " ".join(unicodedata.normalize("NFKC", sentence).replace("\u2019", "'").split())
    if any(unicodedata.category(char) in ("Cc", "Cs") for char in sentence):
        return None, "unsafe_sentence"
    tokens = sentence.split()
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(tokens):
        return None, "invalid_word_idx"
    if swipe._trim_edge_punctuation(swipe._normalize_label(tokens[index])) != target:
        return None, "unaligned_word"
    prefix = " ".join(tokens[:index]) + (" " if index else "")
    # Android bounds String length in UTF-16 units. Drop a cut leading surrogate safely.
    encoded = prefix.encode("utf-16-le")
    prefix = encoded[-MAXIMUM_CONTEXT_UTF16_UNITS * 2:].decode("utf-16-le", errors="ignore")
    return prefix, None


def source_identity(raw, artifact, line, source, policy):
    if not isinstance(raw, dict):
        return None
    session, word = raw.get("session"), raw.get("word")
    if not isinstance(session, str) or not session or len(session) > 256 or not isinstance(word, str):
        return None
    if swipe.split_for_session(session, policy) not in SPLITS:
        return None
    target = swipe._trim_edge_punctuation(swipe._normalize_label(word.strip()))
    source_id = raw.get("id")
    if not isinstance(source_id, (str, int)) or isinstance(source_id, bool):
        source_id = line
    return hashlib.sha256((source.revision + "\0" + artifact.path + "\0" + str(source_id) + "\0"
                           + swipe.session_hash(session, policy) + "\0" + target + "\0" + str(line)).encode()).hexdigest()


def prepare(source_root, swipe_root, corpus_path, output_root):
    if output_root.exists():
        raise ValueError("output root already exists; choose a new directory to preserve prior evidence")
    policy = swipe.load_policy()
    manifest = model_sources.load_manifest()
    source = manifest.source(policy.source_id)
    if source.identifier != "futo-swipe-dataset-v1":
        raise ValueError("reference-prefix semantics are verified only for the pinned FUTO v1 source")
    corpus, corpus_payload = context._load_json(corpus_path, 256 * 1024, "swipe corpus manifest")
    if (corpus["sourceManifestSha256"] != manifest.sha256 or corpus["policySha256"] != policy.sha256
            or corpus["toolSha256"] != model_sources.file_sha256(pathlib.Path(swipe.__file__))):
        raise ValueError("swipe corpus does not match current source, policy and preparation code")
    wanted = {}
    for split in SPLITS:
        path = swipe_root / f"{split}.jsonl"
        for row in audit.verified_rows(path, corpus["outputs"][path.name], split):
            if row["id"] in wanted:
                raise ValueError("duplicate prepared swipe identity")
            wanted[row["id"]] = {key: row[key] for key in ("id", "sessionId", "split", "target", "strata")}
    counts = collections.Counter()
    strata = {split: collections.Counter() for split in SPLITS}
    found = set()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".swipe-reference-", dir=output_root.parent) as temporary:
        staged = pathlib.Path(temporary)
        with contextlib.ExitStack() as stack:
            handles = {split: stack.enter_context((staged / f"{split}.jsonl").open("wb")) for split in SPLITS}
            for name in policy.data_artifacts:
                artifact = source.artifact(name)
                path = model_sources.artifact_path(source_root, source, artifact)
                for line, raw in context._verified_json_lines(path, artifact, 2 * 1024 * 1024):
                    identity = source_identity(raw, artifact, line, source, policy)
                    if identity not in wanted:
                        continue
                    if identity in found:
                        raise ValueError("duplicate verified source mapping")
                    found.add(identity)
                    row = wanted[identity]
                    split = row["split"]
                    counts[(split, "matched")] += 1
                    prefix, rejection = reference_prefix(raw, row["target"])
                    if rejection:
                        counts[(split, rejection)] += 1
                        continue
                    counts[(split, "aligned")] += 1
                    counts[(split, "nonempty_prefix" if prefix else "empty_prefix")] += 1
                    strata[split].update(row["strata"])
                    record = {"schemaVersion": 1, **row, "languageTag": "en-US",
                              "contextOrigin": "publisher_reference_prefix",
                              "precedingReferenceContext": prefix}
                    handles[split].write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                                   ensure_ascii=False).encode() + b"\n")
        if len(found) != len(wanted):
            raise ValueError("prepared swipe identities are missing verified source mappings")
        outputs = {}
        for split in SPLITS:
            path = staged / f"{split}.jsonl"
            outputs[path.name] = {"records": counts[(split, "aligned")], "bytes": path.stat().st_size,
                                  "sha256": model_sources.file_sha256(path)}
        report = {
            "schemaVersion": 1, "diagnosticOnly": True, "releaseEligible": False,
            "contextOrigin": "publisher_reference_prefix", "sourceId": source.identifier,
            "license": source.license, "sourceManifestSha256": manifest.sha256,
            "swipeCorpusManifestSha256": hashlib.sha256(corpus_payload).hexdigest(),
            "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
            "helperSha256": {pathlib.Path(module.__file__).name: model_sources.file_sha256(pathlib.Path(module.__file__))
                              for module in (audit, context, swipe)},
            "maximumContextUtf16Units": MAXIMUM_CONTEXT_UTF16_UNITS, "outputs": outputs,
            "counts": {split: {key: count for (kind, key), count in sorted(counts.items()) if kind == split}
                       for split in SPLITS},
            "alignedStrata": {split: dict(sorted(value.items())) for split, value in strata.items()},
            "limitations": ["Prefixes come from publisher prompts, not captured prior editor input.",
                            "Target and following words are excluded from every context prefix.",
                            "Prepared CTC identities and splits are preserved; joint context-model holdout must be audited separately.",
                            "No model quality, full fusion, device, memory or release gate is qualified."],
        }
        (staged / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.rename(staged, output_root)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=model_sources.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--swipe-root", type=pathlib.Path, default=swipe.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--corpus-manifest", type=pathlib.Path, default=ROOT / "models/swipe/corpus-manifest.json")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.source_root, args.swipe_root, args.corpus_manifest, args.output_root), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as failure:
        print(f"Swipe reference-prefix preparation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
