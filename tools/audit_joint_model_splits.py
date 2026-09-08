#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Audit shared source sessions across prepared context/swipe corpora; never release evidence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

import model_sources
import prepare_context_dataset as context_data
import prepare_swipe_dataset as swipe_data

ROOT = pathlib.Path(__file__).resolve().parents[1]
STRATA = ("short", "medium", "long", "clean", "sloppy", "very_sloppy", "double_letter", "return_trip")
CONTEXT_SPLITS = ("train", "validation", "test", "absent")


class JointSplitAuditError(ValueError):
    pass


def verified_rows(path, details, split):
    count = size = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while line := stream.readline(2 * 1024 * 1024 + 1):
            size += len(line)
            if len(line) > 2 * 1024 * 1024 or size > details["bytes"]:
                raise JointSplitAuditError("prepared file exceeds its bounded manifest")
            digest.update(line)
            row = json.loads(line)
            if row["split"] != split:
                raise JointSplitAuditError("prepared row crosses its declared split")
            count += 1
            yield row
    if count != details["records"] or size != details["bytes"] or digest.hexdigest() != details["sha256"]:
        raise JointSplitAuditError("prepared records do not match their pinned manifest")


def count_overlap(rows, mapping):
    counts = collections.Counter()
    sessions = collections.defaultdict(set)
    strata = collections.Counter()
    for row in rows:
        session = row["sessionId"]
        if session not in mapping:
            raise JointSplitAuditError("a prepared swipe session has no verified source mapping")
        split = mapping[session]
        if split not in CONTEXT_SPLITS:
            raise JointSplitAuditError("unknown context split in source mapping")
        counts[split] += 1
        sessions[split].add(session)
        if split in ("test", "absent"):
            strata.update(row["strata"])
    eligible = counts["test"] + counts["absent"]
    return {
        "contextSplitRows": {key: counts[key] for key in CONTEXT_SPLITS},
        "contextSplitSessions": {key: len(sessions[key]) for key in CONTEXT_SPLITS},
        "contextUnseenRows": eligible,
        "contextUnseenStrata": {key: strata[key] for key in STRATA},
    }


def audit(args):
    source_manifest = model_sources.load_manifest()
    context_policy = context_data.load_policy()
    swipe_policy = swipe_data.load_policy()
    if context_policy.source_id != swipe_policy.source_id:
        raise JointSplitAuditError("this audit requires the same external source in both pipelines")
    source = source_manifest.source(context_policy.source_id)
    context_manifest, context_payload = context_data._load_json(args.context_manifest, 256 * 1024, "context manifest")
    swipe_manifest, swipe_payload = context_data._load_json(args.swipe_manifest, 256 * 1024, "swipe manifest")
    for manifest, policy, tool in (
        (context_manifest, context_policy, context_data.__file__),
        (swipe_manifest, swipe_policy, swipe_data.__file__),
    ):
        if (manifest["sourceManifestSha256"] != source_manifest.sha256
                or manifest["policySha256"] != policy.sha256
                or manifest["toolSha256"] != model_sources.file_sha256(pathlib.Path(tool))):
            raise JointSplitAuditError("prepared manifest does not match the current source/policy/tool")
    context_sessions = {}
    for split in context_data.SPLITS:
        path = args.context_root / f"{split}.sentences.jsonl"
        for row in verified_rows(path, context_manifest["outputs"][path.name], split):
            if row["sourceId"] != source.identifier:
                continue
            previous = context_sessions.setdefault(row["sessionId"], split)
            if previous != split:
                raise JointSplitAuditError("a context session crosses prepared splits")
    if not context_sessions:
        raise JointSplitAuditError("no shared-source context sessions were found")
    mapping = {}
    seen = set()
    for name in context_policy.data_artifacts:
        artifact = source.artifact(name)
        path = model_sources.artifact_path(args.source_root, source, artifact)
        for _, row in context_data._verified_json_lines(path, artifact, context_policy.maximum_line_bytes):
            session = row.get("session") if isinstance(row, dict) else None
            if not isinstance(session, str) or not session or len(session) > 256 or session in seen:
                continue
            seen.add(session)
            mapping[swipe_data.session_hash(session, swipe_policy)] = context_sessions.get(
                context_data.session_hash(source.identifier, session, context_policy), "absent",
            )
    counts = {}
    for split in ("validation", "test"):
        path = args.swipe_root / f"{split}.jsonl"
        counts[split] = count_overlap(verified_rows(path, swipe_manifest["outputs"][path.name], split), mapping)
    test = counts["test"]
    return {
        "schemaVersion": 1, "diagnosticOnly": True, "releaseEligible": False,
        "scope": "Source-session overlap only; no inference or quality qualification",
        "sourceManifestSha256": source_manifest.sha256,
        "contextCorpusManifestSha256": hashlib.sha256(context_payload).hexdigest(),
        "swipeCorpusManifestSha256": hashlib.sha256(swipe_payload).hexdigest(),
        "auditToolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
        "counts": counts,
        "jointTestPoolMeetsCountMinimums": test["contextUnseenRows"] >= 5000
            and all(test["contextUnseenStrata"][key] >= 500 for key in STRATA),
        "limitations": ["Absent means no accepted context sentence from that source session.",
                        "Session separation does not prove prompt/text separation across sessions.",
                        "Component-only diagnostics do not invoke the other model."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=model_sources.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--context-root", type=pathlib.Path, default=context_data.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--swipe-root", type=pathlib.Path, default=swipe_data.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--context-manifest", type=pathlib.Path, default=context_data.DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--swipe-manifest", type=pathlib.Path, default=ROOT / "models/swipe/corpus-manifest.json")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
