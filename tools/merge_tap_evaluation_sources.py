#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Combine verified evaluation components without changing IDs, session grouping, or labels."""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import tempfile

import model_sources
import prepare_tap_evaluation as tap


class MergeError(ValueError):
    pass


def merge(manifest_paths, output_root):
    if not 1 <= len(manifest_paths) <= 32:
        raise MergeError("supply between one and 32 component manifests")
    policy = tap.load_policy()
    components, records = [], []
    datasets, identifiers, licenses = set(), set(), set()
    session_components = collections.defaultdict(set)
    any_human = False
    for path in manifest_paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 65536:
            raise MergeError("component manifest is missing, linked, or oversized")
        definition = json.loads(path.read_bytes())
        filename = definition.get("dataFile") if isinstance(definition, dict) else None
        if not isinstance(filename, str) or pathlib.PurePath(filename).name != filename:
            raise MergeError("component dataFile must be a leaf filename")
        data = path.parent / filename
        if not data.is_file() or data.is_symlink():
            raise MergeError("component data is missing or linked")
        component = tap.load_source_manifest(path, data)
        if component.dataset_id in datasets:
            raise MergeError("component dataset IDs must be unique")
        datasets.add(component.dataset_id)
        licenses.add(component.license)
        if len(licenses) != 1:
            raise MergeError("component licenses differ; merger cannot choose a combined license")
        row_count, human = 0, False
        with data.open("rb") as stream:
            while line := stream.readline(policy.maximum_line_bytes + 1):
                if len(line) > policy.maximum_line_bytes:
                    raise MergeError("component row exceeds its byte bound")
                if not line.strip():
                    continue
                raw = json.loads(line)
                normalized = tap.normalize_record(raw, row_count + 1, policy)
                if normalized["id"] in identifiers:
                    raise MergeError("duplicate row ID across evaluation components")
                identifiers.add(normalized["id"])
                session_components[normalized["sessionId"]].add(component.dataset_id)
                human |= raw["collectionMethod"] != "project_authored"
                records.append(raw)
                row_count += 1
        if row_count == 0 or human != component.contains_human_contributions:
            raise MergeError("component rows disagree with declared human provenance or are empty")
        any_human |= human
        components.append({"datasetId": component.dataset_id, "manifestSha256": component.sha256,
                           "manifest": definition, "records": row_count})
    components.sort(key=lambda value: value["datasetId"])
    records.sort(key=lambda value: value["id"])
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        staged = pathlib.Path(temporary)
        provenance = staged / "component-manifests.json"
        provenance.write_text(json.dumps({"schemaVersion": 1, "components": components}, sort_keys=True, indent=2) + "\n")
        provenance_hash = model_sources.file_sha256(provenance)
        data = staged / "tap-cases.jsonl"
        with data.open("wb") as stream:
            for record in records:
                stream.write(tap._canonical(record))
        protocol = (
            "Unmodified evaluation rows from independently hash-validated component manifests. "
            "Original IDs, participant/session grouping, labels, text and points are preserved. "
            f"Exact source manifests are retained in component-manifests.json (SHA-256 {provenance_hash})."
        )
        consent = ("Original human-contribution and consent statements are preserved per component in "
                   f"component-manifests.json (SHA-256 {provenance_hash}). The merger adds no consent claim.") if any_human else ""
        manifest = {"schemaVersion": 1, "datasetId": "libreboard-merged-tap-evaluation-v1",
                    "dataFile": data.name, "dataSha256": model_sources.file_sha256(data),
                    "license": next(iter(licenses)), "collectionProtocol": protocol,
                    "containsHumanContributions": any_human, "consentStatement": consent}
        report = {"schemaVersion": 1, "releaseEligible": False, "records": len(records),
                  "components": len(components), "sessions": len(session_components),
                  "crossComponentSessions": sum(len(values) > 1 for values in session_components.values()),
                  "componentManifestsSha256": provenance_hash,
                  "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
                  "nextStep": "Run prepare_tap_evaluation.py; merging does not satisfy corpus or quality gates."}
        for name, value in (("source-manifest.json", manifest), ("merge-report.json", report)):
            (staged / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        for name in (data.name, provenance.name, "source-manifest.json", "merge-report.json"):
            os.replace(staged / name, output_root / name)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(merge(args.component, args.output_root), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as failure:
        print(f"tap component merge failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
