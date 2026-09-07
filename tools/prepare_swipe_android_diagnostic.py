#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare bounded, hash-bound validation replay for the real Android CTC decoder."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import evaluate_swipe_ctc as evaluation
import model_sources
import swipe_model_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("build/model-data/swipe-latin-v1"))
    parser.add_argument("--export-report", type=pathlib.Path, default=pathlib.Path("build/model-export/swipe-latin-v1/export-report.json"))
    parser.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("build/device-evidence/swipe-android-diagnostic"))
    args = parser.parse_args()
    swipe_model_contract.load_prepared_manifest(args.data_root, require_pinned=True)
    report, model, report_hash = evaluation.load_export(args.export_report, development=False)
    if report["dataManifestSha256"] != model_sources.file_sha256(args.data_root / "split-manifest.json"):
        raise ValueError("export and replay corpus manifests differ")
    if report["modelSpecSha256"] != swipe_model_contract.load_spec().sha256:
        raise ValueError("export and model specifications differ")
    layout = evaluation.load_layout(args.data_root / "layout.json")
    rows = evaluation.select_rows(evaluation.load_test_rows(args.data_root / "validation.jsonl", layout, "validation"),
                                  sample_count=100, minimum_per_stratum=10)
    fixture = {
        "schemaVersion": 1, "releaseEligible": False, "split": "validation",
        "protocol": "android-ctc-native-dictionary-diagnostic-v1", "layout": layout,
        "modelSha256": report["model"]["sha256"], "exportReportSha256": report_hash,
        "sourceDataSha256": model_sources.file_sha256(args.data_root / "validation.jsonl"),
        "generatorSha256": model_sources.file_sha256(pathlib.Path(__file__)),
        "cases": [{"id": row.identifier, "sessionId": row.session_id, "target": row.target,
                   "path": list(row.path), "strata": sorted(row.strata)} for row in rows],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, args.output_root / "swipe.onnx")
    path = args.output_root / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"fixtureSha256": model_sources.file_sha256(path), "modelSha256": fixture["modelSha256"]}))


if __name__ == "__main__":
    main()
