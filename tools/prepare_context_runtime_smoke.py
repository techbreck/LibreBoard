#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Bind synthetic Android kernel-parity inputs to a checked context export; never quality evidence."""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import context_model_contract
import evaluate_context_model
import export_context_model
import model_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-report", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    spec = context_model_contract.load_spec()
    report, model, _, export_hash = evaluate_context_model.load_export(args.export_report, spec, args.development)
    torch, np, onnx, ort, *rest = export_context_model._dependencies()
    if report["toolchain"] != rest[-1]:
        raise ValueError("runtime smoke toolchain differs from checked export")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    cases = []
    for rows in (1, 8, 32):
        ids = np.zeros((rows, 32), dtype=np.int64)
        ids[:, :4] = [1, 3, 6, 7]
        ids[:, 24:26] = [8, 9]
        attention = np.zeros_like(ids)
        attention[:, :4] = 1
        attention[:, 24:26] = 1
        candidate = np.zeros((rows, 32), dtype=np.float32)
        candidate[:, 24:26] = 1
        scores = session.run(["candidate_log_likelihood"], {
            "input_ids": ids, "attention_mask": attention, "candidate_mask": candidate,
            "field_class": np.zeros(rows, dtype=np.int64),
        })[0]
        if scores.shape != (rows,) or not np.isfinite(scores).all():
            raise ValueError("invalid reference scores")
        cases.append({"rows": rows, "expectedScores": scores.tolist()})
    fixture = {
        "schemaVersion": 1, "protocol": "context-synthetic-kernel-parity-v1",
        "releaseEligible": False, "modelSha256": report["model"]["sha256"],
        "exportReportSha256": export_hash, "generatorSha256": model_sources.file_sha256(pathlib.Path(__file__)),
        "absoluteTolerance": 0.001, "cases": cases,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, args.output_root / "context.onnx")
    path = args.output_root / "fixture.json"
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fixtureSha256": model_sources.file_sha256(path), "modelSha256": fixture["modelSha256"]}))


if __name__ == "__main__":
    main()
