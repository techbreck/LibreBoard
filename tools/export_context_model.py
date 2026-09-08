#!/usr/bin/env python3
"""Export and verify LibreBoard's INT4 bilingual candidate rescorer."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any

import context_model_contract
import context_tokenizer_contract
import export_swipe_model
import model_sources


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = ROOT / "build" / "model-training" / "context-en-de-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-export" / "context-en-de-v1"
EXPECTED_CUSTOM_OPERATORS = {
    "com.microsoft::GatherBlockQuantized",
    "com.microsoft::MatMulNBits",
}


class ContextExportError(ValueError):
    pass


def _dependencies():
    try:
        import numpy
        import onnx
        import onnx_ir
        import onnxruntime
        import safetensors
        import torch
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as failure:
        raise ContextExportError(
            "install the pinned context export dependencies from "
            "models/training/requirements-context-linux-x86_64.lock"
        ) from failure
    versions = {
        "torch": str(torch.__version__).split("+", 1)[0],
        "numpy": numpy.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxIr": onnx_ir.__version__,
        "safetensors": safetensors.__version__,
    }
    expected = {
        "torch": "2.8.0",
        "numpy": "2.2.6",
        "onnx": "1.19.0",
        "onnxruntime": "1.26.0",
        "onnxIr": "1.0.0",
        "safetensors": "0.6.2",
    }
    if versions != expected:
        raise ContextExportError(f"context export dependency versions do not match the lock: {versions}")
    return (
        torch, numpy, onnx, onnxruntime, safe_open, load_file,
        DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer, versions,
    )


def _load_training_report(
    path: pathlib.Path,
    development: bool,
    spec: context_model_contract.ContextModelSpec,
) -> dict[str, Any]:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise ContextExportError("context training report is missing, linked, or too large")
        report = json.loads(path.read_bytes())
    except ContextExportError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise ContextExportError(f"cannot read context training report: {failure}") from failure
    if not isinstance(report, dict) or report.get("schemaVersion") != 1:
        raise ContextExportError("context training report has an unsupported schema")
    if report.get("modelId") != "context-en-de-v1":
        raise ContextExportError("context training report has the wrong model identity")
    if report.get("modelSpecSha256") != spec.sha256:
        raise ContextExportError("context training report was produced from a different model spec")
    if report.get("parameterCount") != spec.raw["parameterCount"]:
        raise ContextExportError("context training report has the wrong parameter count")
    if report.get("releaseEligible") is not True and not development:
        raise ContextExportError("development or partial training cannot produce a release context model")
    for field in ("appCommit", "dataManifestSha256"):
        value = report.get(field)
        pattern = model_sources.REVISION if field == "appCommit" else model_sources.SHA256
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ContextExportError(f"context training report has an invalid {field}")
    teacher = report.get("teacher")
    if not isinstance(teacher, dict) or set(teacher) != {"modelSha256"}:
        raise ContextExportError("context training report has invalid teacher metadata")
    teacher_hash = teacher["modelSha256"]
    if not isinstance(teacher_hash, str) or not model_sources.SHA256.fullmatch(teacher_hash):
        raise ContextExportError("context training report has an invalid teacher model hash")
    teacher_source = model_sources.load_manifest().source("hanse2-100m-base-teacher-v1")
    if teacher_hash != teacher_source.artifact("model.safetensors").sha256:
        raise ContextExportError("context training report is not bound to the pinned teacher model")
    provenance = report.get("provenance")
    if (
        not isinstance(provenance, list)
        or not provenance
        or len(provenance) > 64
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "revision", "license", "source_url"}
            or any(not isinstance(item.get(field), str) or not item[field] for field in ("name", "revision", "license"))
            or not isinstance(item.get("source_url"), str)
            or not item["source_url"].startswith("https://")
            for item in provenance
        )
    ):
        raise ContextExportError("context training report has invalid provenance")
    if not any(
        item["revision"] == teacher_source.revision
        and item["license"] == teacher_source.license
        and item["source_url"] == teacher_source.source_url
        for item in provenance
    ):
        raise ContextExportError("context training report provenance omits the pinned teacher")
    for artifact_name in ("weights", "tokenizer"):
        artifact = report.get(artifact_name)
        if not isinstance(artifact, dict) or set(artifact) != {"file", "bytes", "sha256"}:
            raise ContextExportError(f"context training report has no {artifact_name} artifact")
        filename = artifact["file"]
        if (
            not isinstance(filename, str)
            or pathlib.PurePosixPath(filename).name != filename
            or filename in {"", ".", ".."}
        ):
            raise ContextExportError(f"context training report has an invalid {artifact_name} filename")
        if not isinstance(artifact.get("sha256"), str) or not model_sources.SHA256.fullmatch(artifact["sha256"]):
            raise ContextExportError(f"context training report has an invalid {artifact_name} hash")
        if isinstance(artifact.get("bytes"), bool) or not isinstance(artifact.get("bytes"), int) or artifact["bytes"] <= 0:
            raise ContextExportError(f"context training report has an invalid {artifact_name} size")
    return report


def _tensor_shape(value_info) -> tuple[int, ...]:
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param") and dimension.dim_param == "batch":
            dimensions.append(-1)
        else:
            raise ContextExportError(f"ONNX tensor {value_info.name} has an unexpected dynamic dimension")
    return tuple(dimensions)


def _validate_tensor_abi(model, spec: context_model_contract.ContextModelSpec, onnx) -> None:
    type_names = {"float32": onnx.TensorProto.FLOAT, "int64": onnx.TensorProto.INT64}
    expected_inputs = {
        tensor["name"]: (type_names[tensor["elementType"]], tuple(tensor["shape"]))
        for tensor in spec.export["inputs"]
    }
    expected_outputs = {
        tensor["name"]: (type_names[tensor["elementType"]], tuple(tensor["shape"]))
        for tensor in spec.export["outputs"]
    }
    initializers = {initializer.name for initializer in model.graph.initializer}
    actual_inputs = {
        value.name: (value.type.tensor_type.elem_type, _tensor_shape(value))
        for value in model.graph.input
        if value.name not in initializers
    }
    actual_outputs = {
        value.name: (value.type.tensor_type.elem_type, _tensor_shape(value))
        for value in model.graph.output
    }
    if actual_inputs != expected_inputs:
        raise ContextExportError(f"context ONNX inputs violate the fixed tensor ABI: {actual_inputs}")
    if actual_outputs != expected_outputs:
        raise ContextExportError(f"context ONNX outputs violate the fixed tensor ABI: {actual_outputs}")


def _runtime_smoke(model_path: pathlib.Path, onnxruntime, numpy) -> None:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for rows in (1, 32):
        input_ids = numpy.zeros((rows, 32), dtype=numpy.int64)
        input_ids[:, :4] = numpy.asarray([1, 3, 6, 7], dtype=numpy.int64)
        input_ids[:, 24:26] = numpy.asarray([8, 9], dtype=numpy.int64)
        attention_mask = numpy.zeros((rows, 32), dtype=numpy.int64)
        attention_mask[:, :4] = 1
        attention_mask[:, 24:26] = 1
        candidate_mask = numpy.zeros((rows, 32), dtype=numpy.float32)
        candidate_mask[:, 24:26] = 1
        scores = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "candidate_mask": candidate_mask,
            "field_class": numpy.zeros((rows,), dtype=numpy.int64),
        })[0]
        if scores.shape != (rows,) or not numpy.isfinite(scores).all():
            raise ContextExportError("quantized context model failed its CPU runtime smoke test")


def export(args: argparse.Namespace) -> dict[str, Any]:
    (
        torch, numpy, onnx, onnxruntime, safe_open, load_file,
        DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer, versions,
    ) = _dependencies()
    spec = context_model_contract.load_spec(args.spec)
    report_path = args.training_report.resolve()
    report = _load_training_report(report_path, args.development, spec)
    try:
        model_sources.verify_git_sources_at_commit(
            report["appCommit"],
            ("tools/train_context_model.py", "models/training/context_model.py"),
        )
    except model_sources.ModelSourceError as failure:
        raise ContextExportError(f"context model source provenance failed: {failure}") from failure
    artifacts = {}
    for artifact_name in ("weights", "tokenizer"):
        details = report[artifact_name]
        path = report_path.parent / details["file"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != details["bytes"]:
            raise ContextExportError(f"context {artifact_name} artifact is missing or has the wrong size")
        if model_sources.file_sha256(path) != details["sha256"]:
            raise ContextExportError(f"context {artifact_name} hash does not match its report")
        artifacts[artifact_name] = path
    tokenizer = context_tokenizer_contract.load_tokenizer(
        artifacts["tokenizer"],
        expected_vocabulary_size=spec.architecture["vocabularySize"],
    )
    if tokenizer.sha256 != report["tokenizer"]["sha256"]:
        raise ContextExportError("validated context tokenizer hash does not match the training report")

    with safe_open(artifacts["weights"], framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
    expected_metadata = {
        "appCommit": report["appCommit"],
        "dataManifestSha256": report["dataManifestSha256"],
        "modelSpecSha256": report["modelSpecSha256"],
        "teacherModelSha256": report["teacher"]["modelSha256"],
        "tokenizerSha256": report["tokenizer"]["sha256"],
    }
    if metadata != expected_metadata:
        raise ContextExportError("context safetensors metadata does not match the training report")

    sys.path.insert(0, str(ROOT))
    from models.training.context_model import ContextCandidateModel, trainable_parameter_count

    model = ContextCandidateModel(spec.architecture).eval()
    model.load_state_dict(load_file(artifacts["weights"], device="cpu"), strict=True)
    if trainable_parameter_count(model) != spec.raw["parameterCount"]:
        raise ContextExportError("loaded context model parameter count violates the spec")

    rows = 2
    input_ids = torch.zeros((rows, 32), dtype=torch.long)
    input_ids[:, :4] = torch.tensor([1, 3, 6, 7])
    input_ids[:, 24:26] = torch.tensor([8, 9])
    attention_mask = torch.zeros((rows, 32), dtype=torch.long)
    attention_mask[:, :4] = 1
    attention_mask[:, 24:26] = 1
    candidate_mask = torch.zeros((rows, 32), dtype=torch.float32)
    candidate_mask[:, 24:26] = 1
    field_class = torch.zeros((rows,), dtype=torch.long)
    dynamic_axes = {
        tensor["name"]: {0: "batch"}
        for tensor in (*spec.export["inputs"], *spec.export["outputs"])
    }

    output_root = args.output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libreboard-context-export-", dir=output_root.parent) as temporary:
        staging = pathlib.Path(temporary)
        fp32_path = staging / "model-fp32.onnx"
        torch.onnx.export(
            model,
            (input_ids, attention_mask, candidate_mask, field_class),
            fp32_path,
            input_names=[tensor["name"] for tensor in spec.export["inputs"]],
            output_names=[tensor["name"] for tensor in spec.export["outputs"]],
            opset_version=spec.export["sourceOpsetVersion"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
        )
        exported = onnx.load_model(fp32_path, load_external_data=False)
        onnx.checker.check_model(exported, full_check=True)
        _validate_tensor_abi(exported, spec, onnx)
        export_swipe_model._reject_external_data(exported, onnx)

        logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
        quantizer = MatMulNBitsQuantizer(
            exported,
            algo_config=DefaultWeightOnlyQuantConfig(
                block_size=spec.export["quantizationBlockSize"],
                is_symmetric=False,
                op_types_to_quantize=("MatMul", "Gather"),
                bits=4,
            ),
        )
        quantizer.process()
        quantized = quantizer.model.model
        export_swipe_model._strip_nonsemantic_metadata(quantized)
        export_swipe_model._reject_external_data(quantized, onnx)
        _validate_tensor_abi(quantized, spec, onnx)
        onnx.checker.check_model(quantized, full_check=True)
        opsets = {entry.domain or "ai.onnx": entry.version for entry in quantized.opset_import}
        if opsets.get("ai.onnx") != spec.export["opsetVersion"] or opsets.get("com.microsoft") != 1:
            raise ContextExportError(f"quantized context model has unexpected opsets: {opsets}")

        required = export_swipe_model.required_operators(quantized, onnx)
        canonical_operators = export_swipe_model._canonical_operator_list(required)
        custom = {operator for operator in canonical_operators if "::" in operator}
        if custom != EXPECTED_CUSTOM_OPERATORS:
            raise ContextExportError(f"quantized context model has unexpected custom operators: {sorted(custom)}")
        custom_counts = {
            operator: sum(
                1 for node in quantized.graph.node
                if f"{node.domain}::{node.op_type}" == operator
            )
            for operator in sorted(EXPECTED_CUSTOM_OPERATORS)
        }
        if custom_counts != {
            "com.microsoft::GatherBlockQuantized": 4,
            "com.microsoft::MatMulNBits": 7 * spec.architecture["layers"],
        }:
            raise ContextExportError(f"quantized context operator counts drifted: {custom_counts}")

        development = args.development or report.get("releaseEligible") is not True
        model_filename = "context-en-de-v1-development.onnx" if development else "context-en-de-v1.onnx"
        model_path = staging / model_filename
        model_path.write_bytes(quantized.SerializeToString(deterministic=True))
        if model_path.stat().st_size > spec.export["maximumModelBytes"]:
            raise ContextExportError(
                f"INT4 context model is {model_path.stat().st_size} bytes; "
                f"limit is {spec.export['maximumModelBytes']}"
            )
        verified = onnx.load_model(model_path, load_external_data=False)
        onnx.checker.check_model(verified, full_check=True)
        export_swipe_model._reject_external_data(verified, onnx)
        _validate_tensor_abi(verified, spec, onnx)
        _runtime_smoke(model_path, onnxruntime, numpy)

        tokenizer_name = "tokenizer.json"
        shutil.copyfile(artifacts["tokenizer"], staging / tokenizer_name)
        manifest = {
            "schemaVersion": 1,
            "engineAbi": spec.raw["engineAbi"],
            "modelKind": "context-rescorer",
            "tensorAbi": spec.raw["tensorAbi"],
            "locales": spec.raw["locales"],
            "architecture": spec.raw["architectureName"],
            "parameterCount": spec.raw["parameterCount"],
            "quantization": spec.export["quantization"],
            "modelSha256": model_sources.file_sha256(model_path),
            "tokenizerSha256": report["tokenizer"]["sha256"],
            "requiredOnnxOperators": canonical_operators,
            "license": spec.raw["license"],
            "provenance": report["provenance"],
            "minimumAppVersionCode": 1,
        }
        manifest_name = "manifest-development.json" if development else "manifest.json"
        (staging / manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        operators_name = "required_operators-development.config" if development else "required_operators.config"
        (staging / operators_name).write_text(export_swipe_model._operator_config(required))
        shutil.copyfile(ROOT / "LICENSE-Apache-2.0", staging / "LICENSE")
        shutil.copyfile(spec.path, staging / "model.json")
        shutil.copyfile(report_path, staging / report_path.name)

        export_report = {
            "schemaVersion": 1,
            "modelId": spec.raw["modelId"],
            "releaseEligible": not development,
            "appCommit": report["appCommit"],
            "modelSpecSha256": spec.sha256,
            "dataManifestSha256": report["dataManifestSha256"],
            "trainingReportSha256": model_sources.file_sha256(report_path),
            "weightsSha256": report["weights"]["sha256"],
            "model": {
                "file": model_filename,
                "bytes": model_path.stat().st_size,
                "sha256": model_sources.file_sha256(model_path),
            },
            "tokenizer": {
                "file": tokenizer_name,
                "bytes": (staging / tokenizer_name).stat().st_size,
                "sha256": model_sources.file_sha256(staging / tokenizer_name),
            },
            "manifest": manifest_name,
            "requiredOperators": operators_name,
            "customOperatorCounts": custom_counts,
            "toolchain": versions,
        }
        export_report_name = "export-report-development.json" if development else "export-report.json"
        (staging / export_report_name).write_text(json.dumps(export_report, indent=2, sort_keys=True) + "\n")

        output_root.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            if path.name == "model-fp32.onnx":
                continue
            os.replace(path, output_root / path.name)
        return export_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=context_model_contract.DEFAULT_SPEC)
    parser.add_argument(
        "--training-report",
        type=pathlib.Path,
        default=DEFAULT_TRAINING_ROOT / "training-report.json",
    )
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        report = export(parse_args(argv))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        ContextExportError,
        context_model_contract.ContextModelContractError,
        context_tokenizer_contract.ContextTokenizerContractError,
        export_swipe_model.SwipeExportError,
    ) as failure:
        print(f"context export error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
