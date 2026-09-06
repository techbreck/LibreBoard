#!/usr/bin/env python3
"""Export, inspect, and package a trained LibreBoard CTC model as deterministic ONNX."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Any

import model_sources
import swipe_model_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_ROOT = ROOT / "build" / "model-training" / "swipe-latin-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-export" / "swipe-latin-v1"


class SwipeExportError(ValueError):
    pass


def _dependencies():
    try:
        import numpy
        import onnx
        import safetensors
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as failure:
        raise SwipeExportError(
            "install the pinned build-only dependencies from models/training/requirements-linux-x86_64.lock"
        ) from failure
    versions = {
        "torch": str(torch.__version__).split("+", 1)[0],
        "numpy": numpy.__version__,
        "onnx": onnx.__version__,
        "safetensors": safetensors.__version__,
    }
    expected = {
        "torch": "2.8.0",
        "numpy": "2.2.6",
        "onnx": "1.19.0",
        "safetensors": "0.6.2",
    }
    if versions != expected:
        raise SwipeExportError(f"export dependency versions do not match the lock: {versions}")
    return torch, numpy, onnx, safe_open, load_file, versions


def _load_training_report(path: pathlib.Path, development: bool) -> dict[str, Any]:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise SwipeExportError("training report is missing, linked, or too large")
        report = json.loads(path.read_bytes())
    except SwipeExportError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise SwipeExportError(f"cannot read training report: {failure}") from failure
    if not isinstance(report, dict) or report.get("schemaVersion") != 1 or report.get("modelId") != "swipe-latin-v1":
        raise SwipeExportError("training report has an unsupported schema or model")
    if report.get("releaseEligible") is not True and not development:
        raise SwipeExportError("development or partial training cannot produce a release model")
    weights = report.get("weights")
    if not isinstance(weights, dict) or not isinstance(weights.get("file"), str):
        raise SwipeExportError("training report has no weight artifact")
    if not isinstance(weights.get("sha256"), str) or not model_sources.SHA256.fullmatch(weights["sha256"]):
        raise SwipeExportError("training report has an invalid weight hash")
    if isinstance(weights.get("bytes"), bool) or not isinstance(weights.get("bytes"), int) or weights["bytes"] <= 0:
        raise SwipeExportError("training report has an invalid weight size")
    return report


def _tensor_shape(value_info) -> tuple[int, ...]:
    dimensions = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise SwipeExportError(f"ONNX tensor {value_info.name} has a dynamic dimension")
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def _validate_tensor_abi(model, spec: swipe_model_contract.SwipeModelSpec, onnx) -> None:
    type_names = {"float32": onnx.TensorProto.FLOAT}
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
        raise SwipeExportError(f"ONNX inputs violate the fixed tensor ABI: {actual_inputs}")
    if actual_outputs != expected_outputs:
        raise SwipeExportError(f"ONNX outputs violate the fixed tensor ABI: {actual_outputs}")


def _all_graphs(graph, onnx):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _all_graphs(attribute.g, onnx)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from _all_graphs(nested, onnx)


def _reject_external_data(model, onnx) -> None:
    def check_tensor(tensor) -> None:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise SwipeExportError("ONNX external tensor data is forbidden")

    for graph in _all_graphs(model.graph, onnx):
        for tensor in graph.initializer:
            check_tensor(tensor)
        for sparse in graph.sparse_initializer:
            check_tensor(sparse.values)
            check_tensor(sparse.indices)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    check_tensor(attribute.t)
                for tensor in attribute.tensors:
                    check_tensor(tensor)
                if attribute.HasField("sparse_tensor"):
                    check_tensor(attribute.sparse_tensor.values)
                    check_tensor(attribute.sparse_tensor.indices)
                for sparse in attribute.sparse_tensors:
                    check_tensor(sparse.values)
                    check_tensor(sparse.indices)


def required_operators(model, onnx) -> dict[tuple[str, int], set[str]]:
    opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
    if not opsets:
        raise SwipeExportError("ONNX model imports no operator set")
    required: dict[tuple[str, int], set[str]] = defaultdict(set)

    def add_node(node, active_opsets: dict[str, int]) -> None:
        domain = node.domain or "ai.onnx"
        version = active_opsets.get(domain)
        if version is None:
            raise SwipeExportError(f"ONNX node uses an unimported domain: {domain}")
        if not node.op_type or any(character.isspace() for character in node.op_type):
            raise SwipeExportError("ONNX node has an invalid operator name")
        required[(domain, version)].add(node.op_type)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                for nested_node in attribute.g.node:
                    add_node(nested_node, active_opsets)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    for nested_node in graph.node:
                        add_node(nested_node, active_opsets)

    for node in model.graph.node:
        add_node(node, opsets)
    for function in model.functions:
        function_opsets = dict(opsets)
        function_opsets.update({entry.domain or "ai.onnx": entry.version for entry in function.opset_import})
        for node in function.node:
            add_node(node, function_opsets)
    if not required or not any(required.values()):
        raise SwipeExportError("ONNX model contains no operators")
    return dict(required)


def _canonical_operator_list(required: dict[tuple[str, int], set[str]]) -> list[str]:
    values = set()
    for (domain, _version), operators in required.items():
        for operator in operators:
            values.add(operator if domain == "ai.onnx" else f"{domain}::{operator}")
    return sorted(values)


def _operator_config(required: dict[tuple[str, int], set[str]]) -> str:
    lines = ["# Generated from the exact LibreBoard model graph; do not edit by hand."]
    for (domain, version), operators in sorted(required.items()):
        if operators:
            lines.append(f"{domain};{version};{','.join(sorted(operators))}")
    return "\n".join(lines) + "\n"


def _strip_nonsemantic_metadata(model) -> None:
    model.producer_name = "LibreBoard"
    model.producer_version = "1"
    model.domain = "org.libreboard.keyboard"
    model.model_version = 1
    model.doc_string = ""
    del model.metadata_props[:]
    for graph in [model.graph]:
        graph.doc_string = ""
        for node in graph.node:
            node.doc_string = ""


def _convert_float_initializers_to_fp16_storage(model, numpy, onnx) -> int:
    """Store learned floats as FP16 while preserving float32 compute and external tensor types.

    Generic whole-graph converters can change PyTorch attention Cast annotations inconsistently.
    LibreBoard instead performs the smaller, explicit transform it needs: each float32 initializer
    is stored as float16 and one graph-local Cast restores float32 before use. ORT can fold these
    casts when it creates the session, the APK stores half-sized weights, and the graph remains
    fully type-checkable on the CPU execution provider.
    """

    converted = 0

    def process_graph(graph, inherited: dict[str, str]) -> None:
        nonlocal converted
        mapping = dict(inherited)
        existing_names = {
            name
            for node in graph.node
            for name in (*node.input, *node.output)
            if name
        }
        cast_nodes = []
        for initializer in graph.initializer:
            if initializer.data_type != onnx.TensorProto.FLOAT:
                continue
            array = onnx.numpy_helper.to_array(initializer)
            replacement = onnx.numpy_helper.from_array(array.astype(numpy.float16), initializer.name)
            initializer.CopyFrom(replacement)
            base = initializer.name + "__libreboard_fp32"
            output_name = base
            suffix = 1
            while output_name in existing_names:
                output_name = f"{base}_{suffix}"
                suffix += 1
            existing_names.add(output_name)
            mapping[initializer.name] = output_name
            cast_nodes.append(onnx.helper.make_node(
                "Cast",
                inputs=[initializer.name],
                outputs=[output_name],
                name=f"LibreBoardFp16StorageCast_{converted}",
                to=onnx.TensorProto.FLOAT,
            ))
            converted += 1
        original_nodes = list(graph.node)
        for node in original_nodes:
            for index, input_name in enumerate(node.input):
                if input_name in mapping:
                    node.input[index] = mapping[input_name]
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    process_graph(attribute.g, mapping)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        process_graph(nested, mapping)
        if cast_nodes:
            del graph.node[:]
            graph.node.extend(cast_nodes)
            graph.node.extend(original_nodes)

    process_graph(model.graph, {})
    if converted == 0:
        raise SwipeExportError("ONNX graph contains no float32 initializers to store as FP16")
    return converted


def export(args: argparse.Namespace) -> dict[str, Any]:
    torch, numpy, onnx, safe_open, load_file, versions = _dependencies()
    spec = swipe_model_contract.load_spec(args.spec)
    training_report_path = args.training_report.resolve()
    training_report = _load_training_report(training_report_path, args.development)
    try:
        model_sources.verify_git_sources_at_commit(
            training_report["appCommit"],
            ("tools/train_swipe_model.py", "models/training/swipe_model.py"),
        )
    except model_sources.ModelSourceError as failure:
        raise SwipeExportError(f"swipe model source provenance failed: {failure}") from failure
    if training_report.get("modelSpecSha256") != spec.sha256:
        raise SwipeExportError("training report was produced from a different model spec")
    weights_path = training_report_path.parent / training_report["weights"]["file"]
    if not weights_path.is_file() or weights_path.is_symlink():
        raise SwipeExportError("training weight artifact is missing")
    if weights_path.stat().st_size != training_report["weights"]["bytes"]:
        raise SwipeExportError("training weight size does not match its report")
    if model_sources.file_sha256(weights_path) != training_report["weights"]["sha256"]:
        raise SwipeExportError("training weight hash does not match its report")
    split_manifest_path = training_report_path.parent / "split-manifest.json"
    if (
        not split_manifest_path.is_file()
        or split_manifest_path.is_symlink()
        or model_sources.file_sha256(split_manifest_path) != training_report["dataManifestSha256"]
    ):
        raise SwipeExportError("prepared split manifest is missing or does not match the training report")
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
    expected_metadata = {
        "appCommit": training_report["appCommit"],
        "dataManifestSha256": training_report["dataManifestSha256"],
        "modelSpecSha256": training_report["modelSpecSha256"],
    }
    if metadata != expected_metadata:
        raise SwipeExportError("safetensors metadata does not match the training report")

    sys.path.insert(0, str(ROOT))
    from models.training.swipe_model import SwipeCtcModel, trainable_parameter_count

    model = SwipeCtcModel(spec.architecture).eval()
    model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
    if trainable_parameter_count(model) != spec.raw["parameterCount"]:
        raise SwipeExportError("loaded model parameter count violates the spec")
    torch.manual_seed(spec.training["seed"])
    dummy_path = torch.zeros((1, 64, 2), dtype=torch.float32)
    dummy_centers = torch.zeros((1, 64, 2), dtype=torch.float32)
    dummy_mask = torch.cat((torch.ones((1, 26)), torch.zeros((1, 38))), dim=1)

    output_root = args.output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libreboard-swipe-export-", dir=output_root.parent) as temporary:
        staging = pathlib.Path(temporary)
        fp32_path = staging / "model-fp32.onnx"
        torch.onnx.export(
            model,
            (dummy_path, dummy_centers, dummy_mask),
            fp32_path,
            input_names=[tensor["name"] for tensor in spec.export["inputs"]],
            output_names=[tensor["name"] for tensor in spec.export["outputs"]],
            opset_version=spec.export["opsetVersion"],
            do_constant_folding=True,
            dynamo=False,
        )
        exported = onnx.load_model(fp32_path, load_external_data=False)
        onnx.checker.check_model(exported, full_check=True)
        converted = exported
        converted_initializer_count = _convert_float_initializers_to_fp16_storage(converted, numpy, onnx)
        _strip_nonsemantic_metadata(converted)
        _reject_external_data(converted, onnx)
        _validate_tensor_abi(converted, spec, onnx)
        onnx.checker.check_model(converted, full_check=True)

        development = args.development or training_report.get("releaseEligible") is not True
        model_filename = "swipe-latin-v1-development.onnx" if development else "swipe-latin-v1.onnx"
        model_path = staging / model_filename
        model_path.write_bytes(converted.SerializeToString(deterministic=True))
        if model_path.stat().st_size > spec.export["maximumModelBytes"]:
            raise SwipeExportError(
                f"FP16 ONNX model is {model_path.stat().st_size} bytes; limit is {spec.export['maximumModelBytes']}"
            )
        verified = onnx.load_model(model_path, load_external_data=False)
        onnx.checker.check_model(verified, full_check=True)
        _reject_external_data(verified, onnx)
        _validate_tensor_abi(verified, spec, onnx)
        required = required_operators(verified, onnx)
        canonical_operators = _canonical_operator_list(required)
        if any("::" in operator for operator in canonical_operators):
            raise SwipeExportError("swipe-latin-v1 may use only standard ai.onnx operators")

        model_hash = model_sources.file_sha256(model_path)
        manifest = {
            "schemaVersion": 1,
            "engineAbi": spec.raw["engineAbi"],
            "modelKind": "swipe-ctc",
            "tensorAbi": spec.raw["tensorAbi"],
            "locales": spec.raw["locales"],
            "architecture": spec.raw["architectureName"],
            "parameterCount": spec.raw["parameterCount"],
            "quantization": spec.export["quantization"],
            "modelSha256": model_hash,
            "requiredOnnxOperators": canonical_operators,
            "license": spec.raw["license"],
            "provenance": [{
                "name": "futo-org/swipe.futo.org dataset",
                "revision": "d71bf5fd7f45b3e7c2ed2d76a21b0dbd3b4ba566",
                "license": "MIT",
                "source_url": "https://huggingface.co/datasets/futo-org/swipe.futo.org",
            }],
            "minimumAppVersionCode": 1,
        }
        manifest_name = "manifest-development.json" if development else "manifest.json"
        (staging / manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ops_name = "required_operators-development.config" if development else "required_operators.config"
        (staging / ops_name).write_text(_operator_config(required), encoding="utf-8")
        shutil.copyfile(ROOT / "LICENSE-Apache-2.0", staging / "LICENSE")
        shutil.copyfile(ROOT / "models" / "swipe" / "MODEL_CARD.md", staging / "MODEL_CARD.md")
        shutil.copyfile(spec.path, staging / "model.json")
        shutil.copyfile(split_manifest_path, staging / "split-manifest.json")
        shutil.copyfile(training_report_path, staging / training_report_path.name)

        report = {
            "schemaVersion": 1,
            "modelId": spec.raw["modelId"],
            "releaseEligible": not development,
            "appCommit": training_report["appCommit"],
            "modelSpecSha256": spec.sha256,
            "dataManifestSha256": training_report["dataManifestSha256"],
            "trainingReportSha256": model_sources.file_sha256(training_report_path),
            "weightsSha256": training_report["weights"]["sha256"],
            "model": {
                "file": model_filename,
                "bytes": model_path.stat().st_size,
                "sha256": model_hash,
            },
            "manifest": manifest_name,
            "requiredOperators": ops_name,
            "modelSpec": "model.json",
            "splitManifest": "split-manifest.json",
            "trainingReport": training_report_path.name,
            "fp16StoredInitializerCount": converted_initializer_count,
            "toolchain": versions,
        }
        report_name = "export-report-development.json" if development else "export-report.json"
        (staging / report_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        output_root.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            if path.name == "model-fp32.onnx":
                continue
            os.replace(path, output_root / path.name)
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=swipe_model_contract.DEFAULT_SPEC)
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
    except (SwipeExportError, swipe_model_contract.SwipeModelContractError) as failure:
        print(f"swipe export error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
