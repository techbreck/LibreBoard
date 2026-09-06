#!/usr/bin/env python3
"""Compose the exact reduced ONNX Runtime operator config from both model exports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
from typing import Any

import model_sources


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SWIPE_REPORT = ROOT / "build" / "model-export" / "swipe-latin-v1" / "export-report.json"
DEFAULT_CONTEXT_REPORT = ROOT / "build" / "model-export" / "context-en-de-v1" / "export-report.json"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-export" / "onnxruntime"
MODEL_LIMITS = {
    "swipe-latin-v1": 3 * 1024 * 1024,
    "context-en-de-v1": 24 * 1024 * 1024,
}
MODEL_IDENTITIES = {
    "swipe-latin-v1": ("swipe-ctc", "swipe-latin-v1"),
    "context-en-de-v1": ("context-rescorer", "context-en-de-v1"),
}
MODEL_DOMAIN_OPSETS = {
    "swipe-latin-v1": {("ai.onnx", 18)},
    "context-en-de-v1": {("ai.onnx", 21), ("com.microsoft", 1)},
}
MAXIMUM_JSON_BYTES = 4 * 1024 * 1024
MAXIMUM_CONFIG_BYTES = 1024 * 1024
OPERATOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DOMAIN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
TYPE_INDEX = re.compile(r"^(0|[1-9][0-9]*)$")
ALLOWED_KERNEL_TYPES = {
    "MLFloat16",
    "BFloat16",
    "bool",
    "double",
    "float",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
}
EXPECTED_ONNXRUNTIME_VERSION = "1.26.0"
ALLOWED_RUNTIME_DOMAINS = {"ai.onnx", "com.microsoft"}


class RuntimeOperatorConfigError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeOperatorConfigError(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _load_json(path: pathlib.Path, label: str) -> tuple[dict[str, Any], str]:
    path = path.absolute()
    try:
        if (
            not path.is_file()
            or path.is_symlink()
            or not 0 < path.stat().st_size <= MAXIMUM_JSON_BYTES
        ):
            raise RuntimeOperatorConfigError(f"{label} is missing, linked, empty, or too large")
        payload = path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except RuntimeOperatorConfigError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise RuntimeOperatorConfigError(f"cannot read {label}: {failure}") from failure
    if not isinstance(value, dict):
        raise RuntimeOperatorConfigError(f"{label} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _artifact(root: pathlib.Path, filename: Any, label: str) -> pathlib.Path:
    if (
        not isinstance(filename, str)
        or pathlib.PurePosixPath(filename).name != filename
        or filename in {"", ".", ".."}
    ):
        raise RuntimeOperatorConfigError(f"{label} has an unsafe filename")
    path = root / filename
    if not path.is_file() or path.is_symlink():
        raise RuntimeOperatorConfigError(f"{label} is missing or linked")
    return path


def _parse_operator_config(path: pathlib.Path) -> dict[tuple[str, int], set[str]]:
    try:
        if not 0 < path.stat().st_size <= MAXIMUM_CONFIG_BYTES:
            raise RuntimeOperatorConfigError("model operator config is empty or too large")
        lines = path.read_text(encoding="utf-8").splitlines()
    except RuntimeOperatorConfigError:
        raise
    except (OSError, UnicodeDecodeError) as failure:
        raise RuntimeOperatorConfigError(f"cannot read model operator config: {failure}") from failure
    if lines[:1] != ["# Generated from the exact LibreBoard model graph; do not edit by hand."]:
        raise RuntimeOperatorConfigError("model operator config has an unexpected provenance header")
    required: dict[tuple[str, int], set[str]] = {}
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split(";")
        if len(parts) != 3 or not DOMAIN.fullmatch(parts[0]):
            raise RuntimeOperatorConfigError(f"invalid reduced-operator line: {line}")
        try:
            opset = int(parts[1])
        except ValueError as failure:
            raise RuntimeOperatorConfigError(f"invalid reduced-operator opset: {line}") from failure
        operators = parts[2].split(",")
        if (
            not 1 <= opset <= 100
            or not operators
            or any(not OPERATOR.fullmatch(operator) for operator in operators)
            or operators != sorted(set(operators))
        ):
            raise RuntimeOperatorConfigError(f"invalid reduced-operator inventory: {line}")
        key = (parts[0], opset)
        if key in required:
            raise RuntimeOperatorConfigError(f"duplicate reduced-operator domain/opset: {line}")
        required[key] = set(operators)
    if not required:
        raise RuntimeOperatorConfigError("model operator config contains no operators")
    return required


def _flatten(required: dict[tuple[str, int], set[str]]) -> set[str]:
    return {
        operator if domain == "ai.onnx" else f"{domain}::{operator}"
        for (domain, _opset), operators in required.items()
        for operator in operators
    }


def _parse_typed_operator_entries(value: str) -> dict[str, dict[str, Any] | None]:
    entries: dict[str, dict[str, Any] | None] = {}
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    position = 0
    while position < len(value):
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", value[position:])
        if match is None:
            raise RuntimeOperatorConfigError(f"invalid typed operator inventory: {value}")
        operator = match.group(0)
        position += len(operator)
        type_info = None
        if position < len(value) and value[position] == "{":
            try:
                decoded, consumed = decoder.raw_decode(value[position:])
            except (json.JSONDecodeError, RuntimeOperatorConfigError) as failure:
                raise RuntimeOperatorConfigError(
                    f"invalid type reduction metadata for {operator}"
                ) from failure
            if not isinstance(decoded, dict) or not decoded or set(decoded) - {"inputs", "outputs"}:
                raise RuntimeOperatorConfigError(f"unsupported type reduction metadata for {operator}")
            for direction, indexes in decoded.items():
                if not isinstance(indexes, dict) or not indexes:
                    raise RuntimeOperatorConfigError(f"invalid {direction} type metadata for {operator}")
                for index, types in indexes.items():
                    if (
                        not isinstance(index, str)
                        or not TYPE_INDEX.fullmatch(index)
                        or not isinstance(types, list)
                        or not types
                        or any(not isinstance(item, str) or item not in ALLOWED_KERNEL_TYPES for item in types)
                        or types != sorted(set(types))
                    ):
                        raise RuntimeOperatorConfigError(f"invalid {direction} type metadata for {operator}")
            type_info = decoded
            position += consumed
        if operator in entries:
            raise RuntimeOperatorConfigError(f"duplicate typed operator: {operator}")
        entries[operator] = type_info
        if position == len(value):
            break
        if value[position] != ",":
            raise RuntimeOperatorConfigError(f"invalid typed operator delimiter: {value}")
        position += 1
    if not entries:
        raise RuntimeOperatorConfigError("typed operator inventory is empty")
    return entries


def _parse_typed_config(path: pathlib.Path) -> dict[tuple[str, int], dict[str, dict[str, Any] | None]]:
    try:
        if not 0 < path.stat().st_size <= MAXIMUM_CONFIG_BYTES:
            raise RuntimeOperatorConfigError("typed operator config is empty or too large")
        lines = path.read_text(encoding="utf-8").splitlines()
    except RuntimeOperatorConfigError:
        raise
    except (OSError, UnicodeDecodeError) as failure:
        raise RuntimeOperatorConfigError(f"cannot read typed operator config: {failure}") from failure
    required: dict[tuple[str, int], dict[str, dict[str, Any] | None]] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split(";", 2)
        if (
            len(parts) != 3
            or not DOMAIN.fullmatch(parts[0])
            or parts[0] not in ALLOWED_RUNTIME_DOMAINS
        ):
            raise RuntimeOperatorConfigError(f"invalid typed reduced-operator line: {line}")
        try:
            opset = int(parts[1])
        except ValueError as failure:
            raise RuntimeOperatorConfigError(f"invalid typed reduced-operator opset: {line}") from failure
        if not 1 <= opset <= 100:
            raise RuntimeOperatorConfigError(f"invalid typed reduced-operator opset: {line}")
        key = (parts[0], opset)
        if key in required:
            raise RuntimeOperatorConfigError(f"duplicate typed reduced-operator domain/opset: {line}")
        required[key] = _parse_typed_operator_entries(parts[2])
    if not required:
        raise RuntimeOperatorConfigError("typed operator config contains no operators")
    return required


def _canonical_typed_config(
    raw: dict[tuple[str, int], set[str]],
    typed: dict[tuple[str, int], dict[str, dict[str, Any] | None]],
) -> tuple[bytes, int]:
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for (domain, _opset), operators in typed.items():
        for operator, annotation in operators.items():
            if annotation is None:
                continue
            key = (domain, operator)
            previous = annotations.get(key)
            if previous is not None and previous != annotation:
                raise RuntimeOperatorConfigError(f"conflicting type metadata for {domain}::{operator}")
            annotations[key] = annotation

    combined = {key: dict(operators) for key, operators in typed.items()}
    for key, operators in raw.items():
        target = combined.setdefault(key, {})
        for operator in operators:
            # Type metadata is valid only for the exact domain/opset entry emitted
            # by ONNX Runtime. Never borrow an annotation from another opset: doing
            # so could remove a kernel type that the original ONNX graph still uses.
            target.setdefault(operator, None)

    lines = [
        "# Generated from the exact LibreBoard ONNX graphs plus deterministic raw/optimized type analysis; do not edit by hand."
    ]
    annotated = 0
    for (domain, opset), operators in sorted(combined.items()):
        rendered = []
        for operator, annotation in sorted(operators.items()):
            if annotation is None:
                rendered.append(operator)
            else:
                annotated += 1
                rendered.append(operator + json.dumps(annotation, sort_keys=True, separators=(",", ":")))
        lines.append(f"{domain};{opset};{','.join(rendered)}")
    return ("\n".join(lines) + "\n").encode("utf-8"), annotated


def _generate_type_reduced_config(
    model_paths: dict[str, pathlib.Path],
    raw: dict[tuple[str, int], set[str]],
) -> tuple[bytes, dict[str, Any]]:
    try:
        runtime_version = importlib.metadata.version("onnxruntime")
        from onnxruntime.tools import convert_onnx_models_to_ort as converter
    except (importlib.metadata.PackageNotFoundError, ImportError) as failure:
        raise RuntimeOperatorConfigError(
            "ONNX Runtime 1.26.0 is required for operator type specialization"
        ) from failure
    if runtime_version != EXPECTED_ONNXRUNTIME_VERSION:
        raise RuntimeOperatorConfigError(
            f"operator type specialization requires ONNX Runtime {EXPECTED_ONNXRUNTIME_VERSION}"
        )

    previous_level = os.environ.get("ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL")
    with tempfile.TemporaryDirectory(prefix="libreboard-runtime-types-") as temporary:
        root = pathlib.Path(temporary)
        inputs = root / "inputs"
        inputs.mkdir()
        for model_id, model_path in sorted(model_paths.items()):
            shutil.copyfile(model_path, inputs / f"{model_id}.onnx")
        converted = []
        try:
            for level in ("disable", "all"):
                output = root / level
                os.environ["ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL"] = level
                converter.convert_onnx_models_to_ort(
                    inputs,
                    output_dir=output,
                    optimization_styles=[converter.OptimizationStyle.Fixed],
                    target_platform="arm",
                    enable_type_reduction=True,
                )
                converted.extend(sorted(output.rglob("*.ort")))
        finally:
            if previous_level is None:
                os.environ.pop("ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL", None)
            else:
                os.environ["ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL"] = previous_level
        if len(converted) != len(model_paths) * 2:
            raise RuntimeOperatorConfigError("operator type specialization did not convert every model twice")
        generated = root / "required_operators_and_types.config"
        converter.create_config_from_models(converted, generated, enable_type_reduction=True)
        typed = _parse_typed_config(generated)
        config, annotated = _canonical_typed_config(raw, typed)

    return config, {
        "enabled": True,
        "onnxRuntimeVersion": runtime_version,
        "optimizationLevels": ["disable", "all"],
        "targetPlatform": "arm",
        "convertedModelCount": len(converted),
        "annotatedOperatorEntries": annotated,
    }


def _load_export(
    report_path: pathlib.Path,
    *,
    expected_model_id: str,
    development: bool,
) -> tuple[dict[tuple[str, int], set[str]], dict[str, Any], pathlib.Path]:
    report_path = report_path.absolute()
    report, report_sha = _load_json(report_path, f"{expected_model_id} export report")
    if (
        report.get("schemaVersion") != 1
        or report.get("modelId") != expected_model_id
        or not isinstance(report.get("releaseEligible"), bool)
        or (not development and report["releaseEligible"] is not True)
    ):
        raise RuntimeOperatorConfigError(f"{expected_model_id} export is not release eligible")
    app_commit = report.get("appCommit")
    if not isinstance(app_commit, str) or not model_sources.REVISION.fullmatch(app_commit):
        raise RuntimeOperatorConfigError(f"{expected_model_id} export has an invalid app commit")
    root = report_path.parent
    model_details = report.get("model")
    if not isinstance(model_details, dict) or set(model_details) != {"file", "bytes", "sha256"}:
        raise RuntimeOperatorConfigError(f"{expected_model_id} export has invalid model metadata")
    model_path = _artifact(root, model_details["file"], f"{expected_model_id} model")
    if (
        isinstance(model_details["bytes"], bool)
        or not isinstance(model_details["bytes"], int)
        or not 0 < model_details["bytes"] <= MODEL_LIMITS[expected_model_id]
        or model_path.stat().st_size != model_details["bytes"]
        or not isinstance(model_details["sha256"], str)
        or not model_sources.SHA256.fullmatch(model_details["sha256"])
        or model_sources.file_sha256(model_path) != model_details["sha256"]
    ):
        raise RuntimeOperatorConfigError(f"{expected_model_id} model does not match its export report")

    manifest_path = _artifact(root, report.get("manifest"), f"{expected_model_id} manifest")
    manifest, manifest_sha = _load_json(manifest_path, f"{expected_model_id} manifest")
    expected_kind, expected_tensor_abi = MODEL_IDENTITIES[expected_model_id]
    operators = manifest.get("requiredOnnxOperators")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("engineAbi") != 1
        or manifest.get("modelKind") != expected_kind
        or manifest.get("tensorAbi") != expected_tensor_abi
        or manifest.get("modelSha256") != model_details["sha256"]
        or not isinstance(operators, list)
        or not operators
        or any(
            not isinstance(operator, str)
            or (
                not OPERATOR.fullmatch(operator)
                and not (
                    operator.count("::") == 1
                    and DOMAIN.fullmatch(operator.split("::", 1)[0])
                    and OPERATOR.fullmatch(operator.split("::", 1)[1])
                )
            )
            for operator in operators
        )
        or operators != sorted(set(operators))
    ):
        raise RuntimeOperatorConfigError(f"{expected_model_id} manifest has an invalid runtime contract")
    config_path = _artifact(root, report.get("requiredOperators"), f"{expected_model_id} operator config")
    required = _parse_operator_config(config_path)
    if set(required) != MODEL_DOMAIN_OPSETS[expected_model_id]:
        raise RuntimeOperatorConfigError(f"{expected_model_id} operator config has unexpected domains or opsets")
    if _flatten(required) != set(operators):
        raise RuntimeOperatorConfigError(
            f"{expected_model_id} operator config does not match its signed manifest"
        )
    return (
        required,
        {
            "appCommit": app_commit,
            "exportReportSha256": report_sha,
            "manifestSha256": manifest_sha,
            "modelSha256": model_details["sha256"],
            "operatorsSha256": model_sources.file_sha256(config_path),
        },
        model_path,
    )


def _canonical_config(required: dict[tuple[str, int], set[str]]) -> bytes:
    lines = [
        "# Generated from the exact LibreBoard swipe-latin-v1 and context-en-de-v1 model graphs; do not edit by hand."
    ]
    for (domain, opset), operators in sorted(required.items()):
        lines.append(f"{domain};{opset};{','.join(sorted(operators))}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    try:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary_path, path)
    finally:
        if not temporary.closed:
            temporary.close()
        temporary_path.unlink(missing_ok=True)


def assemble(args: argparse.Namespace) -> dict[str, Any]:
    type_reduction_enabled = bool(getattr(args, "type_reduction", False))
    if not args.development and not type_reduction_enabled:
        raise RuntimeOperatorConfigError("release operator configuration requires type reduction")
    combined: dict[tuple[str, int], set[str]] = {}
    inputs = {}
    model_paths = {}
    for model_id, report_path in (
        ("swipe-latin-v1", args.swipe_export_report),
        ("context-en-de-v1", args.context_export_report),
    ):
        required, details, model_path = _load_export(
            report_path,
            expected_model_id=model_id,
            development=args.development,
        )
        inputs[model_id] = details
        model_paths[model_id] = model_path
        for key, operators in required.items():
            combined.setdefault(key, set()).update(operators)
    if type_reduction_enabled:
        config, type_reduction = _generate_type_reduced_config(model_paths, combined)
    else:
        config = _canonical_config(combined)
        type_reduction = {"enabled": False}
    config_name = "required_operators-development.config" if args.development else "required_operators.config"
    report_name = "runtime-operators-development.json" if args.development else "runtime-operators.json"
    output_root = args.output_root.absolute()
    config_path = output_root / config_name
    report_path = output_root / report_name
    report = {
        "schemaVersion": 1,
        "releaseEligible": not args.development,
        "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
        "inputs": inputs,
        "typeReduction": type_reduction,
        "operators": {
            "file": config_name,
            "bytes": len(config),
            "sha256": hashlib.sha256(config).hexdigest(),
            "domainOpsets": len(combined),
            "uniqueOperators": len(_flatten(combined)),
        },
    }
    _atomic_write(config_path, config)
    _atomic_write(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swipe-export-report", type=pathlib.Path, default=DEFAULT_SWIPE_REPORT)
    parser.add_argument("--context-export-report", type=pathlib.Path, default=DEFAULT_CONTEXT_REPORT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--type-reduction", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        print(json.dumps(assemble(parse_args(argv)), indent=2, sort_keys=True))
        return 0
    except (RuntimeOperatorConfigError, model_sources.ModelSourceError, OSError) as failure:
        print(f"runtime operator config error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
