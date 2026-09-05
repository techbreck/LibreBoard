#!/usr/bin/env python3
"""Dependency-free validation shared by swipe training and export commands."""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import model_sources


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "models" / "swipe" / "model-spec.json"
MAXIMUM_SPEC_BYTES = 256 * 1024
SPEC_KEYS = {
    "schemaVersion",
    "modelId",
    "engineAbi",
    "tensorAbi",
    "architectureName",
    "license",
    "locales",
    "parameterCount",
    "architecture",
    "training",
    "export",
}
ARCHITECTURE_KEYS = {
    "pathPoints",
    "keySlots",
    "outputFrames",
    "width",
    "heads",
    "layers",
    "feedForwardWidth",
    "dropout",
    "layoutTemperature",
}
TRAINING_KEYS = {
    "seed",
    "epochs",
    "batchSize",
    "learningRate",
    "weightDecay",
    "gradientClip",
    "shuffleBuffer",
    "pathNoiseStd",
    "geometryScaleStd",
    "geometryTranslationStd",
}
EXPORT_KEYS = {"opsetVersion", "quantization", "maximumModelBytes", "inputs", "outputs"}
TENSOR_KEYS = {"name", "elementType", "shape"}
EXPECTED_INPUTS = {
    "path_coordinates": ("float32", (1, 64, 2)),
    "key_centers": ("float32", (1, 64, 2)),
    "key_mask": ("float32", (1, 64)),
}
EXPECTED_OUTPUTS = {"logits": ("float32", (1, 32, 65))}


class SwipeModelContractError(ValueError):
    pass


@dataclass(frozen=True)
class SwipeModelSpec:
    path: pathlib.Path
    sha256: str
    raw: dict[str, Any]

    @property
    def architecture(self) -> dict[str, Any]:
        return self.raw["architecture"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def export(self) -> dict[str, Any]:
        return self.raw["export"]


def expected_parameter_count(architecture: dict[str, Any]) -> int:
    width = architecture["width"]
    ffn = architecture["feedForwardWidth"]
    layers = architecture["layers"]
    path_projection = 5 * width + width
    path_position = architecture["pathPoints"] * width
    key_encoder = 2 * width + width + width * width + width
    attention = 3 * width * width + 3 * width + width * width + width
    feed_forward = width * ffn + ffn + ffn * width + width
    layer_norms = 4 * width
    transformer = layers * (attention + feed_forward + layer_norms)
    output_norm = 2 * width
    blank_head = width + 1
    return path_projection + path_position + key_encoder + transformer + output_norm + blank_head


def _strict_object(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SwipeModelContractError(f"{location} has an unexpected schema")
    return value


def _positive_int(value: Any, location: str, maximum: int = 100_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise SwipeModelContractError(f"{location} must be a positive bounded integer")
    return value


def _finite_float(value: Any, location: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SwipeModelContractError(f"{location} must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise SwipeModelContractError(f"{location} is outside allowed bounds")
    return number


def _validate_tensors(values: Any, expected: dict[str, tuple[str, tuple[int, ...]]], location: str) -> None:
    if not isinstance(values, list) or len(values) != len(expected):
        raise SwipeModelContractError(f"{location} tensor list is incomplete")
    seen = set()
    for index, value in enumerate(values):
        tensor = _strict_object(value, TENSOR_KEYS, f"{location} tensor {index + 1}")
        name = tensor["name"]
        shape = tensor["shape"]
        if name not in expected or name in seen:
            raise SwipeModelContractError(f"{location} has an invalid tensor name")
        if tensor["elementType"] != expected[name][0] or shape != list(expected[name][1]):
            raise SwipeModelContractError(f"{location} tensor {name} violates the fixed ABI")
        seen.add(name)


def load_spec(path: pathlib.Path = DEFAULT_SPEC) -> SwipeModelSpec:
    path = path.resolve()
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAXIMUM_SPEC_BYTES:
            raise SwipeModelContractError("swipe model spec is missing, linked, or too large")
        payload = path.read_bytes()
        raw = json.loads(payload)
    except SwipeModelContractError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise SwipeModelContractError(f"cannot read swipe model spec: {failure}") from failure
    _strict_object(raw, SPEC_KEYS, "swipe model spec")
    if raw["schemaVersion"] != 1 or raw["modelId"] != "swipe-latin-v1":
        raise SwipeModelContractError("unsupported swipe model identity")
    if raw["engineAbi"] != 1 or raw["tensorAbi"] != "swipe-latin-v1":
        raise SwipeModelContractError("swipe model uses an incompatible engine or tensor ABI")
    if raw["architectureName"] != "LayoutConditionedTransformerCTC-v1":
        raise SwipeModelContractError("swipe model architecture name is not recognized")
    if raw["license"] != "Apache-2.0" or raw["locales"] != ["en-US", "de"]:
        raise SwipeModelContractError("swipe model license/locales violate the v1 contract")

    architecture = _strict_object(raw["architecture"], ARCHITECTURE_KEYS, "swipe architecture")
    expected_architecture = {
        "pathPoints": 64,
        "keySlots": 64,
        "outputFrames": 32,
        "width": 128,
        "heads": 4,
        "layers": 6,
        "feedForwardWidth": 256,
    }
    for field, expected in expected_architecture.items():
        if architecture.get(field) != expected:
            raise SwipeModelContractError(f"swipe architecture {field} must be {expected}")
    _finite_float(architecture["dropout"], "swipe dropout", 0, 0.5)
    _finite_float(architecture["layoutTemperature"], "layout temperature", 1, 256)
    declared_parameters = _positive_int(raw["parameterCount"], "parameterCount")
    calculated_parameters = expected_parameter_count(architecture)
    if declared_parameters != calculated_parameters:
        raise SwipeModelContractError(
            f"parameterCount {declared_parameters} does not match architecture {calculated_parameters}"
        )

    training = _strict_object(raw["training"], TRAINING_KEYS, "swipe training config")
    for field in ("seed", "epochs", "batchSize", "shuffleBuffer"):
        _positive_int(training[field], f"training {field}")
    for field, lower, upper in (
        ("learningRate", 1e-8, 1.0),
        ("weightDecay", 0.0, 1.0),
        ("gradientClip", 0.01, 100.0),
        ("pathNoiseStd", 0.0, 0.25),
        ("geometryScaleStd", 0.0, 0.25),
        ("geometryTranslationStd", 0.0, 0.25),
    ):
        _finite_float(training[field], f"training {field}", lower, upper)

    export = _strict_object(raw["export"], EXPORT_KEYS, "swipe export config")
    if export["opsetVersion"] != 18 or export["quantization"] != "FP16":
        raise SwipeModelContractError("swipe export must use opset 18 and FP16")
    maximum_bytes = _positive_int(export["maximumModelBytes"], "maximumModelBytes")
    if maximum_bytes > 3 * 1024 * 1024:
        raise SwipeModelContractError("swipe model exceeds the core-APK size policy")
    _validate_tensors(export["inputs"], EXPECTED_INPUTS, "swipe inputs")
    _validate_tensors(export["outputs"], EXPECTED_OUTPUTS, "swipe outputs")
    return SwipeModelSpec(path=path, sha256=hashlib.sha256(payload).hexdigest(), raw=raw)


def load_prepared_manifest(data_root: pathlib.Path) -> dict[str, Any]:
    manifest_path = data_root.resolve() / "split-manifest.json"
    try:
        if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > 1024 * 1024:
            raise SwipeModelContractError("prepared swipe split manifest is missing or too large")
        manifest = json.loads(manifest_path.read_bytes())
    except SwipeModelContractError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise SwipeModelContractError(f"cannot read prepared swipe split manifest: {failure}") from failure
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise SwipeModelContractError("prepared swipe split manifest has an unsupported schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise SwipeModelContractError("prepared swipe split manifest has no outputs")
    required = {"train.jsonl", "validation.jsonl", "test.jsonl", "layout.json"}
    if set(outputs) != required:
        raise SwipeModelContractError("prepared swipe outputs are incomplete or unexpected")
    for filename in sorted(required):
        details = outputs[filename]
        if not isinstance(details, dict):
            raise SwipeModelContractError(f"prepared output metadata is invalid: {filename}")
        path = data_root / filename
        if not path.is_file() or path.is_symlink():
            raise SwipeModelContractError(f"prepared output is missing: {filename}")
        size = details.get("bytes")
        digest = details.get("sha256")
        if path.stat().st_size != size or not isinstance(digest, str) or not model_sources.SHA256.fullmatch(digest):
            raise SwipeModelContractError(f"prepared output metadata does not match: {filename}")
        if model_sources.file_sha256(path) != digest:
            raise SwipeModelContractError(f"prepared output hash does not match: {filename}")
        if filename.endswith(".jsonl"):
            records = details.get("records")
            sessions = details.get("sessions")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (records, sessions)):
                raise SwipeModelContractError(f"prepared output counts are invalid: {filename}")
    return manifest
