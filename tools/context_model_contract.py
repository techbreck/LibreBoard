#!/usr/bin/env python3
"""Dependency-free validation for the context-en-de-v1 student model."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "models" / "context" / "model-spec.json"
MAXIMUM_SPEC_BYTES = 256 * 1024
SPEC_KEYS = {
    "schemaVersion", "modelId", "engineAbi", "tensorAbi", "architectureName",
    "license", "locales", "parameterCount", "architecture", "training", "export",
}
ARCHITECTURE_KEYS = {
    "sequenceLength", "maximumCandidates", "maximumCandidateTokens", "vocabularySize",
    "fieldClasses", "width", "heads", "layers", "feedForwardWidth", "dropout",
    "rmsNormEpsilon", "ropeTheta", "scoring",
}
TRAINING_KEYS = {
    "seed", "epochs", "batchSize", "learningRate", "weightDecay", "gradientClip",
    "teacherTemperature", "rankingMargin", "teacherLossWeight", "observedLossWeight",
    "rankingLossWeight", "shuffleBufferRecords", "checkpointEveryExamples",
}
EXPORT_KEYS = {
    "sourceOpsetVersion", "opsetVersion", "quantization", "quantizationBlockSize", "maximumModelBytes",
    "inputs", "outputs",
}
TENSOR_KEYS = {"name", "elementType", "shape"}
EXPECTED_INPUTS = {
    "input_ids": ("int64", (-1, 32)),
    "attention_mask": ("int64", (-1, 32)),
    "candidate_mask": ("float32", (-1, 32)),
    "field_class": ("int64", (-1,)),
}
EXPECTED_OUTPUTS = {"candidate_log_likelihood": ("float32", (-1,))}


class ContextModelContractError(ValueError):
    pass


@dataclass(frozen=True)
class ContextModelSpec:
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
    vocabulary = architecture["vocabularySize"]
    fields = architecture["fieldClasses"]
    layers = architecture["layers"]
    feed_forward = architecture["feedForwardWidth"]
    embeddings = vocabulary * width + fields * width
    per_layer = (
        4 * width * width
        + 3 * width * feed_forward
        + 2 * width
    )
    return embeddings + layers * per_layer + width


def _strict_object(value: Any, keys: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContextModelContractError(f"{location} has an unexpected schema")
    return value


def _positive_int(value: Any, location: str, maximum: int = 100_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise ContextModelContractError(f"{location} must be a positive bounded integer")
    return value


def _finite_float(value: Any, location: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContextModelContractError(f"{location} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ContextModelContractError(f"{location} is outside allowed bounds")
    return number


def _validate_tensors(values: Any, expected: dict[str, tuple[str, tuple[int, ...]]], location: str) -> None:
    if not isinstance(values, list) or len(values) != len(expected):
        raise ContextModelContractError(f"{location} tensor list is incomplete")
    seen = set()
    for index, value in enumerate(values):
        tensor = _strict_object(value, TENSOR_KEYS, f"{location} tensor {index + 1}")
        name = tensor["name"]
        if name not in expected or name in seen:
            raise ContextModelContractError(f"{location} has an invalid tensor name")
        if tensor["elementType"] != expected[name][0] or tensor["shape"] != list(expected[name][1]):
            raise ContextModelContractError(f"{location} tensor {name} violates the fixed ABI")
        seen.add(name)


def load_spec(path: pathlib.Path = DEFAULT_SPEC) -> ContextModelSpec:
    path = path.resolve()
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAXIMUM_SPEC_BYTES:
            raise ContextModelContractError("context model spec is missing, linked, or too large")
        payload = path.read_bytes()
        raw = json.loads(payload)
    except ContextModelContractError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise ContextModelContractError(f"cannot read context model spec: {failure}") from failure
    _strict_object(raw, SPEC_KEYS, "context model spec")
    if raw["schemaVersion"] != 1 or raw["modelId"] != "context-en-de-v1":
        raise ContextModelContractError("unsupported context model identity")
    if raw["engineAbi"] != 1 or raw["tensorAbi"] != "context-en-de-v1":
        raise ContextModelContractError("context model uses an incompatible engine or tensor ABI")
    if raw["architectureName"] != "BilingualCandidateTransformer-v1":
        raise ContextModelContractError("context model architecture name is not recognized")
    if raw["license"] != "Apache-2.0" or raw["locales"] != ["en-US", "de"]:
        raise ContextModelContractError("context model license/locales violate the v1 contract")

    architecture = _strict_object(raw["architecture"], ARCHITECTURE_KEYS, "context architecture")
    expected_architecture = {
        "sequenceLength": 32,
        "maximumCandidates": 32,
        "maximumCandidateTokens": 8,
        "vocabularySize": 16_384,
        "fieldClasses": 5,
        "scoring": "shifted-tied-embedding-mean-v1",
    }
    for field, expected in expected_architecture.items():
        if architecture.get(field) != expected:
            raise ContextModelContractError(f"context architecture {field} must be {expected}")
    profile = tuple(architecture[field] for field in ("width", "heads", "layers", "feedForwardWidth"))
    if profile not in ((512, 8, 8, 1536), (256, 4, 4, 768)):
        raise ContextModelContractError("context architecture must match an approved candidate profile")
    if architecture["width"] % architecture["heads"] != 0:
        raise ContextModelContractError("context attention width must divide evenly across heads")
    _finite_float(architecture["dropout"], "context dropout", 0, 0.5)
    _finite_float(architecture["rmsNormEpsilon"], "context RMS epsilon", 1e-9, 1e-2)
    _finite_float(architecture["ropeTheta"], "context RoPE theta", 100, 1_000_000)
    declared_parameters = _positive_int(raw["parameterCount"], "parameterCount")
    calculated_parameters = expected_parameter_count(architecture)
    if declared_parameters != calculated_parameters:
        raise ContextModelContractError(
            f"parameterCount {declared_parameters} does not match architecture {calculated_parameters}"
        )

    training = _strict_object(raw["training"], TRAINING_KEYS, "context training config")
    for field in ("seed", "epochs", "batchSize", "shuffleBufferRecords", "checkpointEveryExamples"):
        _positive_int(training[field], f"training {field}")
    for field, lower, upper in (
        ("learningRate", 1e-8, 1.0),
        ("weightDecay", 0.0, 1.0),
        ("gradientClip", 0.01, 100.0),
        ("teacherTemperature", 0.01, 100.0),
        ("rankingMargin", 0.0, 100.0),
        ("teacherLossWeight", 0.0, 100.0),
        ("observedLossWeight", 0.0, 100.0),
        ("rankingLossWeight", 0.0, 100.0),
    ):
        _finite_float(training[field], f"training {field}", lower, upper)
    if sum(training[field] for field in (
        "teacherLossWeight", "observedLossWeight", "rankingLossWeight",
    )) <= 0:
        raise ContextModelContractError("context training loss weights cannot all be zero")

    export = _strict_object(raw["export"], EXPORT_KEYS, "context export config")
    if export["sourceOpsetVersion"] != 18 or export["opsetVersion"] != 21:
        raise ContextModelContractError("context export must convert an opset-18 graph to final opset 21")
    if export["quantization"] != "INT4_BLOCK128":
        raise ContextModelContractError("context export must use blockwise INT4")
    if export["quantizationBlockSize"] != 128:
        raise ContextModelContractError("context INT4 block size must be 128")
    maximum_bytes = _positive_int(export["maximumModelBytes"], "maximumModelBytes")
    if maximum_bytes > 24 * 1024 * 1024:
        raise ContextModelContractError("context model exceeds the sidecar size policy")
    _validate_tensors(export["inputs"], EXPECTED_INPUTS, "context inputs")
    _validate_tensors(export["outputs"], EXPECTED_OUTPUTS, "context outputs")
    return ContextModelSpec(path=path, sha256=hashlib.sha256(payload).hexdigest(), raw=raw)
