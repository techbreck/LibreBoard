#!/usr/bin/env python3
"""Evaluate session-separated LibreBoard tap/swipe predictions against the Phase 0 gates.

The harness consumes JSONL produced by Android instrumentation or an offline decoder. It never
contains a model implementation and never treats missing measurements as zero or as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = 2
TAP_SYSTEMS = ("heliboard", "fused", "fused_personal", "fused_neural")
SWIPE_SYSTEMS = ("geometric", "ctc", "fused_swipe")
ALL_SYSTEMS = TAP_SYSTEMS + SWIPE_SYSTEMS
TAP_CATEGORIES = ("tap_error", "valid_word", "spacing", "lexical")
MINIMUM_COUNTS = {
    "tap_error": 3_000,
    "valid_word": 1_000,
    "spacing": 500,
    "lexical": 500,
    "swipe": 5_000,
}
MINIMUM_SWIPE_STRATA_COUNTS = {
    "short": 500,
    "medium": 500,
    "long": 500,
    "clean": 500,
    "sloppy": 500,
    "very_sloppy": 500,
    "double_letter": 500,
    "return_trip": 500,
}
REQUIRED_ENVIRONMENTS = {
    "stock_android_hardware",
    "grapheneos_hardware",
    "low_ram_emulator",
}
MINIMUM_ENVIRONMENT_TAP_SAMPLES = 100
MINIMUM_ENVIRONMENT_SWIPE_SAMPLES = 100
MINIMUM_VALID_WORD_CORRECTIONS = 500
MINIMUM_VALID_WORD_KEEPS = 500
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class Example:
    identifier: str
    session_id: str
    environment_kind: str
    test_run_id: str
    split: str
    category: str
    target: str
    raw: str
    predictions: dict[str, tuple[str, ...]]
    latency_ms: dict[str, float]
    strata: frozenset[str]
    should_correct: bool | None


def normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def parse_example(raw: dict[str, Any], line_number: int) -> Example:
    location = f"line {line_number}"
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        raise EvaluationError(f"{location}: unsupported schemaVersion")
    required_strings = ("id", "sessionId", "split", "category", "target")
    for field in required_strings:
        if not isinstance(raw.get(field), str) or not raw[field]:
            raise EvaluationError(f"{location}: {field} must be a non-empty string")
    if raw["split"] not in {"train", "validation", "test"}:
        raise EvaluationError(f"{location}: unsupported split")
    if raw["category"] not in {*TAP_CATEGORIES, "swipe"}:
        raise EvaluationError(f"{location}: unsupported category")
    if not isinstance(raw.get("raw"), str):
        raise EvaluationError(f"{location}: raw must be a string")
    environment_kind = raw.get("environmentKind")
    test_run_id = raw.get("testRunId")
    if raw["split"] == "test":
        if not isinstance(environment_kind, str) or environment_kind not in REQUIRED_ENVIRONMENTS:
            raise EvaluationError(f"{location}: test example requires a valid environmentKind")
        if not isinstance(test_run_id, str) or not test_run_id or len(test_run_id) > 512:
            raise EvaluationError(f"{location}: test example requires testRunId")
    else:
        if environment_kind is not None or test_run_id is not None:
            raise EvaluationError(f"{location}: only test examples may identify a measurement run")
        environment_kind = ""
        test_run_id = ""

    systems = SWIPE_SYSTEMS if raw["category"] == "swipe" else TAP_SYSTEMS
    predictions_raw = raw.get("predictions")
    latency_raw = raw.get("latencyMs")
    if raw["split"] == "test":
        if not isinstance(predictions_raw, dict) or not isinstance(latency_raw, dict):
            raise EvaluationError(f"{location}: test examples require predictions and latencyMs")
        missing = [system for system in systems if system not in predictions_raw or system not in latency_raw]
        if missing:
            raise EvaluationError(f"{location}: missing systems: {', '.join(missing)}")
    predictions: dict[str, tuple[str, ...]] = {}
    for system, values in (predictions_raw or {}).items():
        if system not in ALL_SYSTEMS or not isinstance(values, list) or not values or len(values) > 32:
            raise EvaluationError(f"{location}: invalid prediction list for {system}")
        if any(not isinstance(value, str) or not value for value in values):
            raise EvaluationError(f"{location}: predictions must be non-empty strings")
        predictions[system] = tuple(values)
    if raw["split"] == "test" and raw["category"] != "swipe":
        if not raw["raw"]:
            raise EvaluationError(f"{location}: measured tap raw text must not be empty")
        for system in ("fused", "fused_personal", "fused_neural"):
            if raw["raw"] not in predictions[system]:
                raise EvaluationError(
                    f"{location}: {system} does not preserve the exact raw candidate"
                )
    latency: dict[str, float] = {}
    for system, value in (latency_raw or {}).items():
        if system not in ALL_SYSTEMS or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationError(f"{location}: invalid latency for {system}")
        if not math.isfinite(value) or value < 0 or value > 60_000:
            raise EvaluationError(f"{location}: out-of-range latency for {system}")
        latency[system] = float(value)

    strata_raw = raw.get("strata", [])
    if not isinstance(strata_raw, list) or any(not isinstance(value, str) or not value for value in strata_raw):
        raise EvaluationError(f"{location}: strata must be strings")
    should_correct = raw.get("shouldCorrect")
    if raw["category"] == "valid_word" and not isinstance(should_correct, bool):
        raise EvaluationError(f"{location}: valid_word examples require shouldCorrect")
    if should_correct is not None and not isinstance(should_correct, bool):
        raise EvaluationError(f"{location}: shouldCorrect must be boolean")

    return Example(
        identifier=raw["id"],
        session_id=raw["sessionId"],
        environment_kind=environment_kind,
        test_run_id=test_run_id,
        split=raw["split"],
        category=raw["category"],
        target=raw["target"],
        raw=raw["raw"],
        predictions=predictions,
        latency_ms=latency,
        strata=frozenset(strata_raw),
        should_correct=should_correct,
    )


def read_jsonl(path: pathlib.Path) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as failure:
                    raise EvaluationError(f"line {line_number}: invalid JSON: {failure.msg}") from failure
                if not isinstance(value, dict):
                    raise EvaluationError(f"line {line_number}: example must be an object")
                examples.append(parse_example(value, line_number))
    if not examples:
        raise EvaluationError("dataset is empty")
    return examples


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise EvaluationError("required latency samples are missing")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def accuracy(examples: Iterable[Example], system: str, rank: int = 1) -> float:
    rows = list(examples)
    if not rows:
        raise EvaluationError(f"no examples for {system} accuracy")
    hits = sum(
        normalized(row.target) in {normalized(value) for value in row.predictions[system][:rank]}
        for row in rows
    )
    return hits / len(rows)


def relative_error_reduction(baseline_accuracy: float, new_accuracy: float) -> float:
    baseline_error = 1.0 - baseline_accuracy
    if baseline_error <= 0:
        return 0.0 if new_accuracy >= baseline_accuracy else -math.inf
    return (baseline_error - (1.0 - new_accuracy)) / baseline_error


def false_correction_rate(examples: Iterable[Example], system: str) -> float:
    rows = [row for row in examples if row.should_correct is False]
    if not rows:
        raise EvaluationError("valid-word keep cases are missing")
    false_corrections = sum(normalized(row.predictions[system][0]) != normalized(row.raw) for row in rows)
    return false_corrections / len(rows)


def validate_swipe_strata(
    swipe: list[Example],
    minimums: dict[str, int],
) -> dict[str, int]:
    counts = Counter(stratum for example in swipe for stratum in example.strata)
    missing = [
        f"{stratum}={counts[stratum]}<{minimum}"
        for stratum, minimum in minimums.items()
        if counts[stratum] < minimum
    ]
    if missing:
        raise EvaluationError("swipe stratum minimums not met: " + ", ".join(missing))
    return dict(sorted(counts.items()))


def check_dataset(
    examples: list[Example],
    environments: list[dict[str, Any]],
    enforce_minimum_counts: bool,
) -> tuple[
    list[Example],
    dict[str, int],
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, int],
]:
    ids = Counter(example.identifier for example in examples)
    duplicates = sorted(identifier for identifier, count in ids.items() if count > 1)
    if duplicates:
        raise EvaluationError(f"duplicate example ids: {', '.join(duplicates[:5])}")
    session_splits: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        session_splits[example.session_id].add(example.split)
    leaked = sorted(session for session, splits in session_splits.items() if len(splits) > 1)
    if leaked:
        raise EvaluationError(f"sessions cross dataset splits: {', '.join(leaked[:5])}")

    test = [example for example in examples if example.split == "test"]
    run_to_kind = {environment["testRunId"]: environment["kind"] for environment in environments}
    for example in test:
        expected_kind = run_to_kind.get(example.test_run_id)
        if expected_kind is None:
            raise EvaluationError(
                f"example {example.identifier}: testRunId is not declared by metadata"
            )
        if expected_kind != example.environment_kind:
            raise EvaluationError(
                f"example {example.identifier}: environmentKind disagrees with metadata"
            )

    environment_counts: dict[str, dict[str, int]] = {}
    for environment in environments:
        kind = environment["kind"]
        run_id = environment["testRunId"]
        rows = [example for example in test if example.test_run_id == run_id]
        tap_count = sum(example.category != "swipe" for example in rows)
        swipe_count = sum(example.category == "swipe" for example in rows)
        environment_counts[kind] = {
            "tap": tap_count,
            "swipe": swipe_count,
            "total": len(rows),
        }
        if enforce_minimum_counts and (
            tap_count < MINIMUM_ENVIRONMENT_TAP_SAMPLES
            or swipe_count < MINIMUM_ENVIRONMENT_SWIPE_SAMPLES
        ):
            raise EvaluationError(
                f"measurement coverage for {kind} is insufficient: "
                f"tap={tap_count}<{MINIMUM_ENVIRONMENT_TAP_SAMPLES}, "
                f"swipe={swipe_count}<{MINIMUM_ENVIRONMENT_SWIPE_SAMPLES}"
            )
    counts = Counter(example.category for example in test)
    if enforce_minimum_counts:
        missing = [f"{category}={counts[category]}<{minimum}" for category, minimum in MINIMUM_COUNTS.items()
                   if counts[category] < minimum]
        if missing:
            raise EvaluationError("held-out dataset minimums not met: " + ", ".join(missing))
    valid_word_counts = {
        "correct": sum(
            example.category == "valid_word" and example.should_correct is True
            for example in test
        ),
        "keep": sum(
            example.category == "valid_word" and example.should_correct is False
            for example in test
        ),
    }
    if enforce_minimum_counts and (
        valid_word_counts["correct"] < MINIMUM_VALID_WORD_CORRECTIONS
        or valid_word_counts["keep"] < MINIMUM_VALID_WORD_KEEPS
    ):
        raise EvaluationError(
            "valid-word strata minimums not met: "
            f"correct={valid_word_counts['correct']}<{MINIMUM_VALID_WORD_CORRECTIONS}, "
            f"keep={valid_word_counts['keep']}<{MINIMUM_VALID_WORD_KEEPS}"
        )
    swipe = [example for example in test if example.category == "swipe"]
    strata = validate_swipe_strata(
        swipe,
        MINIMUM_SWIPE_STRATA_COUNTS if enforce_minimum_counts else {"short": 1, "return_trip": 1},
    )
    return test, dict(sorted(counts.items())), strata, environment_counts, valid_word_counts


def validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("schemaVersion") != SCHEMA_VERSION:
        raise EvaluationError("metadata has unsupported schemaVersion")
    peak_memory = metadata.get("peakAddedNeuralMemoryMiB")
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, (int, float)) or peak_memory < 0:
        raise EvaluationError("metadata requires non-negative peakAddedNeuralMemoryMiB")

    app_commit = metadata.get("appCommit")
    if not isinstance(app_commit, str) or not GIT_COMMIT.fullmatch(app_commit):
        raise EvaluationError("metadata requires a full lowercase appCommit")
    hashes: dict[str, str] = {}
    for field in ("coreApkSha256", "swipeModelSha256", "contextModelSha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise EvaluationError(f"metadata requires lowercase {field}")
        hashes[field] = value

    environments = metadata.get("environments")
    if not isinstance(environments, list) or len(environments) != len(REQUIRED_ENVIRONMENTS):
        raise EvaluationError("metadata requires exactly the three reference environments")
    validated_environments: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_run_ids: set[str] = set()
    for index, environment in enumerate(environments):
        location = f"metadata environment {index + 1}"
        if not isinstance(environment, dict):
            raise EvaluationError(f"{location} must be an object")
        kind = environment.get("kind")
        if not isinstance(kind, str) or kind not in REQUIRED_ENVIRONMENTS or kind in seen:
            raise EvaluationError(f"{location} has an invalid or duplicate kind")
        seen.add(kind)
        required_strings = ["deviceModel", "buildFingerprint", "testRunId"]
        if kind == "grapheneos_hardware":
            required_strings.append("grapheneOsBuildNumber")
        for field in required_strings:
            value = environment.get(field)
            if not isinstance(value, str) or not value or len(value) > 512:
                raise EvaluationError(f"{location} requires {field}")
        test_run_id = environment["testRunId"]
        if test_run_id in seen_run_ids:
            raise EvaluationError(f"{location} has a duplicate testRunId")
        seen_run_ids.add(test_run_id)
        api_level = environment.get("apiLevel")
        if isinstance(api_level, bool) or not isinstance(api_level, int) or api_level < 26 or api_level > 100:
            raise EvaluationError(f"{location} has an invalid apiLevel")

        if kind.endswith("_hardware"):
            if environment.get("physicalDevice") is not True or api_level < 35:
                raise EvaluationError(f"{location} must identify Android 15+ physical hardware")
        else:
            memory = environment.get("memoryMiB")
            if (environment.get("physicalDevice") is not False
                    or environment.get("isLowRamDevice") is not True
                    or isinstance(memory, bool)
                    or not isinstance(memory, int)
                    or memory <= 0
                    or memory > 2_048):
                raise EvaluationError(f"{location} must identify a low-RAM emulator")

        if kind == "grapheneos_hardware" and environment.get("sandboxedGooglePlayInstalled") is not False:
            raise EvaluationError(f"{location} must run without sandboxed Google Play")
        validated_environments.append(environment)

    if seen != REQUIRED_ENVIRONMENTS:
        raise EvaluationError("metadata is missing a reference environment")
    return {
        "appCommit": app_commit,
        **hashes,
        "environments": validated_environments,
        "peakAddedNeuralMemoryMiB": float(peak_memory),
    }


def evaluate(
    examples: list[Example],
    metadata: dict[str, Any],
    *,
    measurement_sha256: str,
    enforce_minimum_counts: bool = True,
) -> dict[str, Any]:
    if not isinstance(measurement_sha256, str) or not SHA256.fullmatch(measurement_sha256):
        raise EvaluationError("measurement dataset requires a lowercase SHA-256")
    evidence = validate_metadata(metadata)
    test, counts, swipe_strata, environment_counts, valid_word_counts = check_dataset(
        examples,
        evidence["environments"],
        enforce_minimum_counts,
    )
    evidence["measurementDatasetSha256"] = measurement_sha256
    peak_memory = evidence["peakAddedNeuralMemoryMiB"]

    tap_error = [row for row in test if row.category == "tap_error"]
    valid_word = [row for row in test if row.category == "valid_word"]
    swipe = [row for row in test if row.category == "swipe"]
    tap_all = [row for row in test if row.category in TAP_CATEGORIES]
    metrics: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "evidence": evidence,
        "counts": counts,
        "swipeStrataCounts": swipe_strata,
        "validWordCounts": valid_word_counts,
        "environmentCounts": environment_counts,
        "environmentLatencyMs": {},
        "systems": {},
    }
    for system in TAP_SYSTEMS:
        relevant = tap_all
        metrics["systems"][system] = {
            "top1": accuracy(relevant, system),
            "top3": accuracy(relevant, system, 3),
            "latencyMs": {
                "p50": percentile([row.latency_ms[system] for row in relevant], 0.50),
                "p95": percentile([row.latency_ms[system] for row in relevant], 0.95),
                "p99": percentile([row.latency_ms[system] for row in relevant], 0.99),
            },
        }
    for system in SWIPE_SYSTEMS:
        metrics["systems"][system] = {
            "top1": accuracy(swipe, system),
            "top3": accuracy(swipe, system, 3),
            "latencyMs": {
                "p50": percentile([row.latency_ms[system] for row in swipe], 0.50),
                "p95": percentile([row.latency_ms[system] for row in swipe], 0.95),
                "p99": percentile([row.latency_ms[system] for row in swipe], 0.99),
            },
        }

    for environment in evidence["environments"]:
        kind = environment["kind"]
        run_id = environment["testRunId"]
        environment_taps = [row for row in tap_all if row.test_run_id == run_id]
        environment_swipes = [row for row in swipe if row.test_run_id == run_id]
        metrics["environmentLatencyMs"][kind] = {
            "tap": {
                "p50": percentile([row.latency_ms["fused_neural"] for row in environment_taps], 0.50),
                "p95": percentile([row.latency_ms["fused_neural"] for row in environment_taps], 0.95),
                "p99": percentile([row.latency_ms["fused_neural"] for row in environment_taps], 0.99),
            },
            "swipe": {
                "p50": percentile([row.latency_ms["fused_swipe"] for row in environment_swipes], 0.50),
                "p95": percentile([row.latency_ms["fused_swipe"] for row in environment_swipes], 0.95),
                "p99": percentile([row.latency_ms["fused_swipe"] for row in environment_swipes], 0.99),
            },
        }

    heliboard_tap = accuracy(tap_error, "heliboard")
    fused_tap = accuracy(tap_error, "fused")
    context_confusions = [row for row in valid_word if row.should_correct is True]
    fused_context = accuracy(context_confusions, "fused")
    neural_context = accuracy(context_confusions, "fused_neural")
    fused_valid = accuracy(valid_word, "fused")
    neural_valid = accuracy(valid_word, "fused_neural")
    geometric_swipe = accuracy(swipe, "geometric")
    final_swipe = accuracy(swipe, "fused_swipe")
    metrics["gates"] = {
        "tapRelativeErrorReduction": relative_error_reduction(heliboard_tap, fused_tap),
        "neuralContextRelativeErrorReduction": relative_error_reduction(fused_context, neural_context),
        "neuralValidWordAbsoluteGain": neural_valid - fused_valid,
        "falseCorrectionIncrease": false_correction_rate(valid_word, "fused_neural")
            - false_correction_rate(valid_word, "fused"),
        "swipeGeometricRelativeErrorReduction": relative_error_reduction(geometric_swipe, final_swipe),
        "swipeShortTop3": accuracy([row for row in swipe if "short" in row.strata], "fused_swipe", 3),
        "swipeReturnTripTop3": accuracy([row for row in swipe if "return_trip" in row.strata], "fused_swipe", 3),
        "peakAddedNeuralMemoryMiB": float(peak_memory),
    }
    gates = metrics["gates"]
    checks = {
        "tap_relative_error_reduction": gates["tapRelativeErrorReduction"] >= 0.20,
        "neural_valid_word_relative_error_reduction": gates["neuralContextRelativeErrorReduction"] >= 0.15,
        "neural_valid_word_absolute_gain": gates["neuralValidWordAbsoluteGain"] >= 0.05,
        "false_correction_ceiling": gates["falseCorrectionIncrease"] <= 0.005,
        "swipe_top1": metrics["systems"]["fused_swipe"]["top1"] >= 0.90,
        "swipe_top3": metrics["systems"]["fused_swipe"]["top3"] >= 0.95,
        "swipe_short_top3": gates["swipeShortTop3"] >= 0.90,
        "swipe_return_trip_top3": gates["swipeReturnTripTop3"] >= 0.90,
        "swipe_geometric_relative_error_reduction": gates["swipeGeometricRelativeErrorReduction"] >= 0.20,
        "tap_p95_latency": all(
            values["tap"]["p95"] <= 80.0
            for values in metrics["environmentLatencyMs"].values()
        ),
        "swipe_p95_latency": all(
            values["swipe"]["p95"] <= 200.0
            for values in metrics["environmentLatencyMs"].values()
        ),
        "peak_neural_memory": peak_memory <= 64.0,
    }
    metrics["checks"] = checks
    metrics["passed"] = all(checks.values())
    return metrics


def render(metrics: dict[str, Any]) -> str:
    lines = ["LibreBoard Phase 0 evaluation", ""]
    for system, values in metrics["systems"].items():
        latency = values["latencyMs"]
        lines.append(
            f"{system:16} top1={values['top1']:.3%} top3={values['top3']:.3%} "
            f"p95={latency['p95']:.1f}ms"
        )
    lines.append("")
    for kind, values in metrics["environmentLatencyMs"].items():
        lines.append(
            f"{kind:24} tap-p95={values['tap']['p95']:.1f}ms "
            f"swipe-p95={values['swipe']['p95']:.1f}ms"
        )
    lines.append("")
    lines.extend(f"{'PASS' if passed else 'FAIL'} {name}" for name, passed in metrics["checks"].items())
    lines.append("")
    lines.append("OVERALL PASS" if metrics["passed"] else "OVERALL FAIL")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path, help="session-separated JSONL predictions")
    parser.add_argument("--metadata", required=True, type=pathlib.Path, help="measurement metadata JSON")
    parser.add_argument("--report", type=pathlib.Path, help="write the complete JSON report")
    parser.add_argument("--allow-small-dataset", action="store_true", help="test the harness without release-size data")
    args = parser.parse_args()
    try:
        examples = read_jsonl(args.dataset)
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise EvaluationError("metadata must be an object")
        metrics = evaluate(
            examples,
            metadata,
            measurement_sha256=sha256_file(args.dataset),
            enforce_minimum_counts=not args.allow_small_dataset,
        )
    except (OSError, json.JSONDecodeError, EvaluationError) as failure:
        print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render(metrics))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
