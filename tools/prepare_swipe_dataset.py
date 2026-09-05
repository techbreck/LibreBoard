#!/usr/bin/env python3
"""Create deterministic, session-separated CTC training inputs from pinned swipe data."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import sqlite3
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator

import model_sources


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "swipe" / "data-policy.json"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-data" / "swipe-latin-v1"
POLICY_KEYS = {
    "schemaVersion",
    "sourceId",
    "dataArtifacts",
    "layoutArtifact",
    "splitSalt",
    "splitBasisPoints",
    "pathPoints",
    "keySlots",
    "outputFrames",
    "maximumSourcePoints",
    "maximumLineBytes",
    "maximumRejectedFraction",
    "minimumCoordinate",
    "maximumCoordinate",
    "shortMaximumCodePoints",
    "longMinimumCodePoints",
    "sloppyPercentile",
    "verySloppyPercentile",
}
SPLITS = ("train", "validation", "test")
OUTPUT_SCHEMA_VERSION = 1
MAXIMUM_POLICY_BYTES = 256 * 1024
MAXIMUM_TARGET_CODEPOINTS = 64


class SwipeDataError(ValueError):
    """The source data, split policy, or generated corpus is unsafe or inconsistent."""


@dataclass(frozen=True)
class Policy:
    path: pathlib.Path
    sha256: str
    source_id: str
    data_artifacts: tuple[str, ...]
    layout_artifact: str
    split_salt: str
    split_basis_points: dict[str, int]
    path_points: int
    key_slots: int
    output_frames: int
    maximum_source_points: int
    maximum_line_bytes: int
    maximum_rejected_fraction: float
    minimum_coordinate: float
    maximum_coordinate: float
    short_maximum_codepoints: int
    long_minimum_codepoints: int
    sloppy_percentile: float
    very_sloppy_percentile: float


@dataclass(frozen=True)
class Layout:
    identifier: str
    labels: tuple[str | None, ...]
    centers: tuple[float, ...]
    mask: tuple[int, ...]
    class_by_label: dict[str, int]
    left: float
    top: float
    width: float
    height: float


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_json(path: pathlib.Path, maximum_bytes: int, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        if not path.is_file() or path.is_symlink():
            raise SwipeDataError(f"{label} is not a regular file: {path}")
        if path.stat().st_size > maximum_bytes:
            raise SwipeDataError(f"{label} is too large")
        payload = path.read_bytes()
        value = json.loads(payload)
    except SwipeDataError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise SwipeDataError(f"cannot read {label}: {failure}") from failure
    if not isinstance(value, dict):
        raise SwipeDataError(f"{label} must be a JSON object")
    return value, payload


def load_policy(path: pathlib.Path = DEFAULT_POLICY) -> Policy:
    path = path.resolve()
    raw, payload = _read_json(path, MAXIMUM_POLICY_BYTES, "swipe data policy")
    if set(raw) != POLICY_KEYS or raw.get("schemaVersion") != 1:
        raise SwipeDataError("swipe data policy has an unexpected schema")
    source_id = raw["sourceId"]
    if not isinstance(source_id, str) or not model_sources.SOURCE_ID.fullmatch(source_id):
        raise SwipeDataError("swipe data policy has an invalid sourceId")
    artifacts = raw["dataArtifacts"]
    if not isinstance(artifacts, list) or not artifacts or any(not isinstance(value, str) for value in artifacts):
        raise SwipeDataError("swipe data policy requires data artifacts")
    if len(artifacts) != len(set(artifacts)):
        raise SwipeDataError("swipe data artifacts must be unique")
    layout_artifact = raw["layoutArtifact"]
    if not isinstance(layout_artifact, str) or not layout_artifact:
        raise SwipeDataError("swipe data policy requires a layout artifact")
    split_salt = raw["splitSalt"]
    if not isinstance(split_salt, str) or not 16 <= len(split_salt) <= 128:
        raise SwipeDataError("swipe split salt must be a stable bounded string")
    split_points = raw["splitBasisPoints"]
    if not isinstance(split_points, dict) or set(split_points) != set(SPLITS):
        raise SwipeDataError("swipe split policy must define train, validation, and test")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in split_points.values()):
        raise SwipeDataError("swipe split sizes must be positive integers")
    if sum(split_points.values()) != 10_000:
        raise SwipeDataError("swipe split basis points must total 10000")

    integer_bounds = {
        "pathPoints": (2, 1024),
        "keySlots": (1, 64),
        "outputFrames": (1, 1024),
        "maximumSourcePoints": (2, 100_000),
        "maximumLineBytes": (1024, 16 * 1024 * 1024),
        "shortMaximumCodePoints": (1, 16),
        "longMinimumCodePoints": (2, 64),
    }
    validated_integers: dict[str, int] = {}
    for field, (minimum, maximum) in integer_bounds.items():
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise SwipeDataError(f"swipe data policy has invalid {field}")
        validated_integers[field] = value
    if validated_integers["shortMaximumCodePoints"] >= validated_integers["longMinimumCodePoints"]:
        raise SwipeDataError("swipe length strata overlap")

    finite_values: dict[str, float] = {}
    for field in (
        "maximumRejectedFraction",
        "minimumCoordinate",
        "maximumCoordinate",
        "sloppyPercentile",
        "verySloppyPercentile",
    ):
        value = _finite_number(raw[field])
        if value is None:
            raise SwipeDataError(f"swipe data policy has invalid {field}")
        finite_values[field] = value
    if not 0 <= finite_values["maximumRejectedFraction"] < 1:
        raise SwipeDataError("maximumRejectedFraction must be in [0, 1)")
    if not finite_values["minimumCoordinate"] < 0 < 1 < finite_values["maximumCoordinate"]:
        raise SwipeDataError("coordinate bounds must contain the normalized keyboard")
    if not 0 < finite_values["sloppyPercentile"] < finite_values["verySloppyPercentile"] < 1:
        raise SwipeDataError("sloppiness percentiles must be ordered inside (0, 1)")
    if validated_integers["keySlots"] != 64 or validated_integers["pathPoints"] != 64:
        raise SwipeDataError("swipe-latin-v1 requires exactly 64 path points and key slots")
    if validated_integers["outputFrames"] != 32:
        raise SwipeDataError("swipe-latin-v1 requires exactly 32 output frames")

    return Policy(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_id=source_id,
        data_artifacts=tuple(artifacts),
        layout_artifact=layout_artifact,
        split_salt=split_salt,
        split_basis_points=dict(split_points),
        path_points=validated_integers["pathPoints"],
        key_slots=validated_integers["keySlots"],
        output_frames=validated_integers["outputFrames"],
        maximum_source_points=validated_integers["maximumSourcePoints"],
        maximum_line_bytes=validated_integers["maximumLineBytes"],
        maximum_rejected_fraction=finite_values["maximumRejectedFraction"],
        minimum_coordinate=finite_values["minimumCoordinate"],
        maximum_coordinate=finite_values["maximumCoordinate"],
        short_maximum_codepoints=validated_integers["shortMaximumCodePoints"],
        long_minimum_codepoints=validated_integers["longMinimumCodePoints"],
        sloppy_percentile=finite_values["sloppyPercentile"],
        very_sloppy_percentile=finite_values["verySloppyPercentile"],
    )


def _normalize_label(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().replace("\u2019", "'")


def load_layout(path: pathlib.Path, policy: Policy) -> Layout:
    raw, _payload = _read_json(path, 1024 * 1024, "swipe layout")
    identifier = raw.get("name")
    keys = raw.get("keys")
    if not isinstance(identifier, str) or not identifier or not isinstance(keys, list) or not keys:
        raise SwipeDataError("swipe layout requires a name and keys")
    parsed = []
    for index, key in enumerate(keys):
        if not isinstance(key, dict):
            raise SwipeDataError(f"swipe layout key {index + 1} must be an object")
        label = key.get("letter")
        if not isinstance(label, str):
            raise SwipeDataError(f"swipe layout key {index + 1} has no letter")
        label = _normalize_label(label)
        if len(label) != 1 or not label.isalpha():
            raise SwipeDataError(f"swipe layout key {index + 1} has an invalid letter")
        numbers = tuple(_finite_number(key.get(field)) for field in ("cx", "cy", "rx", "ry"))
        if any(value is None for value in numbers):
            raise SwipeDataError(f"swipe layout key {index + 1} has invalid geometry")
        cx, cy, rx, ry = numbers
        assert cx is not None and cy is not None and rx is not None and ry is not None
        if rx <= 0 or ry <= 0:
            raise SwipeDataError(f"swipe layout key {index + 1} has non-positive geometry")
        parsed.append((label, cx, cy, rx, ry))
    labels = [value[0] for value in parsed]
    if len(labels) != len(set(labels)) or len(labels) > policy.key_slots:
        raise SwipeDataError("swipe layout labels must be unique and fit the tensor ABI")

    # Production KeyGeometry uses Keyboard.sortedKeys: top-to-bottom, then left-to-right.
    parsed.sort(key=lambda value: (value[2], value[1], value[0]))
    left = min(cx - rx for _label, cx, _cy, rx, _ry in parsed)
    right = max(cx + rx for _label, cx, _cy, rx, _ry in parsed)
    top = min(cy - ry for _label, _cx, cy, _rx, ry in parsed)
    bottom = max(cy + ry for _label, _cx, cy, _rx, ry in parsed)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise SwipeDataError("swipe layout bounds are empty")

    output_labels: list[str | None] = [None] * policy.key_slots
    centers = [0.0] * (policy.key_slots * 2)
    mask = [0] * policy.key_slots
    class_by_label: dict[str, int] = {}
    for index, (label, cx, cy, _rx, _ry) in enumerate(parsed):
        output_labels[index] = label
        centers[index * 2] = round((cx - left) / width, 8)
        centers[index * 2 + 1] = round((cy - top) / height, 8)
        mask[index] = 1
        class_by_label[label] = index + 1
    return Layout(
        identifier=identifier,
        labels=tuple(output_labels),
        centers=tuple(centers),
        mask=tuple(mask),
        class_by_label=class_by_label,
        left=left,
        top=top,
        width=width,
        height=height,
    )


def session_hash(session: str, policy: Policy) -> str:
    return hashlib.sha256((policy.split_salt + "\0" + session).encode("utf-8")).hexdigest()


def split_for_session(session: str, policy: Policy) -> str:
    bucket = int(session_hash(session, policy)[:8], 16) % 10_000
    cursor = 0
    for split in SPLITS:
        cursor += policy.split_basis_points[split]
        if bucket < cursor:
            return split
    raise AssertionError("split basis points did not cover the hash space")


def _resample(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]] | None:
    cumulative = [0.0]
    for index in range(1, len(points)):
        cumulative.append(cumulative[-1] + math.dist(points[index - 1], points[index]))
    total = cumulative[-1]
    if total <= 1e-9:
        return None
    result = []
    upper = 1
    for sample_index in range(count):
        target = total * sample_index / (count - 1)
        while upper < len(cumulative) - 1 and cumulative[upper] < target:
            upper += 1
        lower = upper - 1
        span = cumulative[upper] - cumulative[lower]
        fraction = 0.0 if span <= 0 else (target - cumulative[lower]) / span
        x = points[lower][0] + (points[upper][0] - points[lower][0]) * fraction
        y = points[lower][1] + (points[upper][1] - points[lower][1]) * fraction
        result.append((round(x, 8), round(y, 8)))
    return result


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return math.dist(point, start)
    projection = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator
    projection = max(0.0, min(1.0, projection))
    nearest = (start[0] + projection * dx, start[1] + projection * dy)
    return math.dist(point, nearest)


def _geometric_deviation(
    sampled: list[tuple[float, float]],
    labels: list[int],
    layout: Layout,
) -> float:
    centers = [(layout.centers[(label - 1) * 2], layout.centers[(label - 1) * 2 + 1]) for label in labels]
    if len(centers) == 1:
        distances = [math.dist(point, centers[0]) for point in sampled]
    else:
        distances = [
            min(_point_segment_distance(point, centers[index - 1], centers[index]) for index in range(1, len(centers)))
            for point in sampled
        ]
    endpoint = math.dist(sampled[0], centers[0]) + math.dist(sampled[-1], centers[-1])
    return round(sum(distances) / len(distances) + endpoint / 4.0, 8)


def _base_strata(target_labels: list[int], policy: Policy) -> list[str]:
    length = len(target_labels)
    if length <= policy.short_maximum_codepoints:
        strata = ["short"]
    elif length >= policy.long_minimum_codepoints:
        strata = ["long"]
    else:
        strata = ["medium"]
    if any(target_labels[index] == target_labels[index - 1] for index in range(1, length)):
        strata.append("double_letter")
    if any(
        target_labels[index] in target_labels[: index - 1]
        for index in range(2, length)
    ):
        strata.append("return_trip")
    return strata


def normalize_record(
    raw: Any,
    *,
    source_artifact: str,
    line_number: int,
    source_revision: str,
    policy: Policy,
    layout: Layout,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not_object"
    session = raw.get("session")
    if not isinstance(session, str) or not session or len(session) > 256:
        return None, "invalid_session"
    word = raw.get("word")
    if not isinstance(word, str):
        return None, "invalid_target"
    target = _normalize_label(word.strip())
    if not target or len(target) > MAXIMUM_TARGET_CODEPOINTS:
        return None, "invalid_target"
    target_labels = []
    for character in target:
        if character in {"'", "-"}:
            continue
        output_class = layout.class_by_label.get(character)
        if output_class is None:
            return None, "unsupported_target"
        target_labels.append(output_class)
    if not target_labels:
        return None, "invalid_target"
    minimum_ctc_frames = len(target_labels) + sum(
        target_labels[index] == target_labels[index - 1]
        for index in range(1, len(target_labels))
    )
    if minimum_ctc_frames > policy.output_frames:
        return None, "target_exceeds_ctc_frames"

    source_points = raw.get("data")
    if not isinstance(source_points, list) or not 2 <= len(source_points) <= policy.maximum_source_points:
        return None, "invalid_point_count"
    points: list[tuple[float, float]] = []
    for point in source_points:
        if not isinstance(point, dict):
            return None, "invalid_point"
        x = _finite_number(point.get("x"))
        y = _finite_number(point.get("y"))
        if x is None or y is None:
            return None, "invalid_point"
        normalized_x = (x - layout.left) / layout.width
        normalized_y = (y - layout.top) / layout.height
        if not (
            policy.minimum_coordinate <= normalized_x <= policy.maximum_coordinate
            and policy.minimum_coordinate <= normalized_y <= policy.maximum_coordinate
        ):
            return None, "out_of_bounds_point"
        points.append((normalized_x, normalized_y))
    sampled = _resample(points, policy.path_points)
    if sampled is None:
        return None, "stationary_path"
    flattened = [coordinate for point in sampled for coordinate in point]
    hashed_session = session_hash(session, policy)
    split = split_for_session(session, policy)
    source_identifier = raw.get("id")
    if not isinstance(source_identifier, (str, int)) or isinstance(source_identifier, bool):
        source_identifier = line_number
    stable_id = hashlib.sha256(
        (
            source_revision + "\0" + source_artifact + "\0" + str(source_identifier) + "\0"
            + hashed_session + "\0" + target + "\0" + str(line_number)
        ).encode("utf-8")
    ).hexdigest()
    orientation = raw.get("orientation")
    if not isinstance(orientation, str) or not orientation or len(orientation) > 64:
        orientation = "unknown"
    deviation = _geometric_deviation(sampled, target_labels, layout)
    record = {
        "schemaVersion": OUTPUT_SCHEMA_VERSION,
        "id": stable_id,
        "sessionId": hashed_session,
        "split": split,
        "language": "en",
        "target": target,
        "ctcLabels": target_labels,
        "pathCoordinates": flattened,
        "layoutId": layout.identifier,
        "orientation": orientation,
        "geometricDeviation": deviation,
        "strata": _base_strata(target_labels, policy),
    }
    return record, None


def _verified_json_lines(
    path: pathlib.Path,
    artifact: model_sources.Artifact,
    maximum_line_bytes: int,
) -> Iterator[tuple[int, Any]]:
    if not path.is_file() or path.is_symlink():
        raise SwipeDataError(f"missing regular swipe source: {path}")
    if path.stat().st_size != artifact.bytes:
        raise SwipeDataError(f"swipe source has wrong size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(maximum_line_bytes + 1)
            if not line:
                break
            line_number += 1
            digest.update(line)
            if len(line) > maximum_line_bytes:
                raise SwipeDataError(f"{artifact.path} line {line_number} exceeds the size limit")
            try:
                yield line_number, json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise SwipeDataError(f"{artifact.path} line {line_number} is invalid JSON: {failure}") from failure
    if digest.hexdigest() != artifact.sha256:
        raise SwipeDataError(f"swipe source has wrong SHA-256: {path}")


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise SwipeDataError("training split has no geometric-deviation samples")
    values.sort()
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _write_layout(path: pathlib.Path, layout: Layout, policy: Policy) -> None:
    value = {
        "schemaVersion": 1,
        "id": layout.identifier,
        "keyLabels": layout.labels,
        "keyCenters": layout.centers,
        "keyMask": layout.mask,
        "pathShape": [1, policy.path_points, 2],
        "keyCentersShape": [1, policy.key_slots, 2],
        "keyMaskShape": [1, policy.key_slots],
    }
    path.write_bytes(_canonical_line(value))


def _finalize_split(
    staged_path: pathlib.Path,
    final_path: pathlib.Path,
    sloppy_threshold: float,
    very_sloppy_threshold: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    strata_counts: collections.Counter[str] = collections.Counter()
    with staged_path.open("rb") as source, final_path.open("wb") as output:
        for line in source:
            record = json.loads(line)
            deviation = record["geometricDeviation"]
            strata = list(record["strata"])
            if deviation >= very_sloppy_threshold:
                strata.append("very_sloppy")
            if deviation >= sloppy_threshold:
                strata.append("sloppy")
            else:
                strata.append("clean")
            record["strata"] = strata
            strata_counts.update(strata)
            output.write(_canonical_line(record))
        output.flush()
        os.fsync(output.fileno())
    return dict(sorted(strata_counts.items())), {
        "bytes": final_path.stat().st_size,
        "sha256": model_sources.file_sha256(final_path),
    }


def prepare(
    *,
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    manifest_path: pathlib.Path = model_sources.DEFAULT_MANIFEST,
    policy_path: pathlib.Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    manifest = model_sources.load_manifest(manifest_path)
    policy = load_policy(policy_path)
    source = manifest.source(policy.source_id)
    if source.kind != "dataset" or source.license != "MIT":
        raise SwipeDataError("swipe source must be the pinned MIT dataset")
    for artifact_name in (*policy.data_artifacts, policy.layout_artifact):
        source.artifact(artifact_name)
    layout_artifact = source.artifact(policy.layout_artifact)
    layout_path = model_sources.artifact_path(source_root, source, layout_artifact)
    try:
        model_sources.verify_artifact(layout_path, layout_artifact)
    except model_sources.ModelSourceError as failure:
        raise SwipeDataError(str(failure)) from failure
    layout = load_layout(layout_path, policy)

    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    counters: collections.Counter[str] = collections.Counter()
    accepted_by_split: collections.Counter[str] = collections.Counter()
    training_deviations: list[float] = []
    session_counts: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="libreboard-swipe-", dir=output_root.parent) as temporary:
        staging = pathlib.Path(temporary)
        staged_paths = {split: staging / f"{split}.staged.jsonl" for split in SPLITS}
        handles: dict[str, BinaryIO] = {split: path.open("wb") for split, path in staged_paths.items()}
        database_path = staging / "dedup.sqlite3"
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE fingerprints (fingerprint TEXT PRIMARY KEY)")
        database.execute(
            "CREATE TABLE sessions (session_id TEXT NOT NULL, split TEXT NOT NULL, PRIMARY KEY(session_id, split))"
        )
        try:
            for artifact_name in policy.data_artifacts:
                artifact = source.artifact(artifact_name)
                path = model_sources.artifact_path(source_root, source, artifact)
                for line_number, raw in _verified_json_lines(path, artifact, policy.maximum_line_bytes):
                    counters["inputRows"] += 1
                    record, rejection = normalize_record(
                        raw,
                        source_artifact=artifact.path,
                        line_number=line_number,
                        source_revision=source.revision,
                        policy=policy,
                        layout=layout,
                    )
                    if record is None:
                        counters["rejectedRows"] += 1
                        counters[f"rejection:{rejection}"] += 1
                        continue
                    fingerprint = hashlib.sha256(
                        _canonical_line({
                            "sessionId": record["sessionId"],
                            "target": record["target"],
                            "pathCoordinates": record["pathCoordinates"],
                        })
                    ).hexdigest()
                    inserted = database.execute(
                        "INSERT OR IGNORE INTO fingerprints(fingerprint) VALUES (?)",
                        (fingerprint,),
                    ).rowcount
                    if inserted == 0:
                        counters["duplicateRows"] += 1
                        continue
                    database.execute(
                        "INSERT OR IGNORE INTO sessions(session_id, split) VALUES (?, ?)",
                        (record["sessionId"], record["split"]),
                    )
                    handles[record["split"]].write(_canonical_line(record))
                    accepted_by_split[record["split"]] += 1
                    counters["acceptedRows"] += 1
                    if record["split"] == "train":
                        training_deviations.append(record["geometricDeviation"])
            database.commit()
        finally:
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

        if counters["inputRows"] == 0 or counters["acceptedRows"] == 0:
            raise SwipeDataError("swipe source produced no accepted examples")
        rejected_fraction = counters["rejectedRows"] / counters["inputRows"]
        if rejected_fraction > policy.maximum_rejected_fraction:
            raise SwipeDataError(
                f"swipe rejection fraction {rejected_fraction:.6f} exceeds {policy.maximum_rejected_fraction:.6f}"
            )
        if any(accepted_by_split[split] == 0 for split in SPLITS):
            raise SwipeDataError("every session split must contain accepted examples")
        leakage = database.execute(
            "SELECT session_id FROM sessions GROUP BY session_id HAVING COUNT(DISTINCT split) > 1 LIMIT 1"
        ).fetchone()
        if leakage is not None:
            raise SwipeDataError("a collection session crosses generated splits")
        for split in SPLITS:
            session_counts[split] = database.execute(
                "SELECT COUNT(DISTINCT session_id) FROM sessions WHERE split = ?", (split,)
            ).fetchone()[0]
        database.close()

        sloppy_threshold = _nearest_rank(training_deviations, policy.sloppy_percentile)
        very_sloppy_threshold = _nearest_rank(training_deviations, policy.very_sloppy_percentile)
        generated_dir = staging / "generated"
        generated_dir.mkdir()
        output_details: dict[str, dict[str, Any]] = {}
        strata_counts: dict[str, dict[str, int]] = {}
        for split in SPLITS:
            filename = f"{split}.jsonl"
            counts, details = _finalize_split(
                staged_paths[split],
                generated_dir / filename,
                sloppy_threshold,
                very_sloppy_threshold,
            )
            details["records"] = accepted_by_split[split]
            details["sessions"] = session_counts[split]
            output_details[filename] = details
            strata_counts[split] = counts

        layout_output = generated_dir / "layout.json"
        _write_layout(layout_output, layout, policy)
        output_details["layout.json"] = {
            "bytes": layout_output.stat().st_size,
            "sha256": model_sources.file_sha256(layout_output),
        }
        rejections = {
            key.removeprefix("rejection:"): value
            for key, value in sorted(counters.items())
            if key.startswith("rejection:")
        }
        report = {
            "schemaVersion": 1,
            "sourceManifestSha256": manifest.sha256,
            "policySha256": policy.sha256,
            "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
            "source": {
                "id": source.identifier,
                "revision": source.revision,
                "license": source.license,
            },
            "splitPolicy": {
                "method": "sha256-session-basis-points-v1",
                "saltSha256": hashlib.sha256(policy.split_salt.encode("utf-8")).hexdigest(),
                "basisPoints": policy.split_basis_points,
            },
            "counts": {
                "inputRows": counters["inputRows"],
                "acceptedRows": counters["acceptedRows"],
                "duplicateRows": counters["duplicateRows"],
                "rejectedRows": counters["rejectedRows"],
                "rejectedFraction": rejected_fraction,
                "rejections": rejections,
            },
            "sloppiness": {
                "metric": "mean-target-polyline-distance-plus-half-endpoint-error-v1",
                "sloppyThreshold": sloppy_threshold,
                "verySloppyThreshold": very_sloppy_threshold,
                "thresholdPopulation": "train",
            },
            "strata": strata_counts,
            "outputs": output_details,
        }
        manifest_output = generated_dir / "split-manifest.json"
        manifest_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        output_root.mkdir(parents=True, exist_ok=True)
        for path in generated_dir.iterdir():
            os.replace(path, output_root / path.name)
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=model_sources.DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--source-root", type=pathlib.Path, default=model_sources.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = prepare(
            source_root=args.source_root,
            output_root=args.output_root,
            manifest_path=args.manifest,
            policy_path=args.policy,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (SwipeDataError, model_sources.ModelSourceError) as failure:
        print(f"swipe data error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
