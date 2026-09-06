#!/usr/bin/env python3
"""Prepare a deterministic, privacy-bounded Phase 0 tap evaluation corpus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable

import evaluate_engine


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "evaluation" / "tap-data-policy.json"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "evaluation-data" / "tap-v1"
SPLITS = ("train", "validation", "test")
CATEGORIES = tuple(category for category in evaluate_engine.TAP_CATEGORIES)
LANGUAGES = {"en-US", "de"}
FIELD_CLASSES = {"plain", "short_message", "search"}
COLLECTION_METHODS = {"human_natural", "human_replay", "project_authored"}
LEXICAL_KINDS = {"contraction", "personal", "compound"}
ALLOWED_DATA_LICENSES = {"Apache-2.0", "CC-BY-4.0", "CC0-1.0", "MIT"}
MAXIMUM_CONTEXT_CHARACTERS = 256
INPUT_KEYS = {
    "schemaVersion",
    "id",
    "sessionId",
    "category",
    "target",
    "raw",
    "languageTag",
    "precedingContext",
    "fieldClass",
    "collectionMethod",
    "touchPoints",
    "shouldCorrect",
    "lexicalKind",
    "personalWords",
}
REQUIRED_KEYS = {
    "schemaVersion",
    "id",
    "sessionId",
    "category",
    "target",
    "raw",
    "languageTag",
    "precedingContext",
    "fieldClass",
    "collectionMethod",
    "touchPoints",
}


class TapEvaluationDataError(ValueError):
    pass


@dataclass(frozen=True)
class Policy:
    path: pathlib.Path
    sha256: str
    split_salt: str
    split_basis_points: dict[str, int]
    maximum_line_bytes: int
    maximum_text_codepoints: int
    maximum_context_codepoints: int
    maximum_touch_points: int
    minimum_coordinate: float
    maximum_coordinate: float


@dataclass(frozen=True)
class SourceManifest:
    sha256: str
    dataset_id: str
    license: str
    collection_protocol: str
    contains_human_contributions: bool
    consent_statement: str


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_policy(path: pathlib.Path = DEFAULT_POLICY) -> Policy:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
            raise TapEvaluationDataError("tap evaluation policy is missing, linked, or too large")
        payload = path.read_bytes()
        raw = json.loads(payload)
    except TapEvaluationDataError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise TapEvaluationDataError(f"cannot read tap evaluation policy: {failure}") from failure
    expected = {
        "schemaVersion", "splitSalt", "splitBasisPoints", "maximumLineBytes",
        "maximumTextCodePoints", "maximumContextCodePoints", "maximumTouchPoints",
        "minimumCoordinate", "maximumCoordinate",
    }
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schemaVersion") != 1:
        raise TapEvaluationDataError("tap evaluation policy has an unexpected schema")
    split_salt = raw["splitSalt"]
    if not isinstance(split_salt, str) or not 16 <= len(split_salt) <= 128:
        raise TapEvaluationDataError("tap evaluation split salt is invalid")
    split_points = raw["splitBasisPoints"]
    if (
        not isinstance(split_points, dict)
        or set(split_points) != set(SPLITS)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in split_points.values())
        or sum(split_points.values()) != 10_000
    ):
        raise TapEvaluationDataError("tap evaluation split basis points are invalid")
    integers = {}
    for name, minimum, maximum in (
        ("maximumLineBytes", 1024, 1024 * 1024),
        ("maximumTextCodePoints", 1, 1024),
        ("maximumContextCodePoints", 1, MAXIMUM_CONTEXT_CHARACTERS),
        ("maximumTouchPoints", 1, 1024),
    ):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise TapEvaluationDataError(f"tap evaluation policy has invalid {name}")
        integers[name] = value
    minimum_coordinate = _finite(raw["minimumCoordinate"])
    maximum_coordinate = _finite(raw["maximumCoordinate"])
    if minimum_coordinate is None or maximum_coordinate is None or not minimum_coordinate < 0 < 1 < maximum_coordinate:
        raise TapEvaluationDataError("tap evaluation coordinate bounds are invalid")
    return Policy(
        path=path.resolve(),
        sha256=hashlib.sha256(payload).hexdigest(),
        split_salt=split_salt,
        split_basis_points=dict(split_points),
        maximum_line_bytes=integers["maximumLineBytes"],
        maximum_text_codepoints=integers["maximumTextCodePoints"],
        maximum_context_codepoints=integers["maximumContextCodePoints"],
        maximum_touch_points=integers["maximumTouchPoints"],
        minimum_coordinate=minimum_coordinate,
        maximum_coordinate=maximum_coordinate,
    )


def load_source_manifest(path: pathlib.Path, source: pathlib.Path) -> SourceManifest:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
            raise TapEvaluationDataError("tap source manifest is missing, linked, or too large")
        payload = path.read_bytes()
        raw = json.loads(payload)
    except TapEvaluationDataError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise TapEvaluationDataError(f"cannot read tap source manifest: {failure}") from failure
    expected = {
        "schemaVersion",
        "datasetId",
        "dataFile",
        "dataSha256",
        "license",
        "collectionProtocol",
        "containsHumanContributions",
        "consentStatement",
    }
    if not isinstance(raw, dict) or set(raw) != expected or raw.get("schemaVersion") != 1:
        raise TapEvaluationDataError("tap source manifest has an unexpected schema")
    dataset_id = raw["datasetId"]
    data_file = raw["dataFile"]
    data_hash = raw["dataSha256"]
    license_name = raw["license"]
    protocol = raw["collectionProtocol"]
    human = raw["containsHumanContributions"]
    consent = raw["consentStatement"]
    if not isinstance(dataset_id, str) or not dataset_id or len(dataset_id) > 128:
        raise TapEvaluationDataError("tap source manifest has an invalid datasetId")
    if not isinstance(data_file, str) or pathlib.PurePosixPath(data_file).name != data_file or data_file != source.name:
        raise TapEvaluationDataError("tap source manifest does not identify the source file")
    if not isinstance(data_hash, str) or not evaluate_engine.SHA256.fullmatch(data_hash) or data_hash != _sha256(source):
        raise TapEvaluationDataError("tap source manifest data hash does not match")
    if not isinstance(license_name, str) or license_name not in ALLOWED_DATA_LICENSES:
        raise TapEvaluationDataError("tap source manifest license is not permitted")
    if not isinstance(protocol, str) or not protocol or len(protocol) > 1024:
        raise TapEvaluationDataError("tap source manifest has an invalid collection protocol")
    if not isinstance(human, bool) or not isinstance(consent, str) or len(consent) > 1024:
        raise TapEvaluationDataError("tap source manifest has invalid consent metadata")
    if human and not consent:
        raise TapEvaluationDataError("human tap data requires an explicit consent statement")
    if not human and consent:
        raise TapEvaluationDataError("project-authored tap data must not claim participant consent")
    return SourceManifest(
        sha256=hashlib.sha256(payload).hexdigest(),
        dataset_id=dataset_id,
        license=license_name,
        collection_protocol=protocol,
        contains_human_contributions=human,
        consent_statement=consent,
    )


def session_hash(session_id: str, policy: Policy) -> str:
    return hashlib.sha256((policy.split_salt + "\0" + session_id).encode()).hexdigest()


def split_for_session(session_id: str, policy: Policy) -> str:
    bucket = int(session_hash(session_id, policy)[:8], 16) % 10_000
    cursor = 0
    for split in SPLITS:
        cursor += policy.split_basis_points[split]
        if bucket < cursor:
            return split
    raise AssertionError("split policy did not cover the hash space")


def _bounded_string(raw: dict[str, Any], field: str, maximum: int, *, allow_empty: bool = False) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or (not allow_empty and not value) or len(value) > maximum:
        raise TapEvaluationDataError(f"{field} must be a bounded string")
    return value


def normalize_record(raw: Any, line_number: int, policy: Policy) -> dict[str, Any]:
    location = f"line {line_number}"
    if not isinstance(raw, dict) or set(raw) - INPUT_KEYS or not REQUIRED_KEYS <= set(raw):
        raise TapEvaluationDataError(f"{location}: tap row has an unexpected schema")
    if raw.get("schemaVersion") != 1:
        raise TapEvaluationDataError(f"{location}: unsupported schemaVersion")
    identifier = _bounded_string(raw, "id", 256)
    session_id = _bounded_string(raw, "sessionId", 256)
    category = raw.get("category")
    if category not in CATEGORIES:
        raise TapEvaluationDataError(f"{location}: unsupported category")
    target = _bounded_string(raw, "target", policy.maximum_text_codepoints)
    observed = _bounded_string(raw, "raw", policy.maximum_text_codepoints)
    context = _bounded_string(
        raw,
        "precedingContext",
        policy.maximum_context_codepoints,
        allow_empty=True,
    )
    language = raw.get("languageTag")
    field_class = raw.get("fieldClass")
    collection_method = raw.get("collectionMethod")
    if (
        not isinstance(language, str) or language not in LANGUAGES
        or not isinstance(field_class, str) or field_class not in FIELD_CLASSES
        or not isinstance(collection_method, str) or collection_method not in COLLECTION_METHODS
    ):
        raise TapEvaluationDataError(f"{location}: unsupported language, field class, or collection method")

    touch_points_raw = raw.get("touchPoints")
    if not isinstance(touch_points_raw, list) or len(touch_points_raw) > policy.maximum_touch_points:
        raise TapEvaluationDataError(f"{location}: touchPoints are invalid")
    touch_points = []
    previous_time = 0
    for point in touch_points_raw:
        if not isinstance(point, dict) or set(point) != {"x", "y", "timeMillis"}:
            raise TapEvaluationDataError(f"{location}: touch point has an unexpected schema")
        x = _finite(point.get("x"))
        y = _finite(point.get("y"))
        time_millis = point.get("timeMillis")
        if (
            x is None or y is None
            or not policy.minimum_coordinate <= x <= policy.maximum_coordinate
            or not policy.minimum_coordinate <= y <= policy.maximum_coordinate
            or isinstance(time_millis, bool) or not isinstance(time_millis, int)
            or not previous_time <= time_millis <= 60_000
        ):
            raise TapEvaluationDataError(f"{location}: touch point is out of bounds")
        previous_time = time_millis
        touch_points.append({"x": x, "y": y, "timeMillis": time_millis})
    if category == "tap_error" and (collection_method == "project_authored" or not touch_points):
        raise TapEvaluationDataError(f"{location}: spatial tap errors require human touch data")

    should_correct = raw.get("shouldCorrect")
    if category == "valid_word":
        if not isinstance(should_correct, bool):
            raise TapEvaluationDataError(f"{location}: valid_word requires shouldCorrect")
    elif "shouldCorrect" in raw:
        raise TapEvaluationDataError(f"{location}: shouldCorrect is valid only for valid_word")

    lexical_kind = raw.get("lexicalKind")
    if category == "lexical":
        if not isinstance(lexical_kind, str) or lexical_kind not in LEXICAL_KINDS:
            raise TapEvaluationDataError(f"{location}: lexical row requires lexicalKind")
    elif "lexicalKind" in raw:
        raise TapEvaluationDataError(f"{location}: lexicalKind is valid only for lexical rows")

    personal_words_raw = raw.get("personalWords", [])
    if (
        not isinstance(personal_words_raw, list)
        or len(personal_words_raw) > 16
        or any(not isinstance(word, str) or not word or len(word) > 64 for word in personal_words_raw)
    ):
        raise TapEvaluationDataError(f"{location}: personalWords are invalid")
    personal_words = list(dict.fromkeys(personal_words_raw))
    if lexical_kind == "personal" and not personal_words:
        raise TapEvaluationDataError(f"{location}: personal lexical rows require personalWords")
    if lexical_kind != "personal" and personal_words:
        raise TapEvaluationDataError(f"{location}: personalWords are valid only for personal lexical rows")

    hashed_session = session_hash(session_id, policy)
    split = split_for_session(session_id, policy)
    record = {
        "schemaVersion": 1,
        "id": hashlib.sha256((identifier + "\0" + hashed_session).encode()).hexdigest(),
        "sessionId": hashed_session,
        "split": split,
        "category": category,
        "target": target,
        "raw": observed,
        "languageTag": language,
        "precedingContext": context,
        "fieldClass": field_class,
        "collectionMethod": collection_method,
        "touchPoints": touch_points,
    }
    if should_correct is not None:
        record["shouldCorrect"] = should_correct
    if lexical_kind is not None:
        record["lexicalKind"] = lexical_kind
    if personal_words:
        record["personalWords"] = personal_words
    return record


def _records(path: pathlib.Path, policy: Policy) -> Iterable[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise TapEvaluationDataError("tap evaluation source must be a regular file")
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(policy.maximum_line_bytes + 1)
            if not line:
                break
            line_number += 1
            if len(line) > policy.maximum_line_bytes:
                raise TapEvaluationDataError(f"line {line_number}: source line is too large")
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise TapEvaluationDataError(f"line {line_number}: invalid JSON") from failure
            yield normalize_record(raw, line_number, policy)


def prepare(
    source: pathlib.Path,
    source_manifest_path: pathlib.Path,
    output_root: pathlib.Path,
    policy_path: pathlib.Path = DEFAULT_POLICY,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    output_root = output_root.resolve()
    policy = load_policy(policy_path)
    source_manifest = load_source_manifest(source_manifest_path.resolve(), source)
    rows = list(_records(source, policy))
    if not rows:
        raise TapEvaluationDataError("tap evaluation source is empty")
    identifiers = collections.Counter(row["id"] for row in rows)
    duplicates = [identifier for identifier, count in identifiers.items() if count > 1]
    if duplicates:
        raise TapEvaluationDataError("tap evaluation source contains duplicate IDs")
    has_human_rows = any(row["collectionMethod"] != "project_authored" for row in rows)
    if has_human_rows != source_manifest.contains_human_contributions:
        raise TapEvaluationDataError("tap source rows disagree with human-contribution metadata")

    counts = {split: collections.Counter() for split in SPLITS}
    lexical_counts = collections.Counter()
    valid_word_counts = collections.Counter()
    for row in rows:
        counts[row["split"]][row["category"]] += 1
        if row["split"] == "test" and row["category"] == "lexical":
            lexical_counts[row["lexicalKind"]] += 1
        if row["split"] == "test" and row["category"] == "valid_word":
            valid_word_counts["correct" if row["shouldCorrect"] else "keep"] += 1

    release_errors = []
    for split in SPLITS:
        if sum(counts[split].values()) == 0:
            release_errors.append(f"split:{split}=0")
    for category in CATEGORIES:
        minimum = evaluate_engine.MINIMUM_COUNTS[category]
        if counts["test"][category] < minimum:
            release_errors.append(f"{category}={counts['test'][category]}<{minimum}")
    for name, minimum in (
        ("correct", evaluate_engine.MINIMUM_VALID_WORD_CORRECTIONS),
        ("keep", evaluate_engine.MINIMUM_VALID_WORD_KEEPS),
    ):
        if valid_word_counts[name] < minimum:
            release_errors.append(f"valid_word:{name}={valid_word_counts[name]}<{minimum}")
    for kind in LEXICAL_KINDS:
        if lexical_counts[kind] == 0:
            release_errors.append(f"lexical:{kind}=0")
    if release_errors and not allow_small:
        raise TapEvaluationDataError("tap evaluation release minimums not met: " + ", ".join(release_errors))

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libreboard-tap-eval-", dir=output_root.parent) as temporary:
        staging = pathlib.Path(temporary)
        outputs = {}
        for split in SPLITS:
            path = staging / f"{split}.jsonl"
            with path.open("wb") as stream:
                for row in rows:
                    if row["split"] == split:
                        stream.write(_canonical(row))
                stream.flush()
                os.fsync(stream.fileno())
            outputs[path.name] = {
                "records": sum(counts[split].values()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        report = {
            "schemaVersion": 1,
            "releaseEligible": not release_errors,
            "source": {
                "datasetId": source_manifest.dataset_id,
                "file": source.name,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "license": source_manifest.license,
                "collectionProtocol": source_manifest.collection_protocol,
                "containsHumanContributions": source_manifest.contains_human_contributions,
                "consentStatement": source_manifest.consent_statement,
                "manifestSha256": source_manifest.sha256,
            },
            "policySha256": policy.sha256,
            "counts": {split: dict(sorted(value.items())) for split, value in counts.items()},
            "testValidWordCounts": dict(sorted(valid_word_counts.items())),
            "testLexicalCounts": dict(sorted(lexical_counts.items())),
            "releaseErrors": release_errors,
            "outputs": outputs,
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_root.mkdir(parents=True, exist_ok=True)
        for name in (*[f"{split}.jsonl" for split in SPLITS], "manifest.json"):
            os.replace(staging / name, output_root / name)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path, help="consented/project-authored source JSONL")
    parser.add_argument("--source-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--allow-small", action="store_true", help="prepare fixtures without release eligibility")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = prepare(
            args.source,
            args.source_manifest,
            args.output_root,
            args.policy,
            allow_small=args.allow_small,
        )
    except (OSError, TapEvaluationDataError) as failure:
        print(f"tap evaluation data error: {failure}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
