#!/usr/bin/env python3
"""Convert the pinned Google TSI corpus into honest word-level tap-error rows."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable

import model_sources


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_ID = "google-tsi-tap-dataset-v1"
DEFAULT_SOURCE_ROOT = ROOT / "build" / "model-sources"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "evaluation-sources" / "google-tsi-tap-v1"
DEFAULT_SOURCE_MANIFEST = ROOT / "models" / "evaluation" / "sources-v1.json"
PROMPT_FIELDS = ("participant_id", "task_id", "trial_id", "prompt_type", "prompt")
TOUCH_FIELDS = (
    "participant_id", "task_id", "trial_id", "timestamp_ms", "ref_char",
    "ref_char_index_in_prompt", "first_frame_touch_x", "first_frame_touch_y",
    "first_frame_touch_major", "first_frame_touch_minor", "first_frame_touch_orientation",
    "first_frame_touch_heatmap", "first_frame_heatmap_overlap_vector", "was_deleted", "lm_scores",
)
WORD = re.compile(r"[A-Za-z]+")
IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
COLLECTION_PROTOCOL = (
    "Google TSI Study 3 copy-typing on Pixel 6 Pro devices. LibreBoard uses only publisher-aligned "
    "reference characters, touch centroids, timestamps, prompts, and keyboard geometry; heatmaps, "
    "ellipse features, submitted strings, and publisher language-model scores are not consumed."
)
RIGHTS_STATEMENT = (
    "The publisher collected the data from recruited study participants and released these "
    "pseudonymous touch records under CC BY 4.0; its public README does not reproduce the "
    "participant-consent form."
)


class TsiDataError(ValueError):
    pass


@dataclass(frozen=True)
class Touch:
    index: int
    timestamp_ms: int
    intended: str
    x: float
    y: float
    deleted: bool


@dataclass(frozen=True)
class Keyboard:
    width: float
    height: float
    centers: dict[str, tuple[float, float]]


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as failure:
        raise TsiDataError(f"{name} is not numeric") from failure
    if not math.isfinite(number):
        raise TsiDataError(f"{name} is not finite")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as failure:
        raise TsiDataError(f"{name} is not an integer") from failure
    if str(number) != str(value) or not minimum <= number <= maximum:
        raise TsiDataError(f"{name} is outside its permitted range")
    return number


def load_keyboard(path: pathlib.Path) -> Keyboard:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024:
            raise TsiDataError("TSI keyboard geometry is missing, linked, or too large")
        raw = json.loads(path.read_bytes())
    except TsiDataError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise TsiDataError(f"cannot read TSI keyboard geometry: {failure}") from failure
    if not isinstance(raw, dict) or set(raw) != {"device_info", "keyboard_info", "keys_info"}:
        raise TsiDataError("TSI keyboard geometry has an unexpected schema")
    keyboard_info = raw["keyboard_info"]
    keys_info = raw["keys_info"]
    if not isinstance(keyboard_info, dict) or not isinstance(keys_info, dict):
        raise TsiDataError("TSI keyboard geometry sections are invalid")
    width = _finite(keyboard_info.get("keyboard_width"), "keyboard width")
    height = _finite(keyboard_info.get("keyboard_height"), "keyboard height")
    if not 1 <= width <= 10_000 or not 1 <= height <= 10_000:
        raise TsiDataError("TSI keyboard dimensions are invalid")
    expected_keys = LETTERS | {".", "SPACE"}
    if set(keys_info) != expected_keys:
        raise TsiDataError("TSI keyboard key inventory is invalid")
    centers = {}
    for key, details in keys_info.items():
        if not isinstance(details, dict):
            raise TsiDataError(f"TSI keyboard key {key} is invalid")
        x = _finite(details.get("key_center_x"), f"{key} center x")
        y = _finite(details.get("key_center_y"), f"{key} center y")
        if not 0 <= x <= width or not 0 <= y <= height:
            raise TsiDataError(f"TSI keyboard key {key} is outside the layout")
        centers[key] = (x, y)
    return Keyboard(width=width, height=height, centers=centers)


def _require_header(reader: csv.DictReader, expected: tuple[str, ...], name: str) -> None:
    if tuple(reader.fieldnames or ()) != expected:
        raise TsiDataError(f"{name} has an unexpected header")


def _trial_key(row: dict[str, str], location: str) -> tuple[str, str, int]:
    participant = row.get("participant_id", "")
    task = row.get("task_id", "")
    if not IDENTIFIER.fullmatch(participant) or not IDENTIFIER.fullmatch(task):
        raise TsiDataError(f"{location} has an invalid participant or task id")
    trial = _integer(row.get("trial_id"), f"{location} trial_id", 0, 10_000)
    return participant, task, trial


def load_prompts(path: pathlib.Path) -> dict[tuple[str, str, int], dict[str, str]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
        raise TsiDataError("TSI prompt data is missing, linked, or too large")
    prompts = {}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            _require_header(reader, PROMPT_FIELDS, "TSI prompt data")
            for line_number, row in enumerate(reader, 2):
                key = _trial_key(row, f"prompt line {line_number}")
                prompt_type = row.get("prompt_type")
                prompt = row.get("prompt")
                if prompt_type not in {"phrase", "random"}:
                    raise TsiDataError(f"prompt line {line_number} has an invalid type")
                if not isinstance(prompt, str) or not prompt or len(prompt) > 1024:
                    raise TsiDataError(f"prompt line {line_number} has invalid text")
                if key in prompts:
                    raise TsiDataError("TSI prompt data contains duplicate trial keys")
                prompts[key] = {"type": prompt_type, "text": prompt}
    except (OSError, csv.Error, UnicodeError) as failure:
        raise TsiDataError(f"cannot read TSI prompt data: {failure}") from failure
    if not prompts:
        raise TsiDataError("TSI prompt data is empty")
    return prompts


def load_touches(
    path: pathlib.Path,
    prompts: dict[tuple[str, str, int], dict[str, str]],
    keyboard: Keyboard,
) -> tuple[dict[tuple[str, str, int], list[Touch]], collections.Counter[str]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 256 * 1024 * 1024:
        raise TsiDataError("TSI touch data is missing, linked, or too large")
    trials: dict[tuple[str, str, int], list[Touch]] = collections.defaultdict(list)
    counts: collections.Counter[str] = collections.Counter()
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            _require_header(reader, TOUCH_FIELDS, "TSI touch data")
            for line_number, row in enumerate(reader, 2):
                if line_number > 1_000_001:
                    raise TsiDataError("TSI touch data exceeds the row limit")
                key = _trial_key(row, f"touch line {line_number}")
                prompt = prompts.get(key)
                if prompt is None:
                    raise TsiDataError(f"touch line {line_number} has no matching prompt")
                index = _integer(
                    row.get("ref_char_index_in_prompt"),
                    f"touch line {line_number} reference index",
                    0,
                    len(prompt["text"]) - 1,
                )
                intended = row.get("ref_char", "")
                intended = " " if intended == "SPACE" else intended.lower()
                if intended not in LETTERS | {" ", "."}:
                    raise TsiDataError(f"touch line {line_number} has an invalid reference character")
                if prompt["text"][index].lower() != intended:
                    raise TsiDataError(f"touch line {line_number} disagrees with its prompt")
                timestamp = _integer(
                    row.get("timestamp_ms"), f"touch line {line_number} timestamp", 0, 2**63 - 1
                )
                x = _finite(row.get("first_frame_touch_x"), f"touch line {line_number} x")
                y = _finite(row.get("first_frame_touch_y"), f"touch line {line_number} y")
                if not -0.5 * keyboard.width <= x <= 1.5 * keyboard.width or not -0.5 * keyboard.height <= y <= 1.5 * keyboard.height:
                    raise TsiDataError(f"touch line {line_number} is outside the bounded coordinate envelope")
                deleted_value = row.get("was_deleted")
                if deleted_value not in {"True", "False"}:
                    raise TsiDataError(f"touch line {line_number} has an invalid deletion flag")
                deleted = deleted_value == "True"
                trials[key].append(Touch(index, timestamp, intended, x, y, deleted))
                counts["touches"] += 1
                counts["deletedTouches"] += deleted
                counts["outsideLayoutTouches"] += not (0 <= x <= keyboard.width and 0 <= y <= keyboard.height)
                counts[f"promptType:{prompt['type']}"] += 1
    except (OSError, csv.Error, UnicodeError) as failure:
        raise TsiDataError(f"cannot read TSI touch data: {failure}") from failure
    if not trials:
        raise TsiDataError("TSI touch data is empty")
    return dict(trials), counts


def nearest_key(touch: Touch, keyboard: Keyboard) -> str:
    return min(
        keyboard.centers,
        key=lambda key: (
            (touch.x - keyboard.centers[key][0]) ** 2 +
            (touch.y - keyboard.centers[key][1]) ** 2,
            key,
        ),
    )


def convert_records(
    prompts: dict[tuple[str, str, int], dict[str, str]],
    trials: dict[tuple[str, str, int], list[Touch]],
    keyboard: Keyboard,
) -> tuple[list[dict[str, Any]], collections.Counter[str]]:
    records = []
    counts: collections.Counter[str] = collections.Counter()
    for trial_key in sorted(prompts):
        prompt = prompts[trial_key]
        if prompt["type"] != "phrase":
            counts["excludedRandomTrials"] += 1
            continue
        by_index: dict[int, list[Touch]] = collections.defaultdict(list)
        for touch in trials.get(trial_key, []):
            by_index[touch.index].append(touch)
        participant, task, trial = trial_key
        for match in WORD.finditer(prompt["text"]):
            counts["phraseWords"] += 1
            touches = []
            for index in range(match.start(), match.end()):
                available = sorted(by_index.get(index, ()), key=lambda value: (value.deleted, value.timestamp_ms))
                if not available or available[0].deleted:
                    touches = []
                    break
                touches.append(available[0])
            if not touches:
                counts["excludedIncompleteWords"] += 1
                continue
            chronological_touches = sorted(touches, key=lambda value: value.timestamp_ms)
            decoded = [nearest_key(touch, keyboard) for touch in chronological_touches]
            if any(key not in LETTERS for key in decoded):
                counts["excludedNonLetterNearestKey"] += 1
                continue
            target = match.group().lower()
            observed = "".join(decoded)
            if observed == target:
                counts["exactWords"] += 1
                continue
            first_timestamp = chronological_touches[0].timestamp_ms
            points = []
            for touch in chronological_touches:
                relative_time = touch.timestamp_ms - first_timestamp
                if not 0 <= relative_time <= 60_000:
                    points = []
                    break
                points.append({
                    "x": touch.x / keyboard.width,
                    "y": touch.y / keyboard.height,
                    "timeMillis": relative_time,
                })
            if not points:
                counts["excludedInvalidTiming"] += 1
                continue
            records.append({
                "schemaVersion": 1,
                "id": f"google-tsi:{participant}:{task}:{trial}:{match.start()}:{match.end()}",
                "sessionId": f"google-tsi:{participant}",
                "category": "tap_error",
                "target": target,
                "raw": observed,
                "languageTag": "en-US",
                "precedingContext": prompt["text"][:match.start()][-256:],
                "fieldClass": "plain",
                "collectionMethod": "human_replay",
                "touchPoints": points,
            })
            counts["tapErrorWords"] += 1
    return records, counts


def prepare(
    source_root: pathlib.Path = DEFAULT_SOURCE_ROOT,
    output_root: pathlib.Path = DEFAULT_OUTPUT_ROOT,
    source_manifest_path: pathlib.Path = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, Any]:
    manifest = model_sources.load_manifest(source_manifest_path)
    source = manifest.source(SOURCE_ID)
    if source.kind != "dataset" or source.license != "CC-BY-4.0":
        raise TsiDataError("the pinned TSI source kind or license is invalid")
    source_root = source_root.resolve()
    model_sources.verify_source(source_root, source)
    base = source_root / source.identifier
    keyboard = load_keyboard(base / "keyboard_data.json")
    prompts = load_prompts(base / "prompt_data.csv")
    trials, input_counts = load_touches(base / "touch_data.csv", prompts, keyboard)
    records, conversion_counts = convert_records(prompts, trials, keyboard)
    if not records:
        raise TsiDataError("TSI conversion produced no tap-error words")

    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libreboard-tsi-", dir=output_root.parent) as temporary:
        staging = pathlib.Path(temporary)
        data_path = staging / "tap-errors.jsonl"
        with data_path.open("wb") as stream:
            for record in records:
                stream.write(_canonical(record))
            stream.flush()
            os.fsync(stream.fileno())
        data_sha256 = _sha256(data_path)
        derived_manifest = {
            "schemaVersion": 1,
            "datasetId": SOURCE_ID,
            "dataFile": data_path.name,
            "dataSha256": data_sha256,
            "license": source.license,
            "collectionProtocol": COLLECTION_PROTOCOL,
            "containsHumanContributions": True,
            "consentStatement": RIGHTS_STATEMENT,
        }
        (staging / "source-manifest.json").write_text(
            json.dumps(derived_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = {
            "schemaVersion": 1,
            "source": {
                "id": source.identifier,
                "repository": source.repository,
                "revision": source.revision,
                "license": source.license,
                "sourceManifestSha256": manifest.sha256,
            },
            "inputCounts": dict(sorted(input_counts.items())),
            "conversionCounts": dict(sorted(conversion_counts.items())),
            "output": {
                "file": data_path.name,
                "records": len(records),
                "bytes": data_path.stat().st_size,
                "sha256": data_sha256,
            },
            "limitations": [
                "English phrase prompts only; random strings are excluded.",
                "Only one non-deleted aligned touch per character is used.",
                "The publisher's heatmaps, ellipse features, submitted strings, and language-model scores are ignored.",
                "This source is a component corpus and cannot alone satisfy LibreBoard's 3,000 held-out tap-error gate.",
            ],
        }
        (staging / "conversion-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        for name in (data_path.name, "source-manifest.json", "conversion-report.json"):
            os.replace(staging / name, output_root / name)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=pathlib.Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = prepare(args.source_root, args.output_root, args.source_manifest)
    except (OSError, model_sources.ModelSourceError, TsiDataError) as failure:
        print(f"TSI tap preparation error: {failure}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
