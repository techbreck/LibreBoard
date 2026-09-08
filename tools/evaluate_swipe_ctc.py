#!/usr/bin/env python3
"""Evaluate an exported LibreBoard swipe CTC model with the production beam semantics.

This is an offline model diagnostic. Vocabulary comes from earlier corpus splits or an
APK-bound native dictionary export; validation and final test paths are explicitly separated. It mirrors the CTC beam, geometric template cost, normalized decoder union, and
static scorer, but it does not replace Phase 0 device evidence: the production dictionary, complete
shared scorer, Android runtime, latency, and memory still have to be measured there.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
import time
import unicodedata
import zipfile
from typing import Any, Iterable, Sequence

import model_sources
import swipe_model_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "build" / "model-data" / "swipe-latin-v1"
DEFAULT_EXPORT_ROOT = ROOT / "build" / "model-export" / "swipe-latin-v1"
DEFAULT_REPORT = DEFAULT_EXPORT_ROOT / "export-report.json"
DEFAULT_OUTPUT = DEFAULT_EXPORT_ROOT / "ctc-evaluation-report.json"
REQUIRED_STRATA = (
    "short",
    "medium",
    "long",
    "clean",
    "sloppy",
    "very_sloppy",
    "double_letter",
    "return_trip",
)
TEST_ROW_KEYS = {
    "schemaVersion",
    "sessionId",
    "id",
    "split",
    "language",
    "target",
    "ctcLabels",
    "pathCoordinates",
    "layoutId",
    "orientation",
    "geometricDeviation",
    "strata",
}
LAYOUT_KEYS = {
    "schemaVersion",
    "id",
    "keyLabels",
    "keyCenters",
    "keyMask",
    "pathShape",
    "keyCentersShape",
    "keyMaskShape",
}
EXPORT_REPORT_KEYS = {
    "schemaVersion",
    "modelId",
    "releaseEligible",
    "appCommit",
    "modelSpecSha256",
    "dataManifestSha256",
    "trainingReportSha256",
    "weightsSha256",
    "model",
    "manifest",
    "requiredOperators",
    "modelSpec",
    "splitManifest",
    "trainingReport",
    "fp16StoredInitializerCount",
    "toolchain",
}
LOG_ZERO = float("-inf")
MAXIMUM_JSONL_LINE_BYTES = 64 * 1024
MAXIMUM_LEXICON_WORDS = 100_000
LENGTH_TOLERANCE_BELOW = 3
LENGTH_TOLERANCE_ABOVE = 4


class SwipeEvaluationError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class EvaluationRow:
    identifier: str
    session_id: str
    language: str
    target: str
    labels: tuple[int, ...]
    path: tuple[float, ...]
    strata: frozenset[str]


@dataclasses.dataclass(frozen=True)
class LexiconEntry:
    word: str
    language: str
    emissions: tuple[int, ...]
    frequency: int


@dataclasses.dataclass(frozen=True)
class ScoredLexiconEntry:
    entry: LexiconEntry
    spatial: float

    @property
    def word(self) -> str:
        return self.entry.word


@dataclasses.dataclass(frozen=True)
class GeometricLexiconIndex:
    words: tuple[LexiconEntry, ...]
    templates: Any
    sequences: Any
    sequence_lengths: Any
    turn_counts: Any
    word_ids: Any
    buckets: dict[tuple[str, int], Any]
    enabled_centers: Any
    enabled_classes: Any
    key_width: float
    key_height: float


class TrieNode:
    __slots__ = ("children", "words")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.words: list[LexiconEntry] = []


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).lower()


def _read_json(path: pathlib.Path, maximum_bytes: int, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum_bytes:
            raise SwipeEvaluationError(f"{label} is missing, linked, or too large")
        payload = path.read_bytes()
        value = json.loads(payload)
    except SwipeEvaluationError:
        raise
    except (OSError, json.JSONDecodeError) as failure:
        raise SwipeEvaluationError(f"cannot read {label}: {failure}") from failure
    if not isinstance(value, dict):
        raise SwipeEvaluationError(f"{label} must be an object")
    return value, payload


def _safe_child(root: pathlib.Path, filename: Any, label: str) -> pathlib.Path:
    if not isinstance(filename, str) or not filename or pathlib.PurePath(filename).name != filename:
        raise SwipeEvaluationError(f"{label} has an invalid artifact name")
    path = root / filename
    if not path.is_file() or path.is_symlink():
        raise SwipeEvaluationError(f"{label} artifact is missing or linked")
    return path


def load_export(report_path: pathlib.Path, *, development: bool) -> tuple[dict[str, Any], pathlib.Path, str]:
    report_path = report_path.resolve()
    report, report_payload = _read_json(report_path, 4 * 1024 * 1024, "swipe export report")
    if set(report) != EXPORT_REPORT_KEYS or report.get("schemaVersion") != 1 or report.get("modelId") != "swipe-latin-v1":
        raise SwipeEvaluationError("swipe export report has an unsupported schema or model")
    if not isinstance(report.get("appCommit"), str) or not model_sources.REVISION.fullmatch(report["appCommit"]):
        raise SwipeEvaluationError("swipe export report has an invalid app commit")
    if report.get("releaseEligible") is not True and not development:
        raise SwipeEvaluationError("a development export requires --development")
    model = report.get("model")
    if not isinstance(model, dict) or set(model) != {"file", "bytes", "sha256"}:
        raise SwipeEvaluationError("swipe export report has invalid model metadata")
    model_path = _safe_child(report_path.parent, model["file"], "swipe model")
    if (
        isinstance(model["bytes"], bool)
        or not isinstance(model["bytes"], int)
        or model_path.stat().st_size != model["bytes"]
        or not isinstance(model["sha256"], str)
        or not model_sources.SHA256.fullmatch(model["sha256"])
        or model_sources.file_sha256(model_path) != model["sha256"]
    ):
        raise SwipeEvaluationError("swipe model does not match its export report")
    manifest_path = _safe_child(report_path.parent, report.get("manifest"), "model manifest")
    manifest, _ = _read_json(manifest_path, 1024 * 1024, "model manifest")
    if manifest.get("modelKind") != "swipe-ctc" or manifest.get("modelSha256") != model["sha256"]:
        raise SwipeEvaluationError("model manifest does not bind the exported swipe model")

    for field in ("modelSpecSha256", "dataManifestSha256", "trainingReportSha256", "weightsSha256"):
        if not isinstance(report.get(field), str) or not model_sources.SHA256.fullmatch(report[field]):
            raise SwipeEvaluationError(f"swipe export report has an invalid {field}")
    model_spec_path = _safe_child(report_path.parent, report["modelSpec"], "copied model spec")
    split_manifest_path = _safe_child(report_path.parent, report["splitManifest"], "copied split manifest")
    training_report_path = _safe_child(report_path.parent, report["trainingReport"], "copied training report")
    operators_path = _safe_child(report_path.parent, report["requiredOperators"], "operator config")
    if model_sources.file_sha256(model_spec_path) != report["modelSpecSha256"]:
        raise SwipeEvaluationError("copied model spec does not match the export report")
    if model_sources.file_sha256(split_manifest_path) != report["dataManifestSha256"]:
        raise SwipeEvaluationError("copied split manifest does not match the export report")
    if model_sources.file_sha256(training_report_path) != report["trainingReportSha256"]:
        raise SwipeEvaluationError("copied training report does not match the export report")
    training_report, _ = _read_json(training_report_path, 4 * 1024 * 1024, "copied training report")
    training_weights = training_report.get("weights")
    if (
        training_report.get("modelId") != "swipe-latin-v1"
        or not isinstance(training_weights, dict)
        or training_weights.get("sha256") != report["weightsSha256"]
    ):
        raise SwipeEvaluationError("copied training report does not bind the exported weights")
    if operators_path.stat().st_size <= 0 or operators_path.stat().st_size > 1024 * 1024:
        raise SwipeEvaluationError("operator config is empty or too large")
    if (
        isinstance(report.get("fp16StoredInitializerCount"), bool)
        or not isinstance(report.get("fp16StoredInitializerCount"), int)
        or report["fp16StoredInitializerCount"] <= 0
    ):
        raise SwipeEvaluationError("swipe export report has an invalid FP16 initializer count")
    if report.get("toolchain") != {
        "torch": "2.8.0",
        "numpy": "2.2.6",
        "onnx": "1.19.0",
        "safetensors": "0.6.2",
    }:
        raise SwipeEvaluationError("swipe export report toolchain does not match the pinned exporter")
    return report, model_path, hashlib.sha256(report_payload).hexdigest()


def load_layout(path: pathlib.Path) -> dict[str, Any]:
    layout, _ = _read_json(path.resolve(), 1024 * 1024, "prepared swipe layout")
    if set(layout) != LAYOUT_KEYS or layout.get("schemaVersion") != 1:
        raise SwipeEvaluationError("prepared swipe layout has an unexpected schema")
    if layout.get("pathShape") != [1, 64, 2] or layout.get("keyCentersShape") != [1, 64, 2]:
        raise SwipeEvaluationError("prepared swipe layout violates the fixed coordinate ABI")
    if layout.get("keyMaskShape") != [1, 64]:
        raise SwipeEvaluationError("prepared swipe layout violates the fixed mask ABI")
    labels = layout.get("keyLabels")
    centers = layout.get("keyCenters")
    mask = layout.get("keyMask")
    if not isinstance(labels, list) or len(labels) != 64:
        raise SwipeEvaluationError("prepared swipe layout has invalid key labels")
    if (
        not isinstance(centers, list)
        or len(centers) != 128
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in centers)
    ):
        raise SwipeEvaluationError("prepared swipe layout has invalid key centers")
    if not isinstance(mask, list) or len(mask) != 64 or any(value not in {0, 1} for value in mask):
        raise SwipeEvaluationError("prepared swipe layout has invalid key mask")
    enabled = [label for label, value in zip(labels, mask, strict=True) if value]
    if not enabled or any(not isinstance(label, str) or len(label) != 1 for label in enabled):
        raise SwipeEvaluationError("prepared swipe layout has invalid enabled labels")
    if len({_normalize(label) for label in enabled}) != len(enabled):
        raise SwipeEvaluationError("prepared swipe layout labels are not unique")
    return layout


def _class_by_label(layout: dict[str, Any]) -> dict[str, int]:
    return {
        _normalize(label): index + 1
        for index, (label, enabled) in enumerate(zip(layout["keyLabels"], layout["keyMask"], strict=True))
        if enabled and label is not None
    }


def gesture_variants(word: str, language: str) -> list[str]:
    german = language.lower().split("-", 1)[0] == "de"
    variants = [""]
    for character in _normalize(word):
        if character in {"'", "\N{RIGHT SINGLE QUOTATION MARK}", "-"}:
            alternatives = [""]
        elif not german:
            alternatives = [character]
        else:
            alternatives = {
                "ä": ["ä", "a"],
                "ö": ["ö", "o"],
                "ü": ["ü", "u"],
                "ß": ["ß", "s", "ss"],
            }.get(character, [character])
        variants = list(dict.fromkeys(
            prefix + alternative
            for prefix in variants
            for alternative in alternatives
        ))[:8]
    return [variant for variant in variants if variant]


def emission_variants(
    word: str,
    class_by_label: dict[str, int],
    language: str,
) -> list[tuple[int, ...]]:
    results = []
    for gesture in gesture_variants(word, language):
        emissions = tuple(class_by_label.get(character, 0) for character in gesture)
        if emissions and 0 not in emissions and len(emissions) <= 64 and emissions not in results:
            results.append(emissions)
    return results


def _parse_test_row(line: bytes, layout: dict[str, Any], split: str = "test") -> EvaluationRow:
    if split not in ("validation", "test"):
        raise SwipeEvaluationError("evaluation split must be validation or test")
    if len(line) > MAXIMUM_JSONL_LINE_BYTES:
        raise SwipeEvaluationError("held-out swipe row is too large")
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise SwipeEvaluationError(f"held-out swipe data contains invalid JSON: {failure}") from failure
    if not isinstance(value, dict) or set(value) != TEST_ROW_KEYS:
        raise SwipeEvaluationError("held-out swipe row has an unexpected schema")
    if value.get("schemaVersion") != 1 or value.get("split") != split or value.get("layoutId") != layout["id"]:
        raise SwipeEvaluationError("held-out swipe row has an incompatible identity")
    identifier = value.get("id")
    session_id = value.get("sessionId")
    if not isinstance(identifier, str) or not model_sources.SHA256.fullmatch(identifier):
        raise SwipeEvaluationError("held-out swipe row has an invalid id")
    if not isinstance(session_id, str) or not model_sources.SHA256.fullmatch(session_id):
        raise SwipeEvaluationError("held-out swipe row has an invalid session id")
    language = value.get("language")
    target = value.get("target")
    labels = value.get("ctcLabels")
    path = value.get("pathCoordinates")
    strata = value.get("strata")
    if not isinstance(language, str) or not language or not isinstance(target, str) or not target:
        raise SwipeEvaluationError("held-out swipe row has an invalid target identity")
    if (
        not isinstance(labels, list)
        or not labels
        or any(isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 64 for item in labels)
    ):
        raise SwipeEvaluationError("held-out swipe row has invalid CTC labels")
    if (
        not isinstance(path, list)
        or len(path) != 128
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in path)
    ):
        raise SwipeEvaluationError("held-out swipe row has an invalid path")
    if (
        not isinstance(strata, list)
        or not strata
        or any(not isinstance(item, str) or item not in REQUIRED_STRATA for item in strata)
        or len(strata) != len(set(strata))
    ):
        raise SwipeEvaluationError("held-out swipe row has invalid strata")
    expected_labels = emission_variants(target, _class_by_label(layout), language)
    if tuple(labels) not in expected_labels:
        raise SwipeEvaluationError("held-out target and CTC labels disagree")
    return EvaluationRow(
        identifier=identifier,
        session_id=session_id,
        language=language,
        target=_normalize(target),
        labels=tuple(labels),
        path=tuple(float(item) for item in path),
        strata=frozenset(strata),
    )


def load_test_rows(path: pathlib.Path, layout: dict[str, Any], split: str = "test") -> list[EvaluationRow]:
    rows = []
    identifiers = set()
    with path.open("rb") as stream:
        for line in stream:
            row = _parse_test_row(line, layout, split)
            if row.identifier in identifiers:
                raise SwipeEvaluationError("held-out swipe row ids are not unique")
            identifiers.add(row.identifier)
            rows.append(row)
    if not rows:
        raise SwipeEvaluationError("held-out swipe data is empty")
    return rows


def select_rows(
    rows: Sequence[EvaluationRow],
    *,
    sample_count: int,
    minimum_per_stratum: int,
) -> list[EvaluationRow]:
    if sample_count <= 0 or minimum_per_stratum <= 0 or sample_count > len(rows):
        raise SwipeEvaluationError("evaluation sample bounds are invalid")
    ordered = sorted(rows, key=lambda row: row.identifier)
    selected: dict[str, EvaluationRow] = {}
    for stratum in REQUIRED_STRATA:
        candidates = [row for row in ordered if stratum in row.strata]
        if len(candidates) < minimum_per_stratum:
            raise SwipeEvaluationError(f"held-out swipe data lacks {minimum_per_stratum} {stratum} rows")
        for row in candidates[:minimum_per_stratum]:
            selected[row.identifier] = row
    if len(selected) > sample_count:
        raise SwipeEvaluationError("requested sample is too small for the stratum guarantees")
    for row in ordered:
        if len(selected) >= sample_count:
            break
        selected.setdefault(row.identifier, row)
    result = sorted(selected.values(), key=lambda row: row.identifier)
    counts = collections.Counter(stratum for row in result for stratum in row.strata)
    if len(result) != sample_count or any(counts[stratum] < minimum_per_stratum for stratum in REQUIRED_STRATA):
        raise SwipeEvaluationError("deterministic selection failed its stratum guarantees")
    return result


def build_lexicon(
    data_root: pathlib.Path, layout: dict[str, Any], *, evaluation_split: str = "test",
) -> dict[str, list[LexiconEntry]]:
    if evaluation_split not in ("validation", "test"):
        raise SwipeEvaluationError("evaluation split must be validation or test")
    frequencies: collections.Counter[tuple[str, str]] = collections.Counter()
    for filename in (("train.jsonl",) if evaluation_split == "validation" else ("train.jsonl", "validation.jsonl")):
        with (data_root / filename).open("rb") as stream:
            for line in stream:
                if len(line) > MAXIMUM_JSONL_LINE_BYTES:
                    raise SwipeEvaluationError(f"{filename} contains an oversized row")
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                    raise SwipeEvaluationError(f"{filename} contains invalid JSON: {failure}") from failure
                if not isinstance(value, dict) or value.get("split") != filename.removesuffix(".jsonl"):
                    raise SwipeEvaluationError(f"{filename} contains an incompatible row")
                language = value.get("language")
                target = value.get("target")
                if not isinstance(language, str) or not language or not isinstance(target, str) or not target:
                    raise SwipeEvaluationError(f"{filename} contains an invalid target")
                frequencies[(_normalize(target), language)] += 1
    surfaces = sorted(
        ((word, language, frequency) for (word, language), frequency in frequencies.items()),
        key=lambda item: (-item[2], item[0], item[1]),
    )[:MAXIMUM_LEXICON_WORDS]
    return lexicon_entries(surfaces, layout)


def lexicon_entries(surfaces, layout: dict[str, Any]) -> dict[str, list[LexiconEntry]]:
    class_by_label = _class_by_label(layout)
    entries = []
    for word, language, frequency in surfaces:
        for emissions in emission_variants(word, class_by_label, language):
            entries.append(LexiconEntry(word, language, emissions, frequency))
    entries.sort(key=lambda entry: (-entry.frequency, entry.word, entry.language))
    by_language: dict[str, list[LexiconEntry]] = collections.defaultdict(list)
    for entry in entries:
        by_language[entry.language].append(entry)
    if not by_language:
        raise SwipeEvaluationError("training vocabulary produced no decodable lexicon entries")
    return dict(by_language)


def load_dictionary_lexicon(path: pathlib.Path, apk: pathlib.Path, layout: dict[str, Any]):
    """Read an instrumented native vocabulary dump and verify its originating APK."""
    if path.stat().st_size > 32 * 1024 * 1024:
        raise SwipeEvaluationError("dictionary lexicon export exceeds its size bound")
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as failure:
        raise SwipeEvaluationError("dictionary lexicon export is invalid JSON") from failure
    if not isinstance(value, dict) or value.get("schemaVersion") != 1 or value.get("source") != "bundled-static-dictionary":
        raise SwipeEvaluationError("dictionary lexicon export has an incompatible identity")
    if value.get("maximumWords") != MAXIMUM_LEXICON_WORDS:
        raise SwipeEvaluationError("dictionary vocabulary bound differs from production")
    if value.get("apkSha256") != model_sources.file_sha256(apk):
        raise SwipeEvaluationError("dictionary export and APK hashes differ")
    asset = value.get("dictionaryAsset")
    if asset != "dicts/main_en-US.dict":
        raise SwipeEvaluationError("only the bundled en-US dictionary is supported by this diagnostic")
    try:
        with zipfile.ZipFile(apk) as archive:
            digest = hashlib.sha256(archive.read("assets/" + asset)).hexdigest()
    except (KeyError, zipfile.BadZipFile) as failure:
        raise SwipeEvaluationError("APK lacks the exported dictionary asset") from failure
    if digest != value.get("dictionarySha256"):
        raise SwipeEvaluationError("dictionary export and bundled asset hashes differ")
    words = value.get("words")
    if not isinstance(words, list) or not 1 <= len(words) <= MAXIMUM_LEXICON_WORDS:
        raise SwipeEvaluationError("dictionary export has an invalid vocabulary size")
    identities = set()
    surfaces = []
    for word in words:
        if not isinstance(word, dict) or set(word) != {"word", "languageTag", "frequency", "possiblyOffensive"}:
            raise SwipeEvaluationError("dictionary word has an invalid schema")
        surface, frequency = word["word"], word["frequency"]
        if (not isinstance(surface, str) or not 1 <= len(surface) <= 128
                or word["languageTag"] != "en-US" or type(frequency) is not int or not 0 <= frequency <= 255
                or type(word["possiblyOffensive"]) is not bool):
            raise SwipeEvaluationError("dictionary word has invalid values")
        normalized = _normalize(surface)
        if normalized in identities:
            raise SwipeEvaluationError("dictionary export contains duplicate normalized words")
        identities.add(normalized)
        # Default keyboard policy excludes possibly offensive static candidates.
        if not word["possiblyOffensive"]:
            surfaces.append((normalized, "en", frequency))
    return lexicon_entries(surfaces, layout), {
        "source": "bundled-static-dictionary",
        "sourceSplits": [],
        "apkSha256": value["apkSha256"],
        "dictionaryAsset": asset,
        "dictionarySha256": digest,
        "lexiconExportSha256": model_sources.file_sha256(path),
        "possiblyOffensiveWordsExcluded": True,
    }


def build_trie(entries: Iterable[LexiconEntry], approximate_length: int) -> TrieNode:
    root = TrieNode()
    minimum = max(1, approximate_length - LENGTH_TOLERANCE_BELOW)
    maximum = min(64, approximate_length + LENGTH_TOLERANCE_ABOVE)
    for entry in entries:
        if not minimum <= len(entry.emissions) <= maximum:
            continue
        node = root
        for output_class in entry.emissions:
            node = node.children.setdefault(output_class, TrieNode())
        node.words.append(entry)
    return root


def _log_add(first: float, second: float) -> float:
    if first == LOG_ZERO:
        return second
    if second == LOG_ZERO:
        return first
    maximum = max(first, second)
    return maximum + math.log(math.exp(first - maximum) + math.exp(second - maximum))


def _log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    denominator = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return [value - denominator for value in values]


def collapse_greedy(logits: Sequence[Sequence[float]]) -> tuple[int, ...]:
    result = []
    previous = -1
    for frame in logits:
        output_class = max(range(len(frame)), key=frame.__getitem__)
        if output_class != 0 and output_class != previous:
            result.append(output_class)
        previous = output_class
    return tuple(result)


def path_length_estimate(path: Sequence[float], layout: dict[str, Any]) -> int:
    """Mirror the Android live-key-unit estimate without consulting the target label."""
    enabled_centers = [
        (float(layout["keyCenters"][index * 2]), float(layout["keyCenters"][index * 2 + 1]))
        for index, enabled in enumerate(layout["keyMask"])
        if enabled
    ]
    if len(path) != 128 or not enabled_centers:
        return 0
    x_steps = []
    rows: dict[float, list[float]] = collections.defaultdict(list)
    for x, y in enabled_centers:
        rows[round(y, 6)].append(x)
    for xs in rows.values():
        ordered = sorted(set(xs))
        x_steps.extend(right - left for left, right in zip(ordered, ordered[1:]) if right > left)
    ys = sorted({round(y, 6) for _x, y in enabled_centers})
    y_steps = [right - left for left, right in zip(ys, ys[1:]) if right > left]
    if not x_steps or not y_steps:
        return 0
    key_width = sorted(x_steps)[len(x_steps) // 2]
    key_height = sorted(y_steps)[len(y_steps) // 2]
    points = list(zip(path[::2], path[1::2], strict=True))
    path_length_in_keys = sum(
        math.hypot((right[0] - left[0]) / key_width, (right[1] - left[1]) / key_height)
        for left, right in zip(points, points[1:])
    )
    return max(1, min(64, round(path_length_in_keys / 3.9) + 1))


def prefix_beam_decode_scored(
    logits: Sequence[Sequence[float]],
    trie: TrieNode,
    *,
    beam_width: int = 64,
    maximum_results: int = 32,
) -> list[ScoredLexiconEntry]:
    if not logits or beam_width <= 0 or maximum_results <= 0:
        return []
    beam: dict[tuple[int, ...], tuple[float, float, TrieNode]] = {(): (0.0, LOG_ZERO, trie)}
    for frame in logits:
        probabilities = _log_softmax(frame)
        next_beam: dict[tuple[int, ...], tuple[float, float, TrieNode]] = {}

        def merge(prefix: tuple[int, ...], blank: float, non_blank: float, node: TrieNode) -> None:
            old_blank, old_non_blank, _ = next_beam.get(prefix, (LOG_ZERO, LOG_ZERO, node))
            next_beam[prefix] = (_log_add(old_blank, blank), _log_add(old_non_blank, non_blank), node)

        for prefix, (blank, non_blank, node) in beam.items():
            total = _log_add(blank, non_blank)
            merge(prefix, total + probabilities[0], LOG_ZERO, node)
            repeated_class = prefix[-1] if prefix else None
            if repeated_class is not None:
                merge(prefix, LOG_ZERO, non_blank + probabilities[repeated_class], node)
            for output_class, child in node.children.items():
                extension_probability = blank if output_class == repeated_class else total
                if extension_probability != LOG_ZERO:
                    merge(prefix + (output_class,), LOG_ZERO, extension_probability + probabilities[output_class], child)
        ranked = sorted(
            next_beam.items(),
            key=lambda item: (-_log_add(item[1][0], item[1][1]), item[0]),
        )[:beam_width]
        beam = dict(ranked)

    candidates: list[tuple[float, LexiconEntry]] = []
    for _prefix, (blank, non_blank, node) in beam.items():
        score = _log_add(blank, non_blank) / len(logits)
        candidates.extend((score, entry) for entry in node.words)
    candidates.sort(key=lambda item: (-item[0], -item[1].frequency, item[1].word, item[1].language))
    result: list[ScoredLexiconEntry] = []
    seen = set()
    for score, entry in candidates:
        key = (_normalize(entry.word), entry.language)
        if key not in seen:
            seen.add(key)
            result.append(ScoredLexiconEntry(entry, score))
            if len(result) >= maximum_results:
                break
    return result


def prefix_beam_decode(
    logits: Sequence[Sequence[float]],
    trie: TrieNode,
    *,
    beam_width: int = 64,
    maximum_results: int = 32,
) -> list[LexiconEntry]:
    return [
        scored.entry
        for scored in prefix_beam_decode_scored(
            logits,
            trie,
            beam_width=beam_width,
            maximum_results=maximum_results,
        )
    ]


def _z_normalize(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return [1.0]
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation < 1e-9:
        return [0.0] * len(values)
    return [(value - mean) / deviation for value in values]


def rank_static_fusion(candidates: Sequence[ScoredLexiconEntry]) -> list[LexiconEntry]:
    """Mirror the production scorer's spatial/static weighting for the isolated CTC slate."""
    spatial = _z_normalize([candidate.spatial for candidate in candidates])
    frequencies = _z_normalize([math.log1p(candidate.entry.frequency) for candidate in candidates])
    ranked = sorted(
        zip(candidates, spatial, frequencies, strict=True),
        key=lambda item: (
            -(item[1] + item[2] * 0.65),
            _normalize(item[0].entry.word),
            item[0].entry.language,
        ),
    )
    # Production reserves one of its 32 bounded slots for the empty swipe raw form;
    # LegacySuggestionFusion removes that placeholder before publication.
    return [item[0].entry for item in ranked[:31]]


def _gesture_length(word: str) -> int:
    return sum(character not in {"'", "\N{RIGHT SINGLE QUOTATION MARK}", "-"} for character in _normalize(word))


def _resample_template(points: Any, numpy: Any, count: int = 64) -> Any:
    if len(points) == 1:
        return numpy.repeat(points, count, axis=0)
    deltas = points[1:] - points[:-1]
    segment_lengths = numpy.sqrt(numpy.sum(deltas * deltas, axis=1))
    keep = numpy.concatenate((numpy.asarray([True]), segment_lengths > 0))
    points = points[keep]
    if len(points) == 1:
        return numpy.repeat(points, count, axis=0)
    deltas = points[1:] - points[:-1]
    cumulative = numpy.concatenate((numpy.asarray([0.0]), numpy.cumsum(
        numpy.sqrt(numpy.sum(deltas * deltas, axis=1)),
    )))
    targets = numpy.linspace(0.0, float(cumulative[-1]), count)
    return numpy.stack((
        numpy.interp(targets, cumulative, points[:, 0]),
        numpy.interp(targets, cumulative, points[:, 1]),
    ), axis=1).astype(numpy.float32)


def _turn_count(points: Any, numpy: Any) -> int:
    if len(points) < 3:
        return 0
    first = points[1:-1] - points[:-2]
    second = points[2:] - points[1:-1]
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    directions = numpy.sign(cross[numpy.abs(cross) > 0.002])
    if len(directions) < 2:
        return 0
    return int(numpy.count_nonzero(directions[1:] != directions[:-1]))


def build_geometric_index(
    lexicon: dict[str, list[LexiconEntry]],
    layout: dict[str, Any],
    numpy: Any,
) -> GeometricLexiconIndex:
    centers_by_class = numpy.asarray(layout["keyCenters"], dtype=numpy.float32).reshape(64, 2)
    enabled_indices = numpy.flatnonzero(numpy.asarray(layout["keyMask"], dtype=numpy.int8))
    enabled_centers = centers_by_class[enabled_indices]
    enabled_classes = enabled_indices.astype(numpy.int16) + 1

    rows: dict[float, list[float]] = collections.defaultdict(list)
    for x, y in enabled_centers.tolist():
        rows[round(y, 6)].append(x)
    x_steps = [
        right - left
        for values in rows.values()
        for left, right in zip(sorted(set(values)), sorted(set(values))[1:])
        if right > left
    ]
    ys = sorted(rows)
    y_steps = [right - left for left, right in zip(ys, ys[1:]) if right > left]
    if not x_steps or not y_steps:
        raise SwipeEvaluationError("prepared swipe layout cannot derive live key dimensions")
    key_width = sorted(x_steps)[len(x_steps) // 2]
    key_height = sorted(y_steps)[len(y_steps) // 2]

    words: list[LexiconEntry] = []
    word_index: dict[tuple[str, str], int] = {}
    templates = []
    sequences = []
    word_ids = []
    canonical_lengths = []
    languages = []
    for language in sorted(lexicon):
        for entry in lexicon[language]:
            key = (_normalize(entry.word), entry.language)
            index = word_index.get(key)
            if index is None:
                index = len(words)
                word_index[key] = index
                words.append(entry)
            points = centers_by_class[numpy.asarray(entry.emissions, dtype=numpy.int16) - 1]
            templates.append(_resample_template(points, numpy))
            sequences.append(entry.emissions)
            word_ids.append(index)
            canonical_lengths.append(_gesture_length(entry.word))
            languages.append(entry.language)
    if not templates:
        raise SwipeEvaluationError("training vocabulary produced no geometric templates")

    maximum_length = max(map(len, sequences))
    padded = numpy.full((len(sequences), maximum_length), -1, dtype=numpy.int16)
    lengths = numpy.asarray([len(sequence) for sequence in sequences], dtype=numpy.int16)
    for index, sequence in enumerate(sequences):
        padded[index, :len(sequence)] = sequence
    template_array = numpy.stack(templates).astype(numpy.float32)
    turns = numpy.asarray([_turn_count(template, numpy) for template in template_array], dtype=numpy.int16)
    buckets: dict[tuple[str, int], Any] = {}
    grouped: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for index, (language, length) in enumerate(zip(languages, canonical_lengths, strict=True)):
        grouped[(language, length)].append(index)
    for key, indices in grouped.items():
        buckets[key] = numpy.asarray(indices, dtype=numpy.int32)
    return GeometricLexiconIndex(
        words=tuple(words),
        templates=template_array,
        sequences=padded,
        sequence_lengths=lengths,
        turn_counts=turns,
        word_ids=numpy.asarray(word_ids, dtype=numpy.int32),
        buckets=buckets,
        enabled_centers=enabled_centers,
        enabled_classes=enabled_classes,
        key_width=key_width,
        key_height=key_height,
    )


def _trace_classes(path: Any, index: GeometricLexiconIndex, numpy: Any) -> Any:
    difference = path[:, None, :] - index.enabled_centers[None, :, :]
    difference[:, :, 0] /= index.key_width
    difference[:, :, 1] /= index.key_height
    nearest = index.enabled_classes[numpy.argmin(numpy.sum(difference * difference, axis=2), axis=1)]
    if len(nearest) <= 1:
        return nearest
    return nearest[numpy.concatenate((numpy.asarray([True]), nearest[1:] != nearest[:-1]))]


def _normalized_edit_distances(trace: Any, sequences: Any, lengths: Any, numpy: Any) -> Any:
    rows, maximum_length = sequences.shape
    if rows == 0:
        return numpy.empty(0, dtype=numpy.float64)
    if len(trace) == 0:
        return numpy.ones(rows, dtype=numpy.float64)
    previous = numpy.broadcast_to(
        numpy.arange(maximum_length + 1, dtype=numpy.int16),
        (rows, maximum_length + 1),
    ).copy()
    current = numpy.empty_like(previous)
    for trace_index, output_class in enumerate(trace, 1):
        current[:, 0] = trace_index
        for sequence_index in range(1, maximum_length + 1):
            current[:, sequence_index] = numpy.minimum(
                numpy.minimum(
                    current[:, sequence_index - 1] + 1,
                    previous[:, sequence_index] + 1,
                ),
                previous[:, sequence_index - 1] +
                (sequences[:, sequence_index - 1] != output_class),
            )
        previous, current = current, previous
    distances = previous[numpy.arange(rows), lengths]
    return distances.astype(numpy.float64) / numpy.maximum(len(trace), lengths)


def geometric_decode(
    path_values: Sequence[float],
    language: str,
    index: GeometricLexiconIndex,
    numpy: Any,
    *,
    maximum_results: int = 32,
) -> list[ScoredLexiconEntry]:
    approximate_length = path_length_estimate(path_values, {
        "keyCenters": index.enabled_centers.reshape(-1).tolist(),
        "keyMask": [1] * len(index.enabled_centers),
    })
    minimum = max(1, approximate_length - LENGTH_TOLERANCE_BELOW)
    maximum = min(64, approximate_length + LENGTH_TOLERANCE_ABOVE)
    selected = [index.buckets[(language, length)] for length in range(minimum, maximum + 1)
                if (language, length) in index.buckets]
    if not selected:
        return []
    indices = numpy.concatenate(selected)
    path = numpy.asarray(path_values, dtype=numpy.float32).reshape(64, 2)
    templates = index.templates[indices]
    shape_cost = numpy.sqrt(numpy.sum((templates - path[None, :, :]) ** 2, axis=2)).mean(axis=1)
    start_end_cost = (
        numpy.sqrt(numpy.sum((templates[:, 0, :] - path[0]) ** 2, axis=1)) +
        numpy.sqrt(numpy.sum((templates[:, -1, :] - path[-1]) ** 2, axis=1))
    )
    trace = _trace_classes(path, index, numpy)
    trace_cost = _normalized_edit_distances(
        trace,
        index.sequences[indices],
        index.sequence_lengths[indices],
        numpy,
    )
    turn_cost = numpy.abs(index.turn_counts[indices] - _turn_count(path, numpy)) / numpy.maximum(
        index.sequence_lengths[indices],
        1,
    )
    costs = shape_cost * 2.2 + start_end_cost * 1.4 + trace_cost * 0.8 + turn_cost * 0.25
    word_costs = numpy.full(len(index.words), numpy.inf, dtype=numpy.float64)
    numpy.minimum.at(word_costs, index.word_ids[indices], costs)
    available = numpy.flatnonzero(numpy.isfinite(word_costs))
    ordered = sorted(
        available.tolist(),
        key=lambda word_id: (
            float(word_costs[word_id]),
            -index.words[word_id].frequency,
            index.words[word_id].word,
            index.words[word_id].language,
        ),
    )[:maximum_results]
    return [ScoredLexiconEntry(index.words[word_id], -float(word_costs[word_id])) for word_id in ordered]


def merge_swipe_slates(
    ctc_candidates: Sequence[ScoredLexiconEntry],
    geometric_candidates: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    merged: dict[tuple[str, str], ScoredLexiconEntry] = {}
    for slate in (ctc_candidates, geometric_candidates):
        normalized = _z_normalize([candidate.spatial for candidate in slate])
        for candidate, score in zip(slate, normalized, strict=True):
            key = (_normalize(candidate.word), candidate.entry.language)
            previous = merged.get(key)
            replacement = ScoredLexiconEntry(candidate.entry, score)
            if previous is None or score > previous.spatial:
                merged[key] = replacement
    return sorted(
        merged.values(),
        key=lambda candidate: (
            -candidate.spatial,
            -candidate.entry.frequency,
            candidate.entry.word,
            candidate.entry.language,
        ),
    )[:32]


def _dependencies():
    try:
        import numpy
        import onnxruntime
    except ImportError as failure:
        raise SwipeEvaluationError(
            "install the pinned model dependencies from models/training/requirements-linux-x86_64.lock "
            "and models/training/requirements-onnxruntime-build-linux-x86_64.lock"
        ) from failure
    versions = {"numpy": numpy.__version__, "onnxruntime": onnxruntime.__version__}
    if versions != {"numpy": "2.2.6", "onnxruntime": "1.26.0"}:
        raise SwipeEvaluationError(f"evaluation dependency versions do not match the locks: {versions}")
    return numpy, onnxruntime, versions


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise SwipeEvaluationError("cannot compute latency percentiles without samples")
    ordered = sorted(values)

    def nearest(percentile: float) -> float:
        return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

    return {
        "p50": nearest(0.50),
        "p95": nearest(0.95),
        "p99": nearest(0.99),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    destination = getattr(args, "slates_output", None)
    if destination is None:
        return _evaluate(args)
    destination = destination.resolve()
    if destination == args.output.resolve():
        raise SwipeEvaluationError("candidate slates and evaluation report need separate paths")
    if destination.exists():
        raise SwipeEvaluationError("candidate slate output already exists; preserve earlier evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".swipe-slates-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            report = _evaluate(args, stream)
            stream.flush()
            os.fsync(stream.fileno())
        staged = pathlib.Path(temporary)
        report["candidateSlates"] = {
            "schemaVersion": 1,
            "path": str(destination),
            "sha256": model_sources.file_sha256(staged),
            "bytes": staged.stat().st_size,
            "records": report["sample"]["rows"],
            "diagnosticOnly": True,
        }
        # Publish only a complete successful run, without replacing concurrent/prior evidence.
        os.link(staged, destination)
        return report
    finally:
        pathlib.Path(temporary).unlink(missing_ok=True)


def _write_candidate_slate(stream, row, ctc, geometric, merged):
    def candidates(slate):
        return [{"word": item.entry.word, "language": item.entry.language,
                 "frequency": item.entry.frequency, "spatial": item.spatial} for item in slate]
    record = {
        "schemaVersion": 1, "id": row.identifier, "sessionId": row.session_id,
        "language": row.language, "target": row.target, "strata": sorted(row.strata),
        "ctc": candidates(ctc), "geometric": candidates(geometric),
        "merged": candidates(merged),
    }
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False).encode() + b"\n")


def _evaluate(args: argparse.Namespace, slate_stream=None) -> dict[str, Any]:
    if not args.development and (args.sample_count < 5_000 or args.minimum_per_stratum < 500):
        raise SwipeEvaluationError("release diagnostics require 5,000 rows and 500 rows per stratum")
    spec = swipe_model_contract.load_spec(args.spec)
    data_root = args.data_root.resolve()
    prepared = swipe_model_contract.load_prepared_manifest(data_root, require_pinned=not args.development)
    export_report, model_path, export_report_hash = load_export(args.export_report, development=args.development)
    if export_report.get("modelSpecSha256") != spec.sha256:
        raise SwipeEvaluationError("exported model and current model spec do not match")
    split_manifest_path = data_root / "split-manifest.json"
    split_manifest_hash = model_sources.file_sha256(split_manifest_path)
    if export_report.get("dataManifestSha256") != split_manifest_hash:
        raise SwipeEvaluationError("export and evaluation data manifests do not match")
    layout = load_layout(data_root / "layout.json")
    rows = select_rows(
        load_test_rows(data_root / f"{args.split}.jsonl", layout, args.split),
        sample_count=args.sample_count,
        minimum_per_stratum=args.minimum_per_stratum,
    )
    if bool(args.dictionary_lexicon) != bool(args.dictionary_apk):
        raise SwipeEvaluationError("dictionary lexicon and originating APK must be supplied together")
    lexicon_source = {
        "source": "corpus-derived",
        "sourceSplits": ["train"] if args.split == "validation" else ["train", "validation"],
    }
    if args.dictionary_lexicon:
        lexicon, lexicon_source = load_dictionary_lexicon(args.dictionary_lexicon, args.dictionary_apk, layout)
    else:
        lexicon = build_lexicon(data_root, layout, evaluation_split=args.split)
    lexicon_words = {
        language: {entry.word for entry in entries}
        for language, entries in lexicon.items()
    }
    numpy, onnxruntime, versions = _dependencies()
    geometric_index = build_geometric_index(lexicon, layout, numpy)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    expected_inputs = {
        "path_coordinates": ("tensor(float)", [1, 64, 2]),
        "key_centers": ("tensor(float)", [1, 64, 2]),
        "key_mask": ("tensor(float)", [1, 64]),
    }
    actual_inputs = {value.name: (value.type, value.shape) for value in session.get_inputs()}
    actual_outputs = {value.name: (value.type, value.shape) for value in session.get_outputs()}
    if actual_inputs != expected_inputs or actual_outputs != {"logits": ("tensor(float)", [1, 32, 65])}:
        raise SwipeEvaluationError("ONNX Runtime exposes a tensor ABI that differs from the model spec")

    key_centers = numpy.asarray(layout["keyCenters"], dtype=numpy.float32).reshape(1, 64, 2)
    key_mask = numpy.asarray(layout["keyMask"], dtype=numpy.float32).reshape(1, 64)
    trie_cache: dict[tuple[str, int], TrieNode] = {}
    metric_names = ("overall",) + REQUIRED_STRATA
    metrics = {
        name: {"rows": 0, "top1": 0, "top3": 0, "inVocabulary": 0, "lengthWindowEligible": 0}
        for name in metric_names
    }
    static_fusion_metrics = {
        name: {"rows": 0, "top1": 0, "top3": 0, "inVocabulary": 0, "lengthWindowEligible": 0}
        for name in metric_names
    }
    geometric_metrics = {
        name: {"rows": 0, "top1": 0, "top3": 0, "inVocabulary": 0, "lengthWindowEligible": 0}
        for name in metric_names
    }
    ctc_geometric_metrics = {
        name: {"rows": 0, "top1": 0, "top3": 0, "inVocabulary": 0, "lengthWindowEligible": 0}
        for name in metric_names
    }
    inference_ms = []
    decode_ms = []
    total_ms = []
    geometric_ms = []
    fused_total_ms = []
    greedy_exact = 0
    greedy_blank_fallbacks = 0

    for index, row in enumerate(rows, 1):
        started = time.perf_counter_ns()
        path = numpy.asarray(row.path, dtype=numpy.float32).reshape(1, 64, 2)
        inference_started = time.perf_counter_ns()
        output = session.run(["logits"], {
            "path_coordinates": path,
            "key_centers": key_centers,
            "key_mask": key_mask,
        })[0]
        inference_finished = time.perf_counter_ns()
        if output.shape != (1, 32, 65) or not numpy.isfinite(output).all():
            raise SwipeEvaluationError(f"model returned invalid logits for held-out row {row.identifier}")
        logits = output[0].tolist()
        greedy = collapse_greedy(logits)
        greedy_exact += greedy == row.labels
        approximate_length = len(greedy)
        if approximate_length <= 0:
            greedy_blank_fallbacks += 1
            approximate_length = path_length_estimate(row.path, layout)
        cache_key = (row.language, approximate_length)
        trie = trie_cache.get(cache_key)
        if trie is None:
            trie = build_trie(lexicon.get(row.language, ()), approximate_length)
            trie_cache[cache_key] = trie
        decoded = prefix_beam_decode_scored(logits, trie, beam_width=args.beam_width)
        static_fusion = rank_static_fusion(decoded)
        ctc_finished = time.perf_counter_ns()
        geometric_started = time.perf_counter_ns()
        geometric = geometric_decode(row.path, row.language, geometric_index, numpy)
        geometric_fusion = rank_static_fusion(geometric)
        merged = merge_swipe_slates(decoded, geometric)
        ctc_geometric_fusion = rank_static_fusion(merged)
        finished = time.perf_counter_ns()
        if slate_stream is not None:
            _write_candidate_slate(slate_stream, row, decoded, geometric, merged)
        predictions = [entry.word for entry in decoded]
        static_predictions = [entry.word for entry in static_fusion]
        geometric_predictions = [entry.word for entry in geometric_fusion]
        ctc_geometric_predictions = [entry.word for entry in ctc_geometric_fusion]
        in_vocabulary = row.target in lexicon_words.get(row.language, set())
        length_window_eligible = (
            approximate_length - LENGTH_TOLERANCE_BELOW
            <= len(row.labels)
            <= approximate_length + LENGTH_TOLERANCE_ABOVE
        )
        names = ["overall", *sorted(row.strata)]
        for name in names:
            metric = metrics[name]
            static_metric = static_fusion_metrics[name]
            geometric_metric = geometric_metrics[name]
            ctc_geometric_metric = ctc_geometric_metrics[name]
            metric["rows"] += 1
            metric["top1"] += bool(predictions and predictions[0] == row.target)
            metric["top3"] += row.target in predictions[:3]
            metric["inVocabulary"] += in_vocabulary
            metric["lengthWindowEligible"] += length_window_eligible
            static_metric["rows"] += 1
            static_metric["top1"] += bool(static_predictions and static_predictions[0] == row.target)
            static_metric["top3"] += row.target in static_predictions[:3]
            static_metric["inVocabulary"] += in_vocabulary
            static_metric["lengthWindowEligible"] += length_window_eligible
            geometric_metric["rows"] += 1
            geometric_metric["top1"] += bool(geometric_predictions and geometric_predictions[0] == row.target)
            geometric_metric["top3"] += row.target in geometric_predictions[:3]
            geometric_metric["inVocabulary"] += in_vocabulary
            geometric_metric["lengthWindowEligible"] += length_window_eligible
            ctc_geometric_metric["rows"] += 1
            ctc_geometric_metric["top1"] += bool(
                ctc_geometric_predictions and ctc_geometric_predictions[0] == row.target
            )
            ctc_geometric_metric["top3"] += row.target in ctc_geometric_predictions[:3]
            ctc_geometric_metric["inVocabulary"] += in_vocabulary
            ctc_geometric_metric["lengthWindowEligible"] += length_window_eligible
        inference_ms.append((inference_finished - inference_started) / 1_000_000)
        decode_ms.append((ctc_finished - inference_finished) / 1_000_000)
        total_ms.append((ctc_finished - started) / 1_000_000)
        geometric_ms.append((finished - geometric_started) / 1_000_000)
        fused_total_ms.append((finished - started) / 1_000_000)
        if index % 250 == 0:
            print(f"evaluated {index}/{len(rows)} held-out swipes", file=sys.stderr, flush=True)

    def report_metrics(values: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float]]:
        return {
            name: {
                **counts,
                "top1Accuracy": counts["top1"] / counts["rows"],
                "top3Accuracy": counts["top3"] / counts["rows"],
                "vocabularyCoverage": counts["inVocabulary"] / counts["rows"],
                "lengthWindowCoverage": counts["lengthWindowEligible"] / counts["rows"],
            }
            for name, counts in values.items()
        }

    metric_report = report_metrics(metrics)
    static_metric_report = report_metrics(static_fusion_metrics)
    geometric_metric_report = report_metrics(geometric_metrics)
    ctc_geometric_metric_report = report_metrics(ctc_geometric_metrics)
    isolated_thresholds = (
        metric_report["overall"]["top1Accuracy"] >= 0.90
        and metric_report["overall"]["top3Accuracy"] >= 0.95
        and metric_report["short"]["top3Accuracy"] >= 0.90
        and metric_report["return_trip"]["top3Accuracy"] >= 0.90
    )
    static_fusion_thresholds = (
        static_metric_report["overall"]["top1Accuracy"] >= 0.90
        and static_metric_report["overall"]["top3Accuracy"] >= 0.95
        and static_metric_report["short"]["top3Accuracy"] >= 0.90
        and static_metric_report["return_trip"]["top3Accuracy"] >= 0.90
    )
    ctc_geometric_thresholds = (
        ctc_geometric_metric_report["overall"]["top1Accuracy"] >= 0.90
        and ctc_geometric_metric_report["overall"]["top3Accuracy"] >= 0.95
        and ctc_geometric_metric_report["short"]["top3Accuracy"] >= 0.90
        and ctc_geometric_metric_report["return_trip"]["top3Accuracy"] >= 0.90
    )
    geometric_error = 1.0 - geometric_metric_report["overall"]["top1Accuracy"]
    ctc_geometric_error = 1.0 - ctc_geometric_metric_report["overall"]["top1Accuracy"]
    relative_error_reduction = (
        (geometric_error - ctc_geometric_error) / geometric_error if geometric_error > 0 else 0.0
    )
    return {
        "schemaVersion": 1,
        "modelId": "swipe-latin-v1",
        "diagnosticOnly": True,
        "phase0ReleaseEvidence": False,
        "releaseEligibleInputs": bool(
            export_report.get("releaseEligible") is True and prepared.get("schemaVersion") == 1 and not args.development
        ),
        "sample": {
            "split": args.split,
            "rows": len(rows),
            "sessions": len({row.session_id for row in rows}),
            "minimumRowsPerStratum": args.minimum_per_stratum,
            "selection": "sha256-row-id-stratified-v1",
        },
        "lexicon": {
            **lexicon_source,
            "evaluatedTargetsExcludedFromConstruction": True,
            "testTargetsExcludedFromConstruction": True,
            "maximumWords": MAXIMUM_LEXICON_WORDS,
            "words": len({(entry.word, entry.language) for entries in lexicon.values() for entry in entries}),
            "gestureVariants": sum(len(entries) for entries in lexicon.values()),
            "languages": sorted(lexicon),
            "lengthToleranceBelow": LENGTH_TOLERANCE_BELOW,
            "lengthToleranceAbove": LENGTH_TOLERANCE_ABOVE,
        },
        "decoder": {
            "algorithm": "lexicon-constrained-ctc-prefix-beam-v1",
            "beamWidth": args.beam_width,
            "greedyExactAccuracy": greedy_exact / len(rows),
            "greedyBlankFallbacks": greedy_blank_fallbacks,
            "geometricAlgorithm": "production-template-cost-v1",
            "geometricTemplates": len(geometric_index.templates),
            "decoderSlatesNormalizedBeforeUnion": True,
        },
        "metrics": metric_report,
        "staticFusionMetrics": static_metric_report,
        "geometricMetrics": geometric_metric_report,
        "ctcGeometricFusionMetrics": ctc_geometric_metric_report,
        "latencyMs": {
            "hostDiagnosticOnly": True,
            "inference": _percentiles(inference_ms),
            "prefixBeam": _percentiles(decode_ms),
            "total": _percentiles(total_ms),
            "vectorizedGeometricAndFusion": _percentiles(geometric_ms),
            "ctcGeometricFusionTotal": _percentiles(fused_total_ms),
        },
        "qualitySnapshot": {
            "meetsPhase0SwipeThresholdsInCtcIsolation": isolated_thresholds,
            "meetsPhase0SwipeThresholdsWithStaticFusion": static_fusion_thresholds,
            "meetsPhase0SwipeThresholdsWithCtcGeometricFusion": ctc_geometric_thresholds,
            "ctcGeometricTop1RelativeErrorReductionOverGeometric": relative_error_reduction,
            "meetsTwentyPercentRelativeErrorReductionOverGeometric": relative_error_reduction >= 0.20,
            "doesNotSatisfyPhase0": True,
        },
        "artifacts": {
            "model": {"path": model_path.name, "sha256": model_sources.file_sha256(model_path)},
            "exportReportSha256": export_report_hash,
            "splitManifestSha256": split_manifest_hash,
            "evaluatorSha256": model_sources.file_sha256(pathlib.Path(__file__)),
            **({"testDataSha256": prepared["outputs"]["test.jsonl"]["sha256"]} if args.split == "test" else {}),
            "evaluationDataSha256": prepared["outputs"][f"{args.split}.jsonl"]["sha256"],
            "modelSpecSha256": spec.sha256,
        },
        "toolchain": versions,
        "limitations": [
            "Host CPU timing is not Android device latency or memory evidence.",
            ("The diagnostic lexicon is corpus-derived rather than the production AOSP dictionary."
             if not args.dictionary_lexicon else
             "The native dictionary dump is hash-bound to the supplied APK; complete runtime fusion remains unmeasured."),
            "Personal, context, language-lock, retained AOSP gesture suggestions, and complete final fusion are not measured.",
        ],
    }


def _write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=swipe_model_contract.DEFAULT_SPEC)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--export-report", type=pathlib.Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--slates-output", type=pathlib.Path,
                        help="Save complete hash-bound candidate scores for later diagnostics; refuses existing paths")
    parser.add_argument("--dictionary-lexicon", type=pathlib.Path)
    parser.add_argument("--dictionary-apk", type=pathlib.Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--sample-count", type=int, default=5_000)
    parser.add_argument("--minimum-per-stratum", type=int, default=500)
    parser.add_argument("--beam-width", type=int, choices=range(1, 257), default=64)
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        report = evaluate(args)
        _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (SwipeEvaluationError, swipe_model_contract.SwipeModelContractError) as failure:
        print(f"swipe CTC evaluation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
