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
    frequency_free: bool = False
    source: str = ""
    oov_map_blend: float = 0.0
    nbest_rank: int | None = None

    @property
    def word(self) -> str:
        return self.entry.word


@dataclasses.dataclass(frozen=True)
class OovScoreCalibration:
    """Maps unconstrained CTC forward log-prob onto lexicon-constrained spatial scores.

    Fitted by OLS on in-lexicon (forward, spatial) pairs. ``optimism_offset`` is the
    mean unconstrained-minus-lexicon gap on out-of-lexicon greedy paths; it is not a
    frequency prior. Evaluation targets never enter the fit.
    """

    slope: float
    intercept: float
    optimism_offset: float
    pair_count: int
    optimism_path_count: int

    def to_lexicon_spatial(self, ctc_forward: float) -> float:
        if not math.isfinite(ctc_forward):
            raise SwipeEvaluationError("CTC forward probability is not finite")
        return self.slope * ctc_forward + self.intercept - self.optimism_offset


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


# Stratum-adaptive decoder union: geometry is near-dead-weight on the failing
# strata, where truncation drops CTC-only targets. Short stays at a full 32/32.
# Geometry 4/4/8/8/8 stole 6k top-3; keep 8/8/12/16/16.
STRATUM_GEOMETRY_LIMITS = {
    "very_sloppy": 8,
    "long": 8,
    "sloppy": 12,
    "return_trip": 16,
    "double_letter": 16,
}


def decoder_slate_budgets(strata: Iterable[str]) -> tuple[int, int]:
    """Return (ctc_limit, geometric_limit) for the 32-slot union.

    Short is already at 99% merged recall and is never reduced, including when a
    short row also belongs to a noisier stratum.
    """
    names = set(strata)
    if "short" in names:
        return (32, 32)
    geometric_limit = 32
    for name, limit in STRATUM_GEOMETRY_LIMITS.items():
        if name in names:
            geometric_limit = min(geometric_limit, limit)
    return (32, geometric_limit)


def ctc_forward_logprob(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Length-normalized CTC forward log-probability of an emission sequence."""
    if not logits or not labels:
        return LOG_ZERO
    sequence = [0]
    for label in labels:
        if not isinstance(label, int) or label <= 0:
            return LOG_ZERO
        sequence.extend((label, 0))
    probabilities = _log_softmax(logits[0])
    state = [LOG_ZERO] * len(sequence)
    state[0] = probabilities[0]
    if labels[0] < len(probabilities):
        state[1] = probabilities[labels[0]]
    for frame in logits[1:]:
        probabilities = _log_softmax(frame)
        following = []
        for index, label in enumerate(sequence):
            total = state[index]
            if index:
                total = _log_add(total, state[index - 1])
            if index > 1 and label and label != sequence[index - 2]:
                total = _log_add(total, state[index - 2])
            following.append(total + probabilities[label] if label < len(probabilities) else LOG_ZERO)
        state = following
    return _log_add(state[-1], state[-2]) / len(logits)


def fit_oov_score_calibration(
    forwards: Sequence[float],
    lexicon_spatials: Sequence[float],
    optimism_gaps: Sequence[float] = (),
) -> OovScoreCalibration:
    """Fit CTC-forward → lexicon spatial mapping. Pairs must not include evaluation targets."""
    if len(forwards) != len(lexicon_spatials) or len(forwards) < 2:
        raise SwipeEvaluationError("OOV calibration needs at least two paired scores")
    if any(
        not math.isfinite(forward) or not math.isfinite(spatial)
        for forward, spatial in zip(forwards, lexicon_spatials, strict=True)
    ):
        raise SwipeEvaluationError("OOV calibration pairs must be finite")
    count = len(forwards)
    mean_forward = sum(forwards) / count
    mean_spatial = sum(lexicon_spatials) / count
    variance = sum((forward - mean_forward) ** 2 for forward in forwards)
    if variance < 1e-18:
        slope = 1.0
        intercept = mean_spatial - mean_forward
    else:
        covariance = sum(
            (forward - mean_forward) * (spatial - mean_spatial)
            for forward, spatial in zip(forwards, lexicon_spatials, strict=True)
        )
        slope = covariance / variance
        intercept = mean_spatial - slope * mean_forward
    gaps = [gap for gap in optimism_gaps if math.isfinite(gap)]
    offset = sum(gaps) / len(gaps) if gaps else 0.0
    return OovScoreCalibration(slope, intercept, offset, count, len(gaps))


def oov_score_calibration_report(calibration: OovScoreCalibration) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "diagnosticOnly": True,
        "releaseEligible": False,
        "slope": calibration.slope,
        "intercept": calibration.intercept,
        "optimismOffset": calibration.optimism_offset,
        "pairCount": calibration.pair_count,
        "optimismPathCount": calibration.optimism_path_count,
        "frequencyPrior": None,
        "evaluatedTargetsExcludedFromFit": True,
        "limitations": [
            "OLS maps unconstrained CTC forward log-prob onto lexicon-constrained spatial scores.",
            "Optimism offset is a path-level CTC gap, not a dictionary frequency prior.",
            "Evaluation targets are not used to construct or score candidates.",
        ],
    }


def load_oov_score_calibration(path: pathlib.Path) -> OovScoreCalibration:
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as failure:
        raise SwipeEvaluationError("OOV calibration file is invalid JSON") from failure
    try:
        calibration = OovScoreCalibration(
            float(value["slope"]),
            float(value["intercept"]),
            float(value["optimismOffset"]),
            int(value["pairCount"]),
            int(value["optimismPathCount"]),
        )
    except (KeyError, TypeError, ValueError) as failure:
        raise SwipeEvaluationError("OOV calibration file is missing fitted fields") from failure
    if value.get("frequencyPrior") not in (None,):
        raise SwipeEvaluationError("OOV calibration must not carry a fabricated frequency prior")
    if not all(math.isfinite(item) for item in (
        calibration.slope, calibration.intercept, calibration.optimism_offset,
    )):
        raise SwipeEvaluationError("OOV calibration parameters must be finite")
    return calibration


def load_known_offensive_words(path: pathlib.Path) -> set[str]:
    """Load possibly-offensive surfaces for the reserved-slot filter. Not a vocabulary expansion."""
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as failure:
        raise SwipeEvaluationError("known-offensive source is invalid JSON") from failure
    words = value.get("words")
    if not isinstance(words, list) or not words:
        raise SwipeEvaluationError("known-offensive source has no words")
    flagged = set()
    for word in words:
        if not isinstance(word, dict) or word.get("possiblyOffensive") is not True:
            continue
        surface = word.get("word")
        if isinstance(surface, str) and surface:
            flagged.add(_normalize(surface))
    return flagged


def load_train_paths(
    path: pathlib.Path,
    layout: dict[str, Any],
    *,
    maximum_rows: int,
) -> list[tuple[str, str, tuple[float, ...]]]:
    """Deterministic training swipe paths for calibration. Targets are not returned."""
    if maximum_rows <= 0:
        raise SwipeEvaluationError("calibration path count must be positive")
    rows: list[tuple[str, str, tuple[float, ...]]] = []
    identifiers = set()
    with path.open("rb") as stream:
        for line in stream:
            if len(rows) >= maximum_rows:
                break
            if len(line) > MAXIMUM_JSONL_LINE_BYTES:
                raise SwipeEvaluationError("training swipe row is too large")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise SwipeEvaluationError(f"training swipe data contains invalid JSON: {failure}") from failure
            if not isinstance(value, dict) or set(value) != TEST_ROW_KEYS:
                raise SwipeEvaluationError("training swipe row has an unexpected schema")
            if value.get("schemaVersion") != 1 or value.get("split") != "train" or value.get("layoutId") != layout["id"]:
                raise SwipeEvaluationError("training swipe row has an incompatible identity")
            identifier = value.get("id")
            language = value.get("language")
            path_values = value.get("pathCoordinates")
            if not isinstance(identifier, str) or not model_sources.SHA256.fullmatch(identifier):
                raise SwipeEvaluationError("training swipe row has an invalid id")
            if identifier in identifiers:
                raise SwipeEvaluationError("training swipe row ids are not unique")
            if not isinstance(language, str) or not language:
                raise SwipeEvaluationError("training swipe row has an invalid language")
            if (
                not isinstance(path_values, list)
                or len(path_values) != 128
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                    for item in path_values
                )
            ):
                raise SwipeEvaluationError("training swipe row has an invalid path")
            identifiers.add(identifier)
            rows.append((identifier, language, tuple(float(item) for item in path_values)))
    if len(rows) < maximum_rows:
        raise SwipeEvaluationError("training split lacks enough rows for calibration")
    return rows


def _z_score_against(value: float, values: Sequence[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return 1.0 if abs(value - values[0]) < 1e-12 else value - values[0]
    mean = sum(values) / len(values)
    variance = sum((item - mean) ** 2 for item in values) / len(values)
    deviation = math.sqrt(variance)
    if deviation < 1e-9:
        return 0.0
    return (value - mean) / deviation


def project_onto_lexicon_scale(value: float, spatials: Sequence[float]) -> float:
    """Clip a mapped score onto the observed lexicon-constrained spatial range."""
    if not spatials:
        return value
    return min(max(value, min(spatials)), max(spatials))


def conservative_lexicon_spatial(spatials: Sequence[float]) -> float:
    """Low-quartile lexicon spatial. Frequency-free OOV compete on-scale without taking top-3."""
    if not spatials:
        return 0.0
    ordered = sorted(spatials)
    return ordered[len(ordered) // 4]


def median_lexicon_spatial(spatials: Sequence[float]) -> float:
    """Median lexicon spatial for in-lexicon reserved without a decoder score."""
    if not spatials:
        return 0.0
    ordered = sorted(spatials)
    return ordered[len(ordered) // 2]


def _pool_mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    if len(values) == 1:
        return values[0], 1.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = math.sqrt(variance)
    return mean, 1.0 if deviation < 1e-9 else deviation


def map_z_onto_pool(z_value: float, dest: Sequence[float]) -> float:
    """Place a z-score onto the destination spatial pool. No targets."""
    mean, deviation = _pool_mean_std(dest)
    return mean + z_value * deviation


def unconstrained_prefix_beam_scored(
    logits: Sequence[Sequence[float]],
    *,
    beam_width: int = 16,
    maximum_results: int = 8,
) -> list[tuple[tuple[int, ...], float]]:
    """Lexicon-free CTC prefix beam. Does not consult evaluation targets or vocabulary."""
    if not logits or beam_width <= 0 or maximum_results <= 0:
        return []
    class_count = len(logits[0])
    beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, LOG_ZERO)}
    for frame in logits:
        probabilities = _log_softmax(frame)
        next_beam: dict[tuple[int, ...], tuple[float, float]] = {}

        def merge(prefix: tuple[int, ...], blank: float, non_blank: float) -> None:
            old_blank, old_non_blank = next_beam.get(prefix, (LOG_ZERO, LOG_ZERO))
            next_beam[prefix] = (_log_add(old_blank, blank), _log_add(old_non_blank, non_blank))

        for prefix, (blank, non_blank) in beam.items():
            total = _log_add(blank, non_blank)
            merge(prefix, total + probabilities[0], LOG_ZERO)
            repeated_class = prefix[-1] if prefix else None
            if repeated_class is not None and 0 <= repeated_class < len(probabilities):
                merge(prefix, LOG_ZERO, non_blank + probabilities[repeated_class])
            for output_class in range(1, class_count):
                extension_probability = blank if output_class == repeated_class else total
                if extension_probability != LOG_ZERO:
                    merge(
                        prefix + (output_class,),
                        LOG_ZERO,
                        extension_probability + probabilities[output_class],
                    )
        ranked = sorted(
            next_beam.items(),
            key=lambda item: (-_log_add(item[1][0], item[1][1]), item[0]),
        )[:beam_width]
        beam = dict(ranked)
    scored = [
        (prefix, _log_add(blank, non_blank) / len(logits))
        for prefix, (blank, non_blank) in beam.items()
        if prefix
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:maximum_results]


def emissions_to_spelling(emissions: Sequence[int], layout: dict[str, Any]) -> str | None:
    labels = layout["keyLabels"]
    characters = []
    for output_class in emissions:
        if not isinstance(output_class, int) or not 1 <= output_class <= len(labels):
            return None
        character = labels[output_class - 1]
        if not isinstance(character, str) or len(character) != 1 or not character.isascii() or not character.isalpha():
            return None
        characters.append(character)
    return "".join(characters).lower() if characters else None


def _collapse_classes(classes: Sequence[int]) -> tuple[int, ...]:
    collapsed = []
    previous = -1
    for output_class in classes:
        if output_class != 0 and output_class != previous:
            collapsed.append(output_class)
        previous = output_class
    return tuple(collapsed)


def greedy_unconstrained_emissions(
    logits: Sequence[Sequence[float]],
) -> list[tuple[tuple[int, ...], float]]:
    """Greedy CTC collapse. No target argument."""
    greedy = collapse_greedy(logits)
    if not greedy:
        return []
    return [(greedy, ctc_forward_logprob(logits, greedy))]


def greedy_alt_unconstrained_emissions(
    logits: Sequence[Sequence[float]],
    seen: set[tuple[int, ...]] | None = None,
) -> list[tuple[tuple[int, ...], float]]:
    """Per-frame 2nd/3rd-best class swap on the greedy path. No beam, no targets."""
    if not logits:
        return []
    known = set(seen or ())
    greedy_classes = [max(range(len(frame)), key=frame.__getitem__) for frame in logits]
    results: list[tuple[tuple[int, ...], float]] = []
    for index, frame in enumerate(logits):
        ordered = sorted(range(len(frame)), key=frame.__getitem__, reverse=True)
        for alt_class in ordered[1:3]:
            alternative = list(greedy_classes)
            alternative[index] = alt_class
            emissions = _collapse_classes(alternative)
            if emissions and emissions not in known:
                known.add(emissions)
                results.append((emissions, ctc_forward_logprob(logits, emissions)))
    return results


def nbest_unconstrained_emissions(
    logits: Sequence[Sequence[float]],
    *,
    n_best: int,
    beam_width: int,
    seen: set[tuple[int, ...]] | None = None,
) -> list[tuple[tuple[int, ...], float, int]]:
    """Lexicon-free prefix-beam n-best. Rank is 1-indexed in the beam list. No targets."""
    if n_best <= 0 or not logits:
        return []
    known = set(seen or ())
    results: list[tuple[tuple[int, ...], float, int]] = []
    for rank, (emissions, score) in enumerate(
        unconstrained_prefix_beam_scored(logits, beam_width=beam_width, maximum_results=n_best),
        start=1,
    ):
        if emissions in known:
            continue
        known.add(emissions)
        results.append((emissions, score, rank))
    return results


def unconstrained_ctc_emissions(
    logits: Sequence[Sequence[float]],
    *,
    n_best: int,
    beam_width: int = 16,
) -> list[tuple[tuple[int, ...], float]]:
    """Greedy top-1 plus lexicon-free n-best CTC emissions. No target argument."""
    if n_best <= 0:
        return []
    results: list[tuple[tuple[int, ...], float]] = []
    seen: set[tuple[int, ...]] = set()
    for emissions, score in greedy_unconstrained_emissions(logits):
        results.append((emissions, score))
        seen.add(emissions)
    if n_best <= 1:
        return results[:1]
    for emissions, score, _rank in nbest_unconstrained_emissions(
        logits, n_best=n_best, beam_width=beam_width, seen=seen,
    ):
        results.append((emissions, score))
        seen.add(emissions)
        if len(results) >= n_best:
            break
    for emissions, score in greedy_alt_unconstrained_emissions(logits, seen=seen):
        results.append((emissions, score))
        seen.add(emissions)
    return results


def lexicon_neighbors(spelling: str, lexicon_by_word: dict[str, LexiconEntry]) -> list[LexiconEntry]:
    """In-lexicon edit-distance-1 and apostrophe variants of a CTC spelling. No targets."""
    if not spelling or not lexicon_by_word:
        return []
    letters = "abcdefghijklmnopqrstuvwxyz"
    found: dict[str, LexiconEntry] = {}

    def consider(candidate: str) -> None:
        entry = lexicon_by_word.get(candidate)
        if entry is not None:
            found[entry.word] = entry

    consider(spelling)
    for index, character in enumerate(spelling):
        consider(spelling[:index] + spelling[index + 1:])
        if index + 1 < len(spelling):
            consider(spelling[:index] + spelling[index + 1] + character + spelling[index + 2:])
        for letter in letters:
            if letter != character:
                consider(spelling[:index] + letter + spelling[index + 1:])
            consider(spelling[:index] + letter + spelling[index:])
        consider(spelling[:index] + "'" + spelling[index:])
    for letter in letters:
        consider(spelling + letter)
    consider(spelling + "'")
    consider(spelling.replace("'", ""))
    return list(found.values())


RESERVED_SOURCE_ORDER = (
    "greedy",
    "greedy_alts",
    "nbest",
    "neighbors",
    "truncated_ctc",
    "truncated_geometry",
)
PUBLISHED_SLATE_BOUND = 32
PUBLISHED_RANKING_BOUND = 31
EXTRA_OOV_FILL = 7
TINY_NBEST_HOLD = 4
ABLATION_EXTRA_SOURCES = (
    "greedy_alts",
    "neighbors",
    "nbest",
    "truncated_ctc",
    "truncated_geometry",
)


@dataclasses.dataclass(frozen=True)
class ReservedCandidate:
    """A reserved-slot spelling with its construction source. No evaluation target."""

    entry: LexiconEntry
    source: str
    forward: float | None = None
    nbest_rank: int | None = None
    decoder_spatial: float | None = None


def _reserved_from_emissions(
    emissions: tuple[int, ...],
    forward: float,
    source: str,
    layout: dict[str, Any],
    *,
    known_offensive: set[str],
    lexicon_by_word: dict[str, LexiconEntry],
    language: str,
    skip_keys: set[tuple[str, str]],
    nbest_rank: int | None = None,
) -> tuple[ReservedCandidate | None, int]:
    spelling = emissions_to_spelling(emissions, layout)
    if spelling is None:
        return None, 0
    normalized = _normalize(spelling)
    if normalized in known_offensive:
        return None, 1
    key = (normalized, language)
    if key in skip_keys:
        return None, 0
    lexicon_entry = lexicon_by_word.get(normalized)
    if lexicon_entry is None:
        entry = LexiconEntry(normalized, language, emissions, 0)
    else:
        entry = LexiconEntry(
            lexicon_entry.word, lexicon_entry.language, emissions, lexicon_entry.frequency,
        )
        key = (_normalize(entry.word), entry.language)
        if key in skip_keys:
            return None, 0
    return ReservedCandidate(entry, source, forward=forward, nbest_rank=nbest_rank), 0


def collect_reserved_sources(
    logits: Sequence[Sequence[float]],
    layout: dict[str, Any],
    *,
    n_best: int,
    beam_width: int,
    existing: Iterable[ScoredLexiconEntry],
    known_offensive: set[str],
    lexicon_by_word: dict[str, LexiconEntry],
    language: str,
    ctc: Sequence[ScoredLexiconEntry] = (),
    geometric: Sequence[ScoredLexiconEntry] = (),
    baseline_merged: Sequence[ScoredLexiconEntry] = (),
    include_lexicon_neighbors: bool = False,
    include_truncated: bool = False,
) -> tuple[dict[str, list[ReservedCandidate]], int, list[str]]:
    """Build per-source reserved spellings. Does not take evaluation targets."""
    sources = {name: [] for name in RESERVED_SOURCE_ORDER}
    rejected = 0
    spellings: list[str] = []
    per_source_keys = {name: set() for name in RESERVED_SOURCE_ORDER}

    def note_spelling(emissions: tuple[int, ...]) -> None:
        spelling = emissions_to_spelling(emissions, layout)
        if spelling is None:
            return
        normalized = _normalize(spelling)
        if normalized in known_offensive:
            return
        spellings.append(normalized)

    decoder_spatial_by_key: dict[tuple[str, str], float] = {}
    for slate in (ctc, geometric):
        if not slate:
            continue
        for item, score in zip(slate, _z_normalize([entry.spatial for entry in slate]), strict=True):
            key = (_normalize(item.word), item.entry.language)
            previous = decoder_spatial_by_key.get(key)
            if previous is None or score > previous:
                decoder_spatial_by_key[key] = score

    def accept(candidate: ReservedCandidate | None, offensive: int) -> None:
        nonlocal rejected
        rejected += offensive
        if candidate is None:
            return
        key = (_normalize(candidate.entry.word), candidate.entry.language)
        bucket = per_source_keys[candidate.source]
        if key in bucket:
            return
        if candidate.decoder_spatial is None and candidate.entry.frequency > 0:
            spatial = decoder_spatial_by_key.get(key)
            if spatial is not None:
                candidate = dataclasses.replace(candidate, decoder_spatial=spatial)
        bucket.add(key)
        sources[candidate.source].append(candidate)

    greedy_emissions: set[tuple[int, ...]] = set()
    greedy_spellings: list[str] = []
    empty_skip: set[tuple[str, str]] = set()
    for emissions, forward in greedy_unconstrained_emissions(logits):
        greedy_emissions.add(emissions)
        spelling = emissions_to_spelling(emissions, layout)
        if spelling is not None:
            normalized = _normalize(spelling)
            if normalized not in known_offensive:
                greedy_spellings.append(normalized)
        note_spelling(emissions)
        candidate, offensive = _reserved_from_emissions(
            emissions, forward, "greedy", layout,
            known_offensive=known_offensive, lexicon_by_word=lexicon_by_word,
            language=language, skip_keys=empty_skip,
        )
        accept(candidate, offensive)
    for emissions, forward in greedy_alt_unconstrained_emissions(logits, seen=greedy_emissions):
        note_spelling(emissions)
        candidate, offensive = _reserved_from_emissions(
            emissions, forward, "greedy_alts", layout,
            known_offensive=known_offensive, lexicon_by_word=lexicon_by_word,
            language=language, skip_keys=empty_skip,
        )
        accept(candidate, offensive)
    if n_best > 1:
        for emissions, forward, rank in nbest_unconstrained_emissions(
            logits, n_best=n_best, beam_width=beam_width, seen=greedy_emissions,
        ):
            note_spelling(emissions)
            candidate, offensive = _reserved_from_emissions(
                emissions, forward, "nbest", layout,
                known_offensive=known_offensive, lexicon_by_word=lexicon_by_word,
                language=language, skip_keys=empty_skip, nbest_rank=rank,
            )
            accept(candidate, offensive)
    if include_lexicon_neighbors:
        for spelling in dict.fromkeys(greedy_spellings):
            for entry in lexicon_neighbors(spelling, lexicon_by_word):
                if _normalize(entry.word) in known_offensive:
                    rejected += 1
                    continue
                key = (_normalize(entry.word), entry.language)
                if key in per_source_keys["neighbors"]:
                    continue
                per_source_keys["neighbors"].add(key)
                sources["neighbors"].append(
                    ReservedCandidate(
                        entry, "neighbors", decoder_spatial=decoder_spatial_by_key.get(key),
                    )
                )
        sources["neighbors"].sort(
            key=lambda candidate: (
                -(candidate.decoder_spatial if candidate.decoder_spatial is not None else float("-inf")),
                -candidate.entry.frequency,
                _normalize(candidate.entry.word),
            )
        )
    if include_truncated:
        leftover_skip = {
            (_normalize(candidate.word), candidate.entry.language)
            for candidate in (baseline_merged or existing)
        }
        for source_name, slate in (("truncated_ctc", ctc), ("truncated_geometry", geometric)):
            if not slate:
                continue
            normalized = _z_normalize([candidate.spatial for candidate in slate])
            for candidate, score in zip(slate, normalized, strict=True):
                key = (_normalize(candidate.word), candidate.entry.language)
                if key in leftover_skip or key in per_source_keys[source_name]:
                    continue
                per_source_keys[source_name].add(key)
                sources[source_name].append(
                    ReservedCandidate(candidate.entry, source_name, decoder_spatial=score),
                )
    return sources, rejected, spellings


def first_recovered_source(
    target: str,
    baseline_words: Iterable[str],
    sources: dict[str, list[ReservedCandidate]],
) -> str | None:
    """Label the first source that recovered ``target``. Construction never calls this."""
    needle = _normalize(target)
    if needle in {_normalize(word) for word in baseline_words}:
        return None
    for source in RESERVED_SOURCE_ORDER:
        for candidate in sources.get(source, ()):
            if _normalize(candidate.entry.word) == needle:
                return source
    return None


def cumulative_recovery_stage(
    target: str,
    sources: dict[str, list[ReservedCandidate]],
) -> dict[str, Any]:
    """Membership of ``target`` in cheaper-to-richer source prefixes. Labeling only."""
    needle = _normalize(target)

    def has(source: str) -> bool:
        return any(_normalize(candidate.entry.word) == needle for candidate in sources.get(source, ()))

    greedy = has("greedy")
    alts = greedy or has("greedy_alts")
    ranks = [
        candidate.nbest_rank
        for candidate in sources.get("nbest", ())
        if _normalize(candidate.entry.word) == needle and candidate.nbest_rank
    ]
    min_rank = min(ranks) if ranks else None
    beam = {str(width): alts or (min_rank is not None and min_rank <= width) for width in range(2, 33)}
    neighbors = beam["32"] or has("neighbors")
    dropped = neighbors or has("truncated_ctc") or has("truncated_geometry")
    return {
        "greedyOnly": greedy,
        "greedyAlts": alts,
        "beamLe": beam,
        "plusNeighbors": neighbors,
        "plusDropped": dropped,
        "nbestRank": min_rank,
        "sourcesPresent": [source for source in RESERVED_SOURCE_ORDER if has(source)],
    }


def flatten_reserved_sources(
    sources: dict[str, list[ReservedCandidate]],
    allowed: Sequence[str] | None = None,
    *,
    nbest_rank_limit: int | None = None,
    skip_keys: Iterable[tuple[str, str]] = (),
) -> list[ReservedCandidate]:
    """Deterministic reserved union. Does not take evaluation targets."""
    names = tuple(allowed) if allowed is not None else RESERVED_SOURCE_ORDER
    skipped = set(skip_keys)
    result: list[ReservedCandidate] = []
    seen: set[tuple[str, str]] = set(skipped)
    for source in names:
        for candidate in sources.get(source, ()):
            if (
                source == "nbest"
                and nbest_rank_limit is not None
                and (candidate.nbest_rank is None or candidate.nbest_rank > nbest_rank_limit)
            ):
                continue
            key = (_normalize(candidate.entry.word), candidate.entry.language)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)
    return result


def score_reserved_candidate(
    candidate: ReservedCandidate,
    *,
    calibration: OovScoreCalibration,
    ctc_spatials: Sequence[float],
    oov_conservative: bool = False,
    oov_map_blend: float = 0.0,
    lexicon_spatials: Sequence[float] = (),
) -> ScoredLexiconEntry:
    """Map a reserved spelling onto the published lexicon spatial scale.

    CTC-scale evidence is converted onto ``lexicon_spatials`` (merged z-scale)
    so in-lexicon reserved can use real frequency without a scale mismatch.
    No fabricated frequency.
    """
    true_oov = candidate.source in {"greedy", "greedy_alts", "nbest"} and candidate.entry.frequency == 0
    dest = list(lexicon_spatials) if lexicon_spatials else list(ctc_spatials)

    def onto_dest_from_ctc(raw: float) -> float:
        z_value = _z_score_against(raw, ctc_spatials) if ctc_spatials else raw
        return map_z_onto_pool(z_value, dest) if dest else z_value

    if candidate.decoder_spatial is not None and not true_oov:
        spatial = map_z_onto_pool(candidate.decoder_spatial, dest) if dest else candidate.decoder_spatial
    elif true_oov and candidate.forward is not None and not oov_conservative:
        mapped = project_onto_lexicon_scale(
            calibration.to_lexicon_spatial(candidate.forward), ctc_spatials,
        )
        if oov_map_blend > 0.0 and dest:
            conservative = conservative_lexicon_spatial(dest)
            weight = min(1.0, oov_map_blend)
            spatial = (1.0 - weight) * onto_dest_from_ctc(mapped) + weight * conservative
        else:
            spatial = onto_dest_from_ctc(mapped)
    elif true_oov:
        spatial = conservative_lexicon_spatial(dest)
    elif candidate.forward is not None:
        mapped = project_onto_lexicon_scale(
            calibration.to_lexicon_spatial(candidate.forward), ctc_spatials,
        )
        spatial = onto_dest_from_ctc(mapped)
    else:
        spatial = median_lexicon_spatial(dest)
    return ScoredLexiconEntry(
        candidate.entry,
        spatial,
        frequency_free=true_oov,
        source=candidate.source,
        oov_map_blend=oov_map_blend if true_oov else 0.0,
        nbest_rank=candidate.nbest_rank,
    )


def _oov_blend_by_margin(
    candidates: Sequence[ReservedCandidate],
    base_blend: float,
) -> dict[int, float]:
    """Greedy true OOV use the full OLS map; alts/n-best keep ``base_blend``. No targets.

    Unclamped OLS on a second reserved alt stole converting greedy (21→15) and
    frozen top-3 (5,566→5,551). Alts stay blended; they are not unique-best OLS.
    """
    weights = {id(candidate): base_blend for candidate in candidates}
    for candidate in candidates:
        if candidate.source == "greedy" and candidate.entry.frequency == 0:
            weights[id(candidate)] = 0.0
    return weights


def score_reserved_sources(
    sources: dict[str, list[ReservedCandidate]],
    *,
    calibration: OovScoreCalibration,
    ctc_spatials: Sequence[float],
    oov_conservative: bool = False,
    oov_map_blend: float = 0.0,
    lexicon_spatials: Sequence[float] = (),
    allowed: Sequence[str] | None = None,
    nbest_rank_limit: int | None = None,
    skip_keys: Iterable[tuple[str, str]] = (),
) -> list[ScoredLexiconEntry]:
    flattened = flatten_reserved_sources(
        sources, allowed, nbest_rank_limit=nbest_rank_limit, skip_keys=skip_keys,
    )
    blends = _oov_blend_by_margin(flattened, oov_map_blend)
    return [
        score_reserved_candidate(
            candidate,
            calibration=calibration,
            ctc_spatials=ctc_spatials,
            oov_conservative=oov_conservative,
            oov_map_blend=blends[id(candidate)],
            lexicon_spatials=lexicon_spatials,
        )
        for candidate in flattened
    ]


def reserved_oov_from_logits(
    logits: Sequence[Sequence[float]],
    layout: dict[str, Any],
    *,
    calibration: OovScoreCalibration,
    n_best: int,
    beam_width: int,
    existing: Iterable[ScoredLexiconEntry],
    known_offensive: set[str],
    lexicon_by_word: dict[str, LexiconEntry],
    language: str,
    ctc_spatials: Sequence[float],
    include_lexicon_neighbors: bool = False,
    ctc: Sequence[ScoredLexiconEntry] = (),
    geometric: Sequence[ScoredLexiconEntry] = (),
    baseline_merged: Sequence[ScoredLexiconEntry] = (),
    include_truncated: bool = False,
    oov_conservative: bool = False,
    oov_map_blend: float = 0.0,
    allowed_sources: Sequence[str] | None = None,
    nbest_rank_limit: int | None = None,
) -> tuple[list[ScoredLexiconEntry], int]:
    """Build append-only reserved-slot candidates. Does not take evaluation targets."""
    sources, rejected, _spellings = collect_reserved_sources(
        logits,
        layout,
        n_best=n_best,
        beam_width=beam_width,
        existing=existing,
        known_offensive=known_offensive,
        lexicon_by_word=lexicon_by_word,
        language=language,
        ctc=ctc,
        geometric=geometric,
        baseline_merged=baseline_merged,
        include_lexicon_neighbors=include_lexicon_neighbors,
        include_truncated=include_truncated,
    )
    skip_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in existing}
    lexicon_spatials = [candidate.spatial for candidate in existing]
    reserved = score_reserved_sources(
        sources,
        calibration=calibration,
        ctc_spatials=ctc_spatials,
        oov_conservative=oov_conservative,
        oov_map_blend=oov_map_blend,
        lexicon_spatials=lexicon_spatials,
        allowed=allowed_sources,
        nbest_rank_limit=nbest_rank_limit,
        skip_keys=skip_keys,
    )
    return reserved, rejected


def append_reserved_slots(
    base: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    """Append reserved candidates without dropping any existing base candidate."""
    seen = {(_normalize(candidate.word), candidate.entry.language) for candidate in base}
    extra = []
    for candidate in reserved:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            continue
        seen.add(key)
        extra.append(candidate)
    return [*base, *extra]


def static_fusion_values(candidates: Sequence[ScoredLexiconEntry]) -> list[float]:
    """Lexicon-only static fusion values. No evaluation targets."""
    if not candidates:
        return []
    spatial = _z_normalize([candidate.spatial for candidate in candidates])
    frequencies = _z_normalize([math.log1p(candidate.entry.frequency) for candidate in candidates])
    return [space + freq * 0.65 for space, freq in zip(spatial, frequencies, strict=True)]


def reserved_fusion_values(
    lexicon_candidates: Sequence[ScoredLexiconEntry],
    reserved_candidates: Sequence[ScoredLexiconEntry],
) -> list[float]:
    """Reserved fusion against the lexicon z-pools. Frequency-free uses spatial only."""
    if not reserved_candidates:
        return []
    if not lexicon_candidates:
        return static_fusion_values(reserved_candidates)
    lexicon_spatials = [candidate.spatial for candidate in lexicon_candidates]
    lexicon_log_freq = [math.log1p(candidate.entry.frequency) for candidate in lexicon_candidates]
    values = []
    for candidate in reserved_candidates:
        spatial_z = _z_score_against(candidate.spatial, lexicon_spatials)
        if candidate.frequency_free:
            values.append(spatial_z)
        else:
            values.append(
                spatial_z
                + 0.65 * _z_score_against(math.log1p(candidate.entry.frequency), lexicon_log_freq)
            )
    return values


def lexicon_top3_fusion_floor(lexicon: Sequence[ScoredLexiconEntry]) -> float:
    """Third-best lexicon static fusion. No evaluation targets."""
    values = sorted(static_fusion_values(lexicon), reverse=True)
    if not values:
        return 0.0
    return values[min(2, len(values) - 1)]


def reserved_gap_to_lexicon_top3(
    candidate: ScoredLexiconEntry,
    lexicon: Sequence[ScoredLexiconEntry],
) -> float:
    """Reserved fusion minus the lexicon top-3 floor. No evaluation targets."""
    gaps = reserved_fusion_values(lexicon, [candidate])
    if not gaps:
        return 0.0
    return gaps[0] - lexicon_top3_fusion_floor(lexicon)


def ols_reserved_candidate(
    candidate: ScoredLexiconEntry,
    lexicon: Sequence[ScoredLexiconEntry],
) -> ScoredLexiconEntry:
    """Unblend a frequency-free reserved spatial onto the train-fit map. No targets."""
    if not candidate.frequency_free:
        return candidate
    dest = [item.spatial for item in lexicon]
    spatial = unblend_oov_spatial(candidate.spatial, dest, candidate.oov_map_blend)
    return ScoredLexiconEntry(
        candidate.entry, spatial, frequency_free=True, source=candidate.source,
        oov_map_blend=0.0,
    )


def reserved_clears_lexicon_top3(
    candidate: ScoredLexiconEntry,
    lexicon: Sequence[ScoredLexiconEntry],
) -> bool:
    """True if OLS (frequency-free) or current fusion (in-lexicon) meets the floor.

    Construction never reads evaluation targets. In-lexicon reserved use real
    frequency; true OOV unblend to the train-fit map with no floor clamp.
    """
    scored = ols_reserved_candidate(candidate, lexicon) if candidate.frequency_free else candidate
    return reserved_gap_to_lexicon_top3(scored, lexicon) >= 0.0


def converting_best_reserved(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
) -> ScoredLexiconEntry | None:
    """Highest-OLS frequency-free occupant that clears the lexicon top-3 floor.

    One converting OOV seat: pick the leftover alt/n-best when its OLS fusion
    beats converting greedy, without a second OOV in top-3. No targets.
    """
    best: tuple[float, str, str, ScoredLexiconEntry] | None = None
    for candidate in ordered:
        if not candidate.frequency_free:
            continue
        ols = ols_reserved_candidate(candidate, lexicon)
        gap = reserved_gap_to_lexicon_top3(ols, lexicon)
        if gap < 0.0:
            continue
        fusion = reserved_fusion_values(lexicon, [ols])[0]
        key = (
            fusion,
            1 if candidate.source == "greedy" else 0,
            _normalize(candidate.word),
            candidate.entry.language,
            candidate,
        )
        if best is None or (key[0], key[1], key[2], key[3]) > (best[0], best[1], best[2], best[3]):
            best = key
    return None if best is None else best[4]


def unblend_oov_spatial(spatial: float, dest: Sequence[float], blend: float) -> float:
    """Undo a blend toward the 25th-percentile spatial. No evaluation targets."""
    if blend <= 0.0 or not dest:
        return spatial
    weight = min(1.0, blend)
    if weight >= 1.0:
        return conservative_lexicon_spatial(dest)
    conservative = conservative_lexicon_spatial(dest)
    return (spatial - weight * conservative) / (1.0 - weight)


def prioritize_reserved_candidates(
    reserved: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Mix unconstrained CTC OOV with in-lexicon recoveries.

    Frozen targets in the published 31 all sit in ranks 1-23, so up to eight
    reserved slots can replace the tail without dropping a frozen membership
    hit. Fill greedy, then the best unconstrained OOV alts/n-best, then
    in-lexicon neighbors/truncated by spatial and frequency. Budget 12 uses
    extra_oov = 11 so oovCtcRank 11 (closest convertingFillLoss) can occupy
    the 12th reserved seat. Construction never reads evaluation targets.
    """
    if not any(candidate.source for candidate in reserved):
        return list(reserved)
    greedy = [candidate for candidate in reserved if candidate.source == "greedy"]
    ctc_alts = [
        candidate for candidate in reserved if candidate.source in {"greedy_alts", "nbest"}
    ]
    in_lexicon_ctc = [candidate for candidate in ctc_alts if not candidate.frequency_free]
    oov_ctc = [candidate for candidate in ctc_alts if candidate.frequency_free]
    other_in_lexicon = [
        candidate for candidate in reserved
        if candidate.source not in {"greedy", "greedy_alts", "nbest"}
        and not candidate.frequency_free
    ]

    def in_lexicon_key(candidate: ScoredLexiconEntry) -> tuple:
        return (-candidate.spatial, -candidate.entry.frequency, _normalize(candidate.word))

    in_lexicon_ctc.sort(key=in_lexicon_key)
    other_in_lexicon.sort(key=in_lexicon_key)
    oov_ctc.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    extra_oov = max(0, extra_oov)
    return [
        *greedy,
        *oov_ctc[:extra_oov],
        *in_lexicon_ctc,
        *other_in_lexicon,
        *oov_ctc[extra_oov:],
    ]


def length_changing_converting_extra_oov(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Hold spatial n-best extra_oov; leftover extra_oov seats prefer length-changing converting greedy_alts.

    OLS-gap extra_oov among blend-0.5 alts matches spatial order, so leftover
    convertingFillLoss (oovCtcRank>=11) cannot enter extra_oov by OLS. Length-
    changing converting greedy_alts can occupy remaining extra_oov seats without
    dropping unique n-best, leftover-greedy-alts-append, extra OLS, or raising
    reserved budget. Construction never reads evaluation targets.
    """
    extra_oov = max(0, extra_oov)
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    greedy_len = len(_normalize(greedy[0].word)) if greedy else 0
    oov_ctc = [
        candidate for candidate in ordered
        if candidate.source in {"greedy_alts", "nbest"} and candidate.frequency_free
    ]
    oov_ctc.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    nbest_held = [candidate for candidate in oov_ctc[:extra_oov] if candidate.source == "nbest"]
    converting_alts = [
        candidate for candidate in oov_ctc
        if candidate.source == "greedy_alts"
        and reserved_clears_lexicon_top3(candidate, lexicon)
    ]
    converting_alts.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    length_changing = [
        candidate for candidate in converting_alts
        if greedy_len and len(_normalize(candidate.word)) != greedy_len
    ]
    same_length = [
        candidate for candidate in converting_alts
        if not greedy_len or len(_normalize(candidate.word)) == greedy_len
    ]
    alt_slots = max(0, extra_oov - len(nbest_held))
    held_alts = length_changing[:alt_slots]
    if len(held_alts) < alt_slots:
        held_alts.extend(same_length[: alt_slots - len(held_alts)])
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(candidate: ScoredLexiconEntry) -> None:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in greedy:
        take(candidate)
    for candidate in nbest_held:
        take(candidate)
    for candidate in held_alts:
        take(candidate)
    for candidate in ordered:
        take(candidate)
    return result


def leftover_converting_after_extra_oov(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Keep spatial extra_oov/n-best; leftover seats prefer leftover converting greedy_alts.

    Length-changing extra_oov dropped unique extra_oov (oovCtcRank 7) and a unique
    neighbor to seat oovCtcRank 11. Leftover in-lex seats after extra_oov can hold
    window-losing converting greedy_alts without unclamping extras, raising budget,
    leftover-greedy-alts-append, or protect-drop. Construction never reads targets.
    """
    extra_oov = max(0, extra_oov)
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    oov_ctc = [
        candidate for candidate in ordered
        if candidate.source in {"greedy_alts", "nbest"} and candidate.frequency_free
    ]
    oov_ctc.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    spatial_extra = oov_ctc[:extra_oov]
    leftover = window_losing_converting_greedy_alts(
        ordered, lexicon, extra_oov=extra_oov,
    )
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(candidate: ScoredLexiconEntry) -> None:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in greedy:
        take(candidate)
    for candidate in spatial_extra:
        take(candidate)
    for candidate in leftover:
        take(candidate)
    for candidate in ordered:
        take(candidate)
    return result


def tiny_nbest_truncated_leftover_fill(
    ordered: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
    nbest_hold: int = TINY_NBEST_HOLD,
) -> list[ScoredLexiconEntry]:
    """Hold extra_oov greedy_alts and n-best rank<=4; leftover seats truncated then n-best 5+.

    Tiny extra n-best 5-8 and truncated CTC/geometry leftovers occupy leftover
    32-slot seats after unique extra_oov/n-best. Not an unbounded bag, leftover-
    greedy-alts-append, extra OLS, or frozen rank 1-23 raise. Construction never
    reads evaluation targets.
    """
    extra_oov = max(0, extra_oov)
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    oov_ctc = [
        candidate for candidate in ordered
        if candidate.source in {"greedy_alts", "nbest"} and candidate.frequency_free
    ]
    oov_ctc.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    eligible = [
        candidate for candidate in oov_ctc
        if candidate.source == "greedy_alts"
        or (
            candidate.source == "nbest"
            and candidate.nbest_rank is not None
            and candidate.nbest_rank <= nbest_hold
        )
    ]
    extra = eligible[:extra_oov]
    extra_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in extra}
    leftover_alts = [
        candidate for candidate in oov_ctc
        if candidate.source == "greedy_alts"
        and (_normalize(candidate.word), candidate.entry.language) not in extra_keys
    ]
    leftover_nbest = [
        candidate for candidate in oov_ctc
        if candidate.source == "nbest"
        and (candidate.nbest_rank is None or candidate.nbest_rank > nbest_hold)
    ]
    truncated = [
        candidate for candidate in ordered
        if candidate.source in {"truncated_ctc", "truncated_geometry"}
    ]
    truncated.sort(key=lambda candidate: (-candidate.spatial, -candidate.entry.frequency, _normalize(candidate.word)))
    leftover_nbest.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(candidate: ScoredLexiconEntry) -> None:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in greedy:
        take(candidate)
    for candidate in extra:
        take(candidate)
    for candidate in leftover_alts:
        take(candidate)
    for candidate in truncated:
        take(candidate)
    for candidate in leftover_nbest:
        take(candidate)
    for candidate in ordered:
        take(candidate)
    return result


def converting_inlex_after_extra_oov(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Keep spatial extra_oov; leftover seats prefer in-lex that clear top-3.

    Neighbors of greedy/alts plus truncated leftovers grew competing 5,798→5,818
    but published-31 5,776→5,775, top-3 5,567→5,566, frozenLost 2→5. Not the
    default fill. Construction never reads evaluation targets.
    """
    extra_oov = max(0, extra_oov)
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    spatial_extra = [
        candidate for candidate in ordered
        if candidate.source in {"greedy_alts", "nbest"} and candidate.frequency_free
    ][:extra_oov]
    converting_inlex = [
        candidate for candidate in ordered
        if candidate.source != "greedy"
        and not candidate.frequency_free
        and reserved_clears_lexicon_top3(candidate, lexicon)
    ]
    converting_inlex.sort(
        key=lambda candidate: (
            -candidate.spatial, -candidate.entry.frequency, _normalize(candidate.word),
        )
    )
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(candidate: ScoredLexiconEntry) -> None:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in greedy:
        take(candidate)
    for candidate in spatial_extra:
        take(candidate)
    for candidate in converting_inlex:
        take(candidate)
    for candidate in ordered:
        take(candidate)
    return result


def ablation_extra_fill(
    reserved: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    """Greedy first, then round-robin extras in ablation source order. No targets.

    Budget-8 spatial fill leaves neighbors at fill rank 9 and leftover greedy_alts
    at 19+ because extra_oov[:7] saturates the 8-slot tail. Round-robin after
    greedy seats unpublished neighbors/truncated without a second greedy slot.
    """
    greedy = [candidate for candidate in reserved if candidate.source == "greedy"]
    buckets = {source: [] for source in ABLATION_EXTRA_SOURCES}
    other: list[ScoredLexiconEntry] = []
    for candidate in reserved:
        if candidate.source == "greedy":
            continue
        if candidate.source in buckets:
            buckets[candidate.source].append(candidate)
        else:
            other.append(candidate)
    extras: list[ScoredLexiconEntry] = []
    while any(buckets.values()):
        for source in ABLATION_EXTRA_SOURCES:
            if buckets[source]:
                extras.append(buckets[source].pop(0))
    return [*greedy, *extras, *other]


def ablation_first_source_fill(
    reserved: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Greedy, greedy_alts, neighbors/truncated, n-best last. No targets.

    Ablation first-source of the 184: greedy 41, greedy_alts 69, n-best 33,
    neighbors 36, truncated 5. Spatial extra_oov mixes 0-converting n-best into
    the extra seats so leftover greedy_alts (fillRank > 11) and neighbors
    publish 0. Length-changing greedy_alts are preferred; leftover alts occupy
    seats before n-best. One extra_oov seat is left for in-lex neighbors when
    those sources are present. Construction never reads evaluation targets.
    """
    greedy = [candidate for candidate in reserved if candidate.source == "greedy"]
    greedy_len = len(_normalize(greedy[0].word)) if greedy else 0
    alts = [
        candidate for candidate in reserved
        if candidate.source == "greedy_alts" and candidate.frequency_free
    ]
    alts.sort(
        key=lambda candidate: (
            0 if greedy_len and len(_normalize(candidate.word)) != greedy_len else 1,
            -candidate.spatial,
            _normalize(candidate.word),
        )
    )
    neighbors = [
        candidate for candidate in reserved
        if candidate.source in {"neighbors", "truncated_ctc", "truncated_geometry"}
    ]
    neighbors.sort(
        key=lambda candidate: (
            -candidate.spatial, -candidate.entry.frequency, _normalize(candidate.word),
        )
    )
    nbest = [
        candidate for candidate in reserved
        if candidate.source == "nbest" and candidate.frequency_free
    ]
    nbest.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    alt_window = extra_oov - 1 if neighbors else extra_oov
    alt_window = max(1, alt_window)
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(items: Sequence[ScoredLexiconEntry]) -> None:
        for candidate in items:
            key = (_normalize(candidate.word), candidate.entry.language)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)

    take(greedy)
    take(alts[:alt_window])
    take(neighbors)
    take(alts[alt_window:])
    take(nbest)
    take(reserved)
    return result


def leftover_inlex_after_converting_oov(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Greedy, converting extra-OOV, leftover extra seats to in-lex neighbors/truncated.

    extra_oov spatial OOV saturates budget 8, so neighbors publish 0. Converting
    OOV keep up to extra_oov extra seats; unused extra_oov slots go to in-lex
    ablation winners before non-converting n-best. No evaluation targets.
    """
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    converting_oov = [
        candidate for candidate in ordered
        if candidate.source in {"greedy_alts", "nbest"}
        and candidate.frequency_free
        and reserved_clears_lexicon_top3(candidate, lexicon)
    ]
    converting_oov.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    in_lex = [
        candidate for candidate in ordered
        if not candidate.frequency_free and candidate.source != "greedy"
    ]
    in_lex.sort(
        key=lambda candidate: (
            -candidate.spatial, -candidate.entry.frequency, _normalize(candidate.word),
        )
    )
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(items: Sequence[ScoredLexiconEntry]) -> None:
        for candidate in items:
            key = (_normalize(candidate.word), candidate.entry.language)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)

    take(greedy)
    take(converting_oov[:extra_oov])
    take(in_lex)
    take(converting_oov[extra_oov:])
    take(ordered)
    return result


def converting_leftover_extras(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    """Greedy first; leftover extra seats only converting frequency-free OOV. No targets.

    Neighbors, truncated leftovers, and non-converting n-best did not enter top-3
    on spatial budget-8 park. Extra seats keep OLS-clearing greedy_alts/nbest.
    Unused extra budget stays lexicon tail.
    """
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    converting = [
        candidate for candidate in ordered
        if candidate.source in {"greedy_alts", "nbest"}
        and candidate.frequency_free
        and reserved_clears_lexicon_top3(candidate, lexicon)
    ]
    converting.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(items: Sequence[ScoredLexiconEntry]) -> None:
        for candidate in items:
            key = (_normalize(candidate.word), candidate.entry.language)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)

    take(greedy)
    take(converting)
    return result


def converting_alt_expand_fill(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    base_budget: int,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Fill extra reserved seats with converting greedy_alts without evicting frozen ranks.

    Unconstrained expand (keep 3 lexicon ranks) frozeLost 7 / targetsLostAtPublish 10.
    Frozen published-31 targets sit in ranks 1-23, so extra converting-alt seats cannot
    grow past ``base_budget`` (8 reserved / 23 lexicon). Extra seats prefer converting
    greedy_alts over non-converting n-best. OLS-if-clears and leftover in-lex are unused.
    No evaluation targets.
    """
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    converting = [
        candidate for candidate in ordered
        if candidate.source == "greedy_alts"
        and candidate.frequency_free
        and reserved_clears_lexicon_top3(candidate, lexicon)
    ]
    converting.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(items: Sequence[ScoredLexiconEntry]) -> None:
        for candidate in items:
            key = (_normalize(candidate.word), candidate.entry.language)
            if key in seen:
                continue
            seen.add(key)
            result.append(candidate)

    take(greedy)
    take(converting)
    take(ordered)
    protected = PUBLISHED_RANKING_BOUND - base_budget
    maximum = PUBLISHED_RANKING_BOUND - protected
    return result[: min(base_budget, maximum)]


CONVERTING_FILL_LOSS_MIN_LEXICON = 20
FROZEN_PROTECTED_RANKS = 23


def protect_frozen_converting_32_fill(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """32-slot occupants: greedy, spatial extra_oov, leftover converting alts.

    Keeper unpublished hits are fillRank>11 converting greedy_alts and frozenLost
    adaptive_merged in ranks 21-23. Nine reserved seats occupy merged ranks 24-32
    so lexicon fusion ranks 1-23 stay in the 32-set. Spatial extra_oov keeps unique
    n-best; leftover converting greedy_alts take the ninth seat. Ranking publishes
    [:31]. leftover-greedy-alts-append stays off. No evaluation targets.
    """
    maximum = PUBLISHED_SLATE_BOUND - FROZEN_PROTECTED_RANKS
    extra_oov = max(0, extra_oov)
    spatial = list(ordered[: min(maximum, extra_oov + 1)])
    seen = {
        (_normalize(candidate.word), candidate.entry.language) for candidate in spatial
    }
    result = list(spatial)

    def take(candidate: ScoredLexiconEntry) -> None:
        if len(result) >= maximum:
            return
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in window_losing_converting_greedy_alts(
        ordered, lexicon, extra_oov=extra_oov,
    ):
        take(candidate)
    for candidate in ordered:
        take(candidate)
    return result[:maximum]


def leftover_converting_greedy_neighbors(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    skip_keys: Iterable[tuple[str, str]] = (),
) -> list[ScoredLexiconEntry]:
    """In-lexicon neighbors-of-greedy that clear the lexicon floor. No targets.

    Neighbors are not frequency-free and are not parked, so a floor-clearing
    neighbor can convert. Construction never CTC-forwards neighbors.
    """
    skipped = set(skip_keys)
    leftover: list[ScoredLexiconEntry] = []
    for candidate in ordered:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in skipped:
            continue
        if candidate.source != "neighbors" or candidate.frequency_free:
            continue
        if reserved_clears_lexicon_top3(candidate, lexicon):
            leftover.append(candidate)
    leftover.sort(
        key=lambda candidate: (
            -candidate.spatial, -candidate.entry.frequency, _normalize(candidate.word),
        )
    )
    return leftover


def converting_fill_loss_append_fill(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    base_budget: int,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Window-losers take non-converting extra_oov seats, then leftover converting alts.

    Keep-20 (at most 11 reserved). Spatial extra_oov stays greedy-first (unique
    n-best seats held; not n-best-last). ``prefer_window_losing_converting_alts``
    reorders that prefix so converting extra_oov greedy_alts/nbest keep seats and
    only non-converting extra_oov give way to window-losing converting greedy_alts.
    Leftover converting greedy_alts fill the keep-20 remainder before leftover
    converting neighbors-of-greedy (neighbors convert 0 top-3). No leftover-greedy-alts-append,
    CTC-forward neighbors, OLS-if-clears, or floor clamp. Construction never reads
    evaluation targets.
    """
    reordered = prefer_window_losing_converting_alts(
        ordered, lexicon, extra_oov=extra_oov,
    )
    spatial = list(reordered[: max(0, base_budget)])
    seen = {
        (_normalize(candidate.word), candidate.entry.language) for candidate in spatial
    }
    result = list(spatial)
    maximum = PUBLISHED_RANKING_BOUND - CONVERTING_FILL_LOSS_MIN_LEXICON

    def take(candidate: ScoredLexiconEntry) -> None:
        if len(result) >= maximum:
            return
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in window_losing_converting_greedy_alts(
        ordered, lexicon, extra_oov=extra_oov,
    ):
        take(candidate)
    for candidate in leftover_converting_greedy_neighbors(ordered, lexicon):
        take(candidate)
    return result


LEFTOVER_GREEDY_ALTS_MIN_LEXICON = 3


def leftover_greedy_alts_append_fill(
    ordered: Sequence[ScoredLexiconEntry],
    *,
    base_budget: int,
) -> list[ScoredLexiconEntry]:
    """Spatial occupants first, then leftover frequency-free greedy_alts. No targets.

    Unpublished competing greedy_alts sit after extra_oov (fillRank > 11 on
    keep-20). This keeps the spatial ``base_budget`` prefix (unique n-best
    seats) and appends leftover greedy_alts up to 3 lexicon ranks so those
    unpublished alts can occupy the published 31. Caller must not park extras
    or floor-clamp if they should convert. No evaluation targets.
    """
    spatial = list(ordered[: max(0, base_budget)])
    seen = {
        (_normalize(candidate.word), candidate.entry.language) for candidate in spatial
    }
    leftover = [
        candidate for candidate in ordered
        if candidate.source == "greedy_alts"
        and candidate.frequency_free
        and (_normalize(candidate.word), candidate.entry.language) not in seen
    ]
    leftover.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    maximum = PUBLISHED_RANKING_BOUND - LEFTOVER_GREEDY_ALTS_MIN_LEXICON
    result = list(spatial)
    for candidate in leftover:
        if len(result) >= maximum:
            break
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def window_losing_converting_greedy_alts(
    reserved: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
    skip_keys: Iterable[tuple[str, str]] = (),
) -> list[ScoredLexiconEntry]:
    """Converting greedy_alts with oovCtcRank > extra_oov. No targets."""
    skipped = set(skip_keys)
    ranks = reserved_oov_ctc_ranks(reserved)
    losers: list[ScoredLexiconEntry] = []
    for candidate in reserved:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in skipped:
            continue
        if candidate.source != "greedy_alts" or not candidate.frequency_free:
            continue
        rank = ranks.get(key)
        if rank is None or rank <= extra_oov:
            continue
        if reserved_clears_lexicon_top3(candidate, lexicon):
            losers.append(candidate)
    losers.sort(
        key=lambda candidate: (
            -reserved_gap_to_lexicon_top3(
                ols_reserved_candidate(candidate, lexicon), lexicon,
            ),
            -candidate.spatial,
            _normalize(candidate.word),
        )
    )
    return losers


def prefer_window_losing_converting_alts(
    ordered: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    extra_oov: int = EXTRA_OOV_FILL,
) -> list[ScoredLexiconEntry]:
    """Window-losing converting greedy_alts displace non-converting extras. No targets.

    Converting extra_oov winners (greedy_alts or nbest with oovCtcRank <= extra_oov
    that clear the lexicon floor) keep their extra seats. Non-converting extra_oov
    (typically nbest) give those seats to converting greedy_alts that lost the
    extra_oov window. Construction never reads evaluation targets.
    """
    greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
    extras = [candidate for candidate in ordered if candidate.source != "greedy"]
    losers = window_losing_converting_greedy_alts(ordered, lexicon, extra_oov=extra_oov)
    ranks = reserved_oov_ctc_ranks(ordered)
    seen: set[tuple[str, str]] = set()
    result: list[ScoredLexiconEntry] = []

    def take(candidate: ScoredLexiconEntry) -> None:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen:
            return
        seen.add(key)
        result.append(candidate)

    for candidate in greedy:
        take(candidate)
    loser_index = 0
    for candidate in extras:
        key = (_normalize(candidate.word), candidate.entry.language)
        extra_oov_winner = (ranks.get(key) or 0) <= extra_oov
        converting_winner = (
            candidate.source in {"greedy_alts", "nbest"}
            and candidate.frequency_free
            and extra_oov_winner
            and reserved_clears_lexicon_top3(candidate, lexicon)
        )
        if converting_winner:
            take(candidate)
            continue
        if loser_index < len(losers):
            take(losers[loser_index])
            loser_index += 1
            continue
        take(candidate)
    for candidate in losers[loser_index:]:
        take(candidate)
    for candidate in extras:
        take(candidate)
    return result


def reserved_occupants(
    reserved: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    reserved_budget: int,
    skip_keys: Iterable[tuple[str, str]] = (),
    converting_best: bool = False,
    prefer_converting_greedy_alts: bool = False,
    ablation_extra_fill_sources: bool = False,
    leftover_inlex_fill: bool = False,
    converting_leftover_extras_fill: bool = False,
    converting_alt_expand: bool = False,
    ablation_first_source_fill_sources: bool = False,
    converting_fill_loss_append: bool = False,
    leftover_greedy_alts_append: bool = False,
    protect_frozen_ranks: bool = False,
    length_changing_extra_oov: bool = False,
    leftover_converting_after_extra_oov_fill: bool = False,
    tiny_nbest_truncated: bool = False,
) -> list[ScoredLexiconEntry]:
    """Fill-order reserved occupants not already in ``skip_keys``. No targets.

    One seat stays greedy-first unless ``converting_best``. Extra seats stay
    spatial unless leftover in-lex, converting leftover extras, converting-alt
    expand, converting-fill-loss append, ``prefer_converting_greedy_alts``, or
    ``ablation_extra_fill_sources``. A second converting OOV in top-3 stole
    6k 5,567→5,551.
    """
    skipped = set(skip_keys)
    extra_oov = EXTRA_OOV_FILL
    if reserved_budget >= 12:
        extra_oov = max(EXTRA_OOV_FILL, reserved_budget - 1)
    ordered = [
        candidate for candidate in prioritize_reserved_candidates(
            reserved, extra_oov=extra_oov,
        )
        if (_normalize(candidate.word), candidate.entry.language) not in skipped
    ]
    if reserved_budget <= 0:
        return []
    if ablation_extra_fill_sources:
        ordered = ablation_extra_fill(ordered)
    elif ablation_first_source_fill_sources:
        ordered = ablation_first_source_fill(ordered)
    elif leftover_inlex_fill:
        ordered = leftover_inlex_after_converting_oov(ordered, lexicon)
    elif converting_leftover_extras_fill:
        ordered = converting_leftover_extras(ordered, lexicon)
    elif converting_alt_expand:
        return converting_alt_expand_fill(ordered, lexicon, base_budget=reserved_budget)
    elif converting_fill_loss_append:
        return converting_fill_loss_append_fill(
            ordered, lexicon, base_budget=reserved_budget,
        )
    elif leftover_greedy_alts_append:
        return leftover_greedy_alts_append_fill(ordered, base_budget=reserved_budget)
    elif protect_frozen_ranks:
        return protect_frozen_converting_32_fill(ordered, lexicon, extra_oov=extra_oov)
    elif length_changing_extra_oov:
        ordered = length_changing_converting_extra_oov(
            ordered, lexicon, extra_oov=extra_oov,
        )
    elif leftover_converting_after_extra_oov_fill:
        ordered = leftover_converting_after_extra_oov(
            ordered, lexicon, extra_oov=extra_oov,
        )
    elif tiny_nbest_truncated:
        ordered = tiny_nbest_truncated_leftover_fill(ordered, extra_oov=extra_oov)
    if prefer_converting_greedy_alts:
        converting_alt_keys = {
            (_normalize(candidate.word), candidate.entry.language)
            for candidate in ordered
            if candidate.source == "greedy_alts"
            and candidate.frequency_free
            and reserved_clears_lexicon_top3(candidate, lexicon)
        }
        greedy = [candidate for candidate in ordered if candidate.source == "greedy"]
        converting_alts = [
            candidate for candidate in ordered
            if (_normalize(candidate.word), candidate.entry.language) in converting_alt_keys
        ]
        rest = [
            candidate for candidate in ordered
            if candidate.source != "greedy"
            and (_normalize(candidate.word), candidate.entry.language) not in converting_alt_keys
        ]
        ordered = [*greedy, *converting_alts, *rest]
    if converting_best:
        best = converting_best_reserved(ordered, lexicon)
        if best is not None:
            best_key = (_normalize(best.word), best.entry.language)
            rest = [
                candidate for candidate in ordered
                if (_normalize(candidate.word), candidate.entry.language) != best_key
            ]
            return [best, *rest][:reserved_budget]
    return ordered[:reserved_budget]


def extra_reserved_occupant_keys(
    reserved: Sequence[ScoredLexiconEntry],
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    reserved_budget: int,
    skip_keys: Iterable[tuple[str, str]] = (),
    converting_best: bool = False,
    prefer_converting_greedy_alts: bool = False,
    ablation_extra_fill_sources: bool = False,
    leftover_inlex_fill: bool = False,
    converting_leftover_extras_fill: bool = False,
    converting_alt_expand: bool = False,
    ablation_first_source_fill_sources: bool = False,
    converting_fill_loss_append: bool = False,
    leftover_greedy_alts_append: bool = False,
    protect_frozen_ranks: bool = False,
    length_changing_extra_oov: bool = False,
    leftover_converting_after_extra_oov_fill: bool = False,
    tiny_nbest_truncated: bool = False,
) -> set[tuple[str, str]]:
    """Keys of fill-order occupants after the first reserved seat. No targets."""
    occupants = reserved_occupants(
        reserved,
        lexicon,
        reserved_budget=reserved_budget,
        skip_keys=skip_keys,
        converting_best=converting_best,
        prefer_converting_greedy_alts=prefer_converting_greedy_alts,
        ablation_extra_fill_sources=ablation_extra_fill_sources,
        leftover_inlex_fill=leftover_inlex_fill,
        converting_leftover_extras_fill=converting_leftover_extras_fill,
        converting_alt_expand=converting_alt_expand,
        ablation_first_source_fill_sources=ablation_first_source_fill_sources,
        converting_fill_loss_append=converting_fill_loss_append,
        leftover_greedy_alts_append=leftover_greedy_alts_append,
        protect_frozen_ranks=protect_frozen_ranks,
        length_changing_extra_oov=length_changing_extra_oov,
        leftover_converting_after_extra_oov_fill=leftover_converting_after_extra_oov_fill,
        tiny_nbest_truncated=tiny_nbest_truncated,
    )
    return {
        (_normalize(candidate.word), candidate.entry.language)
        for candidate in occupants[1:]
    }


def park_extra_reserved_below_rank(
    ranked: Sequence[tuple[ScoredLexiconEntry, float]],
    extra_keys: Iterable[tuple[str, str]],
    *,
    min_rank: int = 4,
) -> list[tuple[ScoredLexiconEntry, float]]:
    """Keep extra reserved occupants at ``min_rank`` or worse. No targets.

    Hard 2-slot OLS stole frozen top-3 (5,567→5,557). Extra occupants still
    publish in the 31; they cannot occupy top-3, including after other extras
    are demoted. The greedy-first occupant is not in ``extra_keys`` and may
    still convert.
    """
    if min_rank < 2 or not ranked:
        return list(ranked)
    extras = set(extra_keys)
    if not extras:
        return list(ranked)
    rest: list[tuple[ScoredLexiconEntry, float]] = []
    extra_items: list[tuple[ScoredLexiconEntry, float]] = []
    for candidate, fusion in ranked:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in extras:
            extra_items.append((candidate, fusion))
        else:
            rest.append((candidate, fusion))
    if not extra_items:
        return list(ranked)
    keep_top = min_rank - 1
    if len(rest) >= keep_top:
        cap = rest[keep_top - 1][1] - 1e-6
    elif rest:
        cap = rest[-1][1] - 1e-6
    else:
        return list(ranked)
    extra_items.sort(
        key=lambda item: (
            -item[1],
            _normalize(item[0].word),
            item[0].entry.language,
        )
    )
    parked = list(rest)
    for index, (candidate, fusion) in enumerate(extra_items):
        parked.append((candidate, min(fusion, cap - 1e-6 * index)))
    parked.sort(
        key=lambda item: (
            -item[1],
            _normalize(item[0].word),
            item[0].entry.language,
        )
    )
    return parked


def park_extra_frequency_free_to_protect_top3(
    ranked: Sequence[tuple[ScoredLexiconEntry, float]],
    extra_keys: Iterable[tuple[str, str]],
    *,
    min_rank: int = 4,
) -> list[tuple[ScoredLexiconEntry, float]]:
    """Park extra frequency-free occupants only to keep one FF in top-3. No targets.

    Always parking extras forbade conversion of OLS-capable published alts.
    Unparking every extra stole frozen top-3 (5,567→5,551). Unparking one
    converting extra into rank 1 stole 5,566→5,556. Letting one converting
    extra occupy ranks 2-3 stole 5,566→5,547 (below frozen 5,551). In-lexicon
    extras are not frequency-free and are not parked. If greedy already occupies
    top-3, extra FF stay at ``min_rank`` or worse. If greedy does not convert,
    the best extra FF may enter top-3; additional extra FF are still parked.
    """
    if min_rank < 2 or not ranked:
        return list(ranked)
    extras = set(extra_keys)
    if not extras:
        return list(ranked)
    top_n = min_rank - 1
    top = ranked[:top_n]
    extra_ff_keys = {
        (_normalize(candidate.word), candidate.entry.language)
        for candidate, _fusion in ranked
        if candidate.frequency_free
        and (_normalize(candidate.word), candidate.entry.language) in extras
    }
    if not extra_ff_keys:
        return list(ranked)

    def key_of(candidate: ScoredLexiconEntry) -> tuple[str, str]:
        return (_normalize(candidate.word), candidate.entry.language)

    without_extras = [
        (candidate, fusion) for candidate, fusion in ranked
        if key_of(candidate) not in extras
    ]
    primary_converts = any(
        candidate.frequency_free and key_of(candidate) not in extras
        for candidate, _fusion in without_extras[:top_n]
    )
    if primary_converts:
        return park_extra_reserved_below_rank(ranked, extra_ff_keys, min_rank=min_rank)
    extra_ff_in_top = [
        candidate for candidate, _fusion in top
        if candidate.frequency_free and key_of(candidate) in extras
    ]
    if len(extra_ff_in_top) <= 1:
        return list(ranked)
    keep = key_of(extra_ff_in_top[0])
    return park_extra_reserved_below_rank(
        ranked, extra_ff_keys - {keep}, min_rank=min_rank,
    )


def reserved_fill_ranks(
    reserved: Sequence[ScoredLexiconEntry],
) -> dict[tuple[str, str], int]:
    """1-based order in ``prioritize_reserved_candidates``. No evaluation targets."""
    ranks: dict[tuple[str, str], int] = {}
    for index, candidate in enumerate(prioritize_reserved_candidates(reserved), 1):
        key = (_normalize(candidate.word), candidate.entry.language)
        ranks.setdefault(key, index)
    return ranks


def reserved_oov_ctc_ranks(
    reserved: Sequence[ScoredLexiconEntry],
) -> dict[tuple[str, str], int]:
    """1-based spatial rank among frequency-free greedy_alts/nbest. No evaluation targets."""
    oov_ctc = [
        candidate for candidate in reserved
        if candidate.source in {"greedy_alts", "nbest"} and candidate.frequency_free
    ]
    oov_ctc.sort(key=lambda candidate: (-candidate.spatial, _normalize(candidate.word)))
    ranks: dict[tuple[str, str], int] = {}
    for index, candidate in enumerate(oov_ctc, 1):
        key = (_normalize(candidate.word), candidate.entry.language)
        ranks.setdefault(key, index)
    return ranks


def publish_reserved_slots(
    base: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
    *,
    reserved_budget: int,
    bound: int = PUBLISHED_RANKING_BOUND,
    converting_best: bool = False,
    prefer_converting_greedy_alts: bool = False,
    ablation_extra_fill_sources: bool = False,
    leftover_inlex_fill: bool = False,
    converting_leftover_extras_fill: bool = False,
    converting_alt_expand: bool = False,
    ablation_first_source_fill_sources: bool = False,
    converting_fill_loss_append: bool = False,
    leftover_greedy_alts_append: bool = False,
    protect_frozen_ranks: bool = False,
    length_changing_extra_oov: bool = False,
    leftover_converting_after_extra_oov_fill: bool = False,
    tiny_nbest_truncated: bool = False,
) -> list[ScoredLexiconEntry]:
    """Replace the worst ``reserved_budget`` of the ranked 31 with reserved spellings.

    Production keeps 32 scorer slots including the empty swipe placeholder and
    publishes 31 words. Membership is this 31-list. Extra spellings always
    replace the lowest-fusion baseline candidates, matching the greedy-41 31/32
    proof; ranking of these 31 items cannot then drop the replacement.
    Protect-frozen fills the 32-slot union while keeping lexicon fusion ranks
    1-23; ranking still publishes [:31].
    """
    if reserved_budget < 0:
        raise SwipeEvaluationError("reserved budget must not be negative")
    if protect_frozen_ranks:
        bound = PUBLISHED_SLATE_BOUND
        budget = min(reserved_budget, bound - FROZEN_PROTECTED_RANKS)
    else:
        budget = min(reserved_budget, bound)
    ordered_base = [
        candidate for candidate, _fusion in sorted(
            zip(base, static_fusion_values(base), strict=True),
            key=lambda item: (
                -item[1],
                _normalize(item[0].word),
                item[0].entry.language,
            ),
        )
    ] if base else []
    published: list[ScoredLexiconEntry] = []
    seen: set[tuple[str, str]] = set()

    def take(candidate: ScoredLexiconEntry) -> bool:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in seen or len(published) >= bound:
            return False
        published.append(candidate)
        seen.add(key)
        return True

    skip_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in ordered_base}
    occupants = reserved_occupants(
        reserved,
        base,
        reserved_budget=budget,
        skip_keys=skip_keys,
        converting_best=converting_best,
        prefer_converting_greedy_alts=prefer_converting_greedy_alts,
        ablation_extra_fill_sources=ablation_extra_fill_sources,
        leftover_inlex_fill=leftover_inlex_fill,
        converting_leftover_extras_fill=converting_leftover_extras_fill,
        converting_alt_expand=converting_alt_expand,
        ablation_first_source_fill_sources=ablation_first_source_fill_sources,
        converting_fill_loss_append=converting_fill_loss_append,
        leftover_greedy_alts_append=leftover_greedy_alts_append,
        protect_frozen_ranks=protect_frozen_ranks,
        length_changing_extra_oov=length_changing_extra_oov,
        leftover_converting_after_extra_oov_fill=leftover_converting_after_extra_oov_fill,
        tiny_nbest_truncated=tiny_nbest_truncated,
    )
    if converting_alt_expand or converting_fill_loss_append or leftover_greedy_alts_append:
        budget = min(bound, max(budget, len(occupants)))
        budget = len(occupants)
    keep = bound - budget
    for candidate in ordered_base:
        if len(published) >= keep:
            break
        take(candidate)
    added = 0
    for candidate in occupants:
        if added >= budget:
            break
        if take(candidate):
            added += 1
    for candidate in ordered_base:
        if len(published) >= bound:
            break
        take(candidate)
    return published


def competing_slate(
    merged: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    """Unbounded ranking-input union. Not the published-31 recall bound."""
    return append_reserved_slots(merged, reserved)


def reserved_from_decoder_slates(
    ctc: Sequence[ScoredLexiconEntry],
    geometric: Sequence[ScoredLexiconEntry],
    merged: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    """Truncated CTC/geometry candidates on each decoder's z-scale so they can compete."""
    merged_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in merged}
    reserved: list[ScoredLexiconEntry] = []
    seen = set(merged_keys)
    for slate in (ctc, geometric):
        if not slate:
            continue
        normalized = _z_normalize([candidate.spatial for candidate in slate])
        for candidate, score in zip(slate, normalized, strict=True):
            key = (_normalize(candidate.word), candidate.entry.language)
            if key in seen:
                continue
            seen.add(key)
            reserved.append(ScoredLexiconEntry(candidate.entry, score, frequency_free=False))
    return reserved


def split_published_slate(
    published: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
) -> tuple[list[ScoredLexiconEntry], list[ScoredLexiconEntry]]:
    """Split a bounded published slate into lexicon vs reserved origin. No targets."""
    reserved_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in reserved}
    lexicon_part = [
        candidate for candidate in published
        if (_normalize(candidate.word), candidate.entry.language) not in reserved_keys
    ]
    reserved_part = [
        candidate for candidate in published
        if (_normalize(candidate.word), candidate.entry.language) in reserved_keys
    ]
    return lexicon_part, reserved_part


def rank_static_fusion_with_reserved(
    lexicon_candidates: Sequence[ScoredLexiconEntry],
    reserved_candidates: Sequence[ScoredLexiconEntry] = (),
) -> list[LexiconEntry]:
    """Production fusion on the lexicon slate; every reserved candidate competes.

    Lexicon spatial/frequency z-scores are computed without reserved candidates.
    Frequency-free reserved entries use only a spatial z-score against the lexicon
    spatial distribution (no fabricated frequency prior and no top-3 floor clamp).
    Published recall may only count targets present in the returned [:31] list.
    """
    if not reserved_candidates:
        return rank_static_fusion(lexicon_candidates)
    if not lexicon_candidates:
        return rank_static_fusion(reserved_candidates)[:PUBLISHED_RANKING_BOUND]
    spatial = _z_normalize([candidate.spatial for candidate in lexicon_candidates])
    frequencies = _z_normalize([math.log1p(candidate.entry.frequency) for candidate in lexicon_candidates])
    scored: list[tuple[ScoredLexiconEntry, float]] = [
        (candidate, space + freq * 0.65)
        for candidate, space, freq in zip(lexicon_candidates, spatial, frequencies, strict=True)
    ]
    lexicon_spatials = [candidate.spatial for candidate in lexicon_candidates]
    lexicon_log_freq = [math.log1p(candidate.entry.frequency) for candidate in lexicon_candidates]
    for candidate in reserved_candidates:
        spatial_z = _z_score_against(candidate.spatial, lexicon_spatials)
        if candidate.frequency_free:
            fusion = spatial_z
        else:
            fusion = spatial_z + 0.65 * _z_score_against(
                math.log1p(candidate.entry.frequency), lexicon_log_freq,
            )
        scored.append((candidate, fusion))
    ranked = sorted(
        scored,
        key=lambda item: (
            -item[1],
            _normalize(item[0].entry.word),
            item[0].entry.language,
        ),
    )
    return [item[0].entry for item in ranked[:PUBLISHED_RANKING_BOUND]]


def published_ranking_scored(
    published: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
    lexicon_reference: Sequence[ScoredLexiconEntry] | None = None,
) -> list[tuple[ScoredLexiconEntry, float]]:
    """Published 31 with fusion values. Lexicon fusions stay those of ``lexicon_reference``."""
    reserved_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in reserved}
    if not published:
        return []
    reference = list(lexicon_reference) if lexicon_reference is not None else [
        candidate for candidate in published
        if (_normalize(candidate.word), candidate.entry.language) not in reserved_keys
    ]
    if not reference:
        values = static_fusion_values(published)
        ranked = sorted(
            zip(published, values, strict=True),
            key=lambda item: (-item[1], _normalize(item[0].word), item[0].entry.language),
        )
        return list(ranked)
    spatials = [candidate.spatial for candidate in reference]
    log_freq = [math.log1p(candidate.entry.frequency) for candidate in reference]
    reference_fusion = {
        (_normalize(candidate.word), candidate.entry.language): fusion
        for candidate, fusion in zip(reference, static_fusion_values(reference), strict=True)
    }
    scored: list[tuple[ScoredLexiconEntry, float]] = []
    for candidate in published:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key in reference_fusion and key not in reserved_keys:
            fusion = reference_fusion[key]
        elif candidate.frequency_free:
            fusion = _z_score_against(candidate.spatial, spatials)
        else:
            fusion = (
                _z_score_against(candidate.spatial, spatials)
                + 0.65 * _z_score_against(math.log1p(candidate.entry.frequency), log_freq)
            )
        scored.append((candidate, fusion))
    return sorted(
        scored,
        key=lambda item: (
            -item[1],
            _normalize(item[0].entry.word),
            item[0].entry.language,
        ),
    )


def protected_lexicon_fusion_keys(
    lexicon: Sequence[ScoredLexiconEntry],
    *,
    ranks: int = FROZEN_PROTECTED_RANKS,
) -> set[tuple[str, str]]:
    """Keys of the best ``ranks`` lexicon fusion occupants. No evaluation targets."""
    if not lexicon or ranks <= 0:
        return set()
    ordered = [
        candidate for candidate, _fusion in sorted(
            zip(lexicon, static_fusion_values(lexicon), strict=True),
            key=lambda item: (
                -item[1],
                _normalize(item[0].word),
                item[0].entry.language,
            ),
        )
    ]
    return {
        (_normalize(candidate.word), candidate.entry.language)
        for candidate in ordered[:ranks]
    }


def drop_nonconverting_extra_for_published_31(
    ranked: Sequence[tuple[ScoredLexiconEntry, float]],
    lexicon: Sequence[ScoredLexiconEntry],
    extra_keys: Iterable[tuple[str, str]],
) -> list[tuple[ScoredLexiconEntry, float]]:
    """Drop one extra so leftover converting greedy_alts occupy ranked[:31].

    Frozen lexicon fusion ranks 1-23 stay. n-best and neighbors (0 top-3 on the
    keeper) go before leftover converting greedy_alts. leftover-greedy-alts-append
    stays off. Construction never reads evaluation targets.
    """
    if len(ranked) <= PUBLISHED_RANKING_BOUND:
        return list(ranked)
    extras = set(extra_keys)
    protected = protected_lexicon_fusion_keys(lexicon)

    def key_of(candidate: ScoredLexiconEntry) -> tuple[str, str]:
        return (_normalize(candidate.word), candidate.entry.language)

    def drop_priority(candidate: ScoredLexiconEntry) -> int:
        key = key_of(candidate)
        if key in protected or key not in extras:
            return -1
        if candidate.source == "nbest":
            return 4
        if candidate.source == "neighbors":
            return 3
        if not reserved_clears_lexicon_top3(candidate, lexicon):
            return 2
        return 1

    chosen: tuple[int, int] | None = None
    for index, (candidate, _fusion) in enumerate(ranked):
        priority = drop_priority(candidate)
        if priority < 0:
            continue
        if chosen is None or priority > chosen[0] or (priority == chosen[0] and index > chosen[1]):
            chosen = (priority, index)
    drop = chosen[1] if chosen is not None else len(ranked) - 1
    return [item for index, item in enumerate(ranked) if index != drop]


def published_ranking(
    published: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
    lexicon_reference: Sequence[ScoredLexiconEntry] | None = None,
    *,
    extra_park_keys: Iterable[tuple[str, str]] = (),
    park_min_rank: int = 0,
    protect_frozen_ranks: bool = False,
) -> list[LexiconEntry]:
    """Reorder the published 31. Lexicon fusions stay those of ``lexicon_reference``.

    The counted list is this result (length of ``published``, at most 31).
    Extra frequency-free occupants in ``extra_park_keys`` are demoted to
    ``park_min_rank`` or worse only when another frequency-free occupant
    already occupies top-3. In-lexicon extras are not parked. A 32-slot
    protect-frozen fill drops one non-converting extra before leftover
    converting or lexicon ranks 1-23.
    """
    scored = published_ranking_scored(published, reserved, lexicon_reference)
    if park_min_rank >= 2 and extra_park_keys:
        scored = park_extra_frequency_free_to_protect_top3(
            scored, extra_park_keys, min_rank=park_min_rank,
        )
    if (
        protect_frozen_ranks
        and lexicon_reference is not None
        and len(scored) > PUBLISHED_RANKING_BOUND
    ):
        scored = drop_nonconverting_extra_for_published_31(
            scored, lexicon_reference, extra_park_keys,
        )
    return [item[0].entry for item in scored[:PUBLISHED_RANKING_BOUND]]


def lift_near_top3_frequency_free(
    published: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
    lexicon_reference: Sequence[ScoredLexiconEntry],
    *,
    blend: float = 0.5,
    optimism_offset: float = 0.0,
    max_rank: int = 6,
    extra_park_keys: Iterable[tuple[str, str]] = (),
    park_min_rank: int = 0,
) -> list[ScoredLexiconEntry]:
    """Raise greedy frequency-free reserved in published ranks 4-max_rank. No targets.

    Unblend when the OLS map was mixed toward the 25th percentile. Restore the
    train-fit optimism offset when those greedy already sit next to top-3.
    Ranks are the parked published list when extra-FF parking is on. Lifting
    parked greedy_alts/nbest with the same map was 6k 5,770 / 5,565 (wash, −1
    top-3). Extra FF stay parked. Default max_rank is 6; 4-10 unblend stole
    1k 928→926 on the 11-slot path.
    """
    if not published or not lexicon_reference:
        return list(published)
    if blend <= 0.0 and optimism_offset <= 0.0:
        return list(published)
    dest = [candidate.spatial for candidate in lexicon_reference]
    ranked = published_ranking_scored(published, reserved, lexicon_reference)
    if park_min_rank >= 2 and extra_park_keys:
        ranked = park_extra_frequency_free_to_protect_top3(
            ranked, extra_park_keys, min_rank=park_min_rank,
        )
    high = max(4, max_rank)
    lift_keys = {
        (_normalize(candidate.word), candidate.entry.language)
        for rank, (candidate, _fusion) in enumerate(ranked, 1)
        if 4 <= rank <= high
        and candidate.frequency_free
        and candidate.source == "greedy"
        and (candidate.oov_map_blend > 0.0 or optimism_offset > 0.0)
    }
    if not lift_keys:
        return list(published)
    lifted = []
    for candidate in published:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key not in lift_keys:
            lifted.append(candidate)
            continue
        spatial = candidate.spatial
        if blend > 0.0 and candidate.oov_map_blend > 0.0:
            spatial = unblend_oov_spatial(spatial, dest, candidate.oov_map_blend or blend)
        if optimism_offset > 0.0:
            spatial += optimism_offset
        lifted.append(
            ScoredLexiconEntry(
                candidate.entry,
                spatial,
                frequency_free=True,
                source=candidate.source,
                oov_map_blend=0.0,
            )
        )
    return lifted


def unblend_greedy_alts_when_greedy_misses_top3(
    published: Sequence[ScoredLexiconEntry],
    reserved: Sequence[ScoredLexiconEntry],
    lexicon_reference: Sequence[ScoredLexiconEntry],
) -> list[ScoredLexiconEntry]:
    """Unblend the best published greedy_alt to OLS only when greedy misses top-3.

    Blend-0.5 extra greedy_alts convert 0. Unclamping every extra stole 6k top-3
    5,566→5,550. If greedy already occupies top-3, alts stay blended. If greedy
    misses, the spatially-best published greedy_alt unblends to the train-fit map
    so one converting alt can enter top-3. No floor clamp, no extra optimism
    offset, no evaluation targets.
    """
    if not published or not lexicon_reference:
        return list(published)
    ranked = published_ranking_scored(published, reserved, lexicon_reference)
    top = [candidate for candidate, _fusion in ranked[:3]]
    if any(candidate.frequency_free and candidate.source == "greedy" for candidate in top):
        return list(published)
    if any(
        candidate.frequency_free and candidate.source == "greedy_alts" for candidate in top
    ):
        return list(published)
    dest = [candidate.spatial for candidate in lexicon_reference]
    alts = [
        candidate for candidate in published
        if candidate.frequency_free
        and candidate.source == "greedy_alts"
        and candidate.oov_map_blend > 0.0
    ]
    if not alts:
        return list(published)
    best = max(alts, key=lambda candidate: (candidate.spatial, _normalize(candidate.word)))
    best_key = (_normalize(best.word), best.entry.language)
    result: list[ScoredLexiconEntry] = []
    for candidate in published:
        key = (_normalize(candidate.word), candidate.entry.language)
        if key != best_key:
            result.append(candidate)
            continue
        spatial = unblend_oov_spatial(candidate.spatial, dest, candidate.oov_map_blend)
        result.append(
            ScoredLexiconEntry(
                candidate.entry,
                spatial,
                frequency_free=True,
                source=candidate.source,
                oov_map_blend=0.0,
            )
        )
    return result


def collect_oov_calibration_observations(
    logits: Sequence[Sequence[float]],
    ctc_candidates: Sequence[ScoredLexiconEntry],
    greedy_emissions: Sequence[int],
    greedy_spelling: str | None,
    lexicon_words: set[str],
) -> tuple[list[tuple[float, float]], float | None]:
    """In-lexicon (forward, spatial) pairs and an optional OOV optimism gap. No targets."""
    pairs = [
        (ctc_forward_logprob(logits, candidate.entry.emissions), candidate.spatial)
        for candidate in ctc_candidates
        if candidate.entry.emissions
    ]
    gap = None
    if greedy_emissions and greedy_spelling and greedy_spelling not in lexicon_words and ctc_candidates:
        gap = ctc_forward_logprob(logits, greedy_emissions) - max(candidate.spatial for candidate in ctc_candidates)
    return pairs, gap


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


def _write_candidate_slate(stream, row, ctc, geometric, merged, reserved=None):
    def candidates(slate):
        return [{"word": item.entry.word, "language": item.entry.language,
                 "frequency": item.entry.frequency, "spatial": item.spatial,
                 **({"frequencyFree": True} if item.frequency_free else {})} for item in slate]
    record = {
        "schemaVersion": 1, "id": row.identifier, "sessionId": row.session_id,
        "language": row.language, "target": row.target, "strata": sorted(row.strata),
        "ctc": candidates(ctc), "geometric": candidates(geometric),
        "merged": candidates(merged),
    }
    if reserved:
        record["reserved"] = candidates(reserved)
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
    lexicon_by_language = {
        language: {_normalize(entry.word): entry for entry in entries}
        for language, entries in lexicon.items()
    }
    oov_nbest = int(getattr(args, "reserved_oov_nbest", 0) or 0)
    adaptive_merge = bool(getattr(args, "stratum_adaptive_merge", False))
    reserved_budget = int(getattr(args, "reserved_budget", 0) or 0)
    include_neighbors = bool(getattr(args, "include_lexicon_neighbors", False))
    include_truncated = bool(getattr(args, "include_truncated_leftovers", False))
    oov_conservative = bool(getattr(args, "oov_conservative_spatial", False))
    oov_map_blend = float(getattr(args, "oov_map_blend", 0.0) or 0.0)
    calibration = None
    known_offensive: set[str] = set()
    if oov_nbest:
        if not getattr(args, "oov_calibration", None) or not getattr(args, "known_offensive_lexicon", None):
            raise SwipeEvaluationError("reserved OOV n-best requires calibration and a known-offensive lexicon")
        calibration = load_oov_score_calibration(args.oov_calibration)
        known_offensive = load_known_offensive_words(args.known_offensive_lexicon)
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
    reserved_added = 0
    known_offensive_rejected = 0
    candidate_recall = {
        name: {"rows": 0, "targetPresent": 0}
        for name in metric_names
    }
    baseline_candidate_recall = {
        name: {"rows": 0, "targetPresent": 0}
        for name in metric_names
    }

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
        ctc_limit, geometric_limit = decoder_slate_budgets(row.strata) if adaptive_merge else (32, 32)
        merged = merge_swipe_slates(decoded[:ctc_limit], geometric[:geometric_limit])
        reserved: list[ScoredLexiconEntry] = []
        if oov_nbest and calibration is not None:
            reserved, rejected = reserved_oov_from_logits(
                logits,
                layout,
                calibration=calibration,
                n_best=oov_nbest,
                beam_width=int(getattr(args, "oov_beam_width", 0) or max(4, min(16, oov_nbest))),
                existing=merged,
                known_offensive=known_offensive,
                lexicon_by_word=lexicon_by_language.get(row.language, {}),
                language=row.language,
                ctc_spatials=[candidate.spatial for candidate in decoded],
                include_lexicon_neighbors=include_neighbors,
                ctc=decoded,
                geometric=geometric,
                include_truncated=include_truncated,
                oov_conservative=oov_conservative,
                oov_map_blend=oov_map_blend,
            )
            known_offensive_rejected += rejected
            reserved_added += len(reserved)
        if reserved_budget > 0:
            converting_best = bool(getattr(args, "converting_best_occupant", False))
            prefer_converting_greedy_alts = bool(
                getattr(args, "prefer_converting_greedy_alts", False)
            )
            ablation_extra_fill_sources = bool(
                getattr(args, "ablation_extra_fill", False)
            )
            leftover_inlex_fill = bool(getattr(args, "leftover_inlex_fill", False))
            converting_leftover_extras_fill = bool(
                getattr(args, "converting_leftover_extras", False)
            )
            converting_alt_expand = bool(getattr(args, "converting_alt_expand", False))
            ablation_first_source_fill_sources = bool(
                getattr(args, "ablation_first_source_fill", False)
            )
            converting_fill_loss_append = bool(
                getattr(args, "converting_fill_loss_append", False)
            )
            leftover_greedy_alts_append = bool(
                getattr(args, "leftover_greedy_alts_append", False)
            )
            protect_frozen_ranks = bool(getattr(args, "protect_frozen_ranks", False))
            length_changing_extra_oov = bool(getattr(args, "length_changing_extra_oov", False))
            leftover_converting_after_extra_oov_fill = bool(
                getattr(args, "leftover_converting_after_extra_oov", False)
            )
            tiny_nbest_truncated = bool(getattr(args, "tiny_nbest_truncated", False))
            published = publish_reserved_slots(
                merged,
                reserved,
                reserved_budget=reserved_budget,
                converting_best=converting_best,
                prefer_converting_greedy_alts=prefer_converting_greedy_alts,
                ablation_extra_fill_sources=ablation_extra_fill_sources,
                leftover_inlex_fill=leftover_inlex_fill,
                converting_leftover_extras_fill=converting_leftover_extras_fill,
                converting_alt_expand=converting_alt_expand,
                ablation_first_source_fill_sources=ablation_first_source_fill_sources,
                converting_fill_loss_append=converting_fill_loss_append,
                leftover_greedy_alts_append=leftover_greedy_alts_append,
                protect_frozen_ranks=protect_frozen_ranks,
                length_changing_extra_oov=length_changing_extra_oov,
                leftover_converting_after_extra_oov_fill=leftover_converting_after_extra_oov_fill,
                tiny_nbest_truncated=tiny_nbest_truncated,
            )
            skip_keys = {(_normalize(candidate.word), candidate.entry.language) for candidate in merged}
            extra_park_keys = extra_reserved_occupant_keys(
                reserved,
                merged,
                reserved_budget=reserved_budget,
                skip_keys=skip_keys,
                converting_best=converting_best,
                prefer_converting_greedy_alts=prefer_converting_greedy_alts,
                ablation_extra_fill_sources=ablation_extra_fill_sources,
                leftover_inlex_fill=leftover_inlex_fill,
                converting_leftover_extras_fill=converting_leftover_extras_fill,
                converting_alt_expand=converting_alt_expand,
                ablation_first_source_fill_sources=ablation_first_source_fill_sources,
                converting_fill_loss_append=converting_fill_loss_append,
                leftover_greedy_alts_append=leftover_greedy_alts_append,
                protect_frozen_ranks=protect_frozen_ranks,
                length_changing_extra_oov=length_changing_extra_oov,
                leftover_converting_after_extra_oov_fill=leftover_converting_after_extra_oov_fill,
                tiny_nbest_truncated=tiny_nbest_truncated,
            )
            park_min_rank = int(getattr(args, "park_extra_reserved_min_rank", 0) or 0)
            published = lift_near_top3_frequency_free(
                published,
                reserved,
                merged,
                blend=oov_map_blend,
                optimism_offset=calibration.optimism_offset if calibration is not None else 0.0,
                max_rank=int(getattr(args, "oov_lift_max_rank", 6) or 6),
                extra_park_keys=extra_park_keys,
                park_min_rank=park_min_rank,
            )
            published = unblend_greedy_alts_when_greedy_misses_top3(
                published, reserved, merged,
            )
            ctc_geometric_fusion = published_ranking(
                published,
                reserved,
                lexicon_reference=merged,
                extra_park_keys=extra_park_keys,
                park_min_rank=park_min_rank,
                protect_frozen_ranks=protect_frozen_ranks,
            )
            competing = published
        else:
            competing = competing_slate(merged, reserved)
            ctc_geometric_fusion = rank_static_fusion_with_reserved(merged, reserved)
        finished = time.perf_counter_ns()
        if slate_stream is not None:
            _write_candidate_slate(slate_stream, row, decoded, geometric, merged, reserved or None)
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
            candidate_recall[name]["rows"] += 1
            if reserved_budget > 0:
                candidate_recall[name]["targetPresent"] += row.target in ctc_geometric_predictions
            else:
                candidate_recall[name]["targetPresent"] += row.target in {item.word for item in competing}
            baseline_candidate_recall[name]["rows"] += 1
            baseline_candidate_recall[name]["targetPresent"] += row.target in {item.word for item in merged}
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
    def report_recall(values: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float]]:
        return {
            name: {
                **counts,
                "recall": counts["targetPresent"] / counts["rows"] if counts["rows"] else 0.0,
            }
            for name, counts in values.items()
        }
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
            "algorithm": (
                "lexicon-constrained-ctc-prefix-beam-v1"
                if not oov_nbest else
                "diagnostic-lexicon-ctc-with-calibrated-reserved-oov-v1"
            ),
            "beamWidth": args.beam_width,
            "greedyExactAccuracy": greedy_exact / len(rows),
            "greedyBlankFallbacks": greedy_blank_fallbacks,
            "geometricAlgorithm": "production-template-cost-v1",
            "geometricTemplates": len(geometric_index.templates),
            "decoderSlatesNormalizedBeforeUnion": True,
            "stratumAdaptiveMerge": adaptive_merge,
            "reservedOovNBest": oov_nbest,
            "reservedBudget": reserved_budget,
            "published31Membership": reserved_budget > 0,
            "includeLexiconNeighbors": include_neighbors,
            "includeTruncatedLeftovers": include_truncated,
            "oovConservativeSpatial": oov_conservative,
            "oovMapBlend": oov_map_blend,
            "reservedOovCandidatesAdded": reserved_added,
            "knownOffensiveRejected": known_offensive_rejected,
        },
        "metrics": metric_report,
        "staticFusionMetrics": static_metric_report,
        "geometricMetrics": geometric_metric_report,
        "ctcGeometricFusionMetrics": ctc_geometric_metric_report,
        "candidateRecall": report_recall(candidate_recall),
        "baselineCandidateRecall": report_recall(baseline_candidate_recall),
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
            *(
                [
                    "Reserved-slot OOV is a diagnostic candidate-membership bound, not a quality or top-3 win.",
                    "Production vocabulary, scoring, and safety policy are unchanged.",
                ]
                if oov_nbest or adaptive_merge else []
            ),
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
    parser.add_argument(
        "--reserved-oov-nbest",
        type=int,
        default=0,
        help="Diagnostic: append this many unconstrained CTC spellings into reserved slots",
    )
    parser.add_argument("--oov-calibration", type=pathlib.Path, help="Fitted OOV score calibration JSON")
    parser.add_argument(
        "--known-offensive-lexicon",
        type=pathlib.Path,
        help="Full native dictionary export used only for the known-offensive filter",
    )
    parser.add_argument(
        "--stratum-adaptive-merge",
        action="store_true",
        help="Diagnostic: allocate CTC vs geometry union slots by stratum",
    )
    parser.add_argument(
        "--reserved-budget",
        type=int,
        default=0,
        help="Diagnostic: published reserved slots inside the 32-slot bound (0 = unbounded competing slate)",
    )
    parser.add_argument(
        "--include-lexicon-neighbors",
        action="store_true",
        help="Diagnostic: add in-lexicon edit-1 neighbors of unconstrained CTC spellings",
    )
    parser.add_argument(
        "--include-truncated-leftovers",
        action="store_true",
        help="Diagnostic: add CTC/geometry candidates dropped by the 32-slot union",
    )
    parser.add_argument(
        "--oov-conservative-spatial",
        action="store_true",
        help="Diagnostic: score true OOV at the 25th-percentile lexicon spatial instead of the OLS map",
    )
    parser.add_argument(
        "--oov-beam-width",
        type=int,
        default=0,
        help="Diagnostic: unconstrained n-best beam (0 = min 4, at most 16, matching n-best)",
    )
    parser.add_argument(
        "--oov-map-blend",
        type=float,
        default=0.0,
        help="Diagnostic: blend train-fit OOV spatial toward the 25th-percentile (0 = OLS map only)",
    )
    parser.add_argument(
        "--oov-lift-max-rank",
        type=int,
        default=6,
        help="Diagnostic: restore train optimism on greedy FF reserved through this published rank (default 6)",
    )
    parser.add_argument(
        "--park-extra-reserved-min-rank",
        type=int,
        default=0,
        help="Diagnostic: demote extra reserved occupants (after greedy-first) to this rank or worse (0 = off)",
    )
    parser.add_argument(
        "--converting-best-occupant",
        action="store_true",
        help="Diagnostic: first reserved seat is the highest-OLS converting OOV, not greedy-first",
    )
    parser.add_argument(
        "--prefer-converting-greedy-alts",
        action="store_true",
        help="Diagnostic: extra reserved seats prefer OLS-converting greedy_alts over non-converting n-best/alts",
    )
    parser.add_argument(
        "--ablation-extra-fill",
        action="store_true",
        help="Diagnostic: extra reserved seats round-robin greedy_alts, neighbors, n-best, truncated",
    )
    parser.add_argument(
        "--leftover-inlex-fill",
        action="store_true",
        help="Diagnostic: leftover extra_oov seats after converting OOV go to neighbors/truncated",
    )
    parser.add_argument(
        "--converting-leftover-extras",
        action="store_true",
        help="Diagnostic: leftover extra seats only OLS-converting greedy_alts/nbest",
    )
    parser.add_argument(
        "--converting-alt-expand",
        action="store_true",
        help="Diagnostic: expand extra converting greedy_alt seats when they overflow extra_oov",
    )
    parser.add_argument(
        "--ablation-first-source-fill",
        action="store_true",
        help="Diagnostic: extra seats follow ablation first-source (length-changing greedy_alts, neighbors, n-best last)",
    )
    parser.add_argument(
        "--converting-fill-loss-append",
        action="store_true",
        help="Diagnostic: append leftover converting greedy_alts after spatial occupants",
    )
    parser.add_argument(
        "--protect-frozen-ranks",
        action="store_true",
        help="Diagnostic: 32-slot fill keeps lexicon fusion ranks 1-23; leftover converting alts occupy 24-32",
    )
    parser.add_argument(
        "--length-changing-extra-oov",
        action="store_true",
        help="Diagnostic: extra_oov leftover seats prefer length-changing converting greedy_alts; hold spatial n-best",
    )
    parser.add_argument(
        "--leftover-converting-after-extra-oov",
        action="store_true",
        help="Diagnostic: leftover seats after spatial extra_oov prefer leftover converting greedy_alts; hold extra_oov/n-best",
    )
    parser.add_argument(
        "--tiny-nbest-truncated",
        action="store_true",
        help="Diagnostic: extra_oov holds n-best rank<=4 and greedy_alts; leftover seats truncated then n-best 5+",
    )
    parser.add_argument(
        "--leftover-greedy-alts-append",
        action="store_true",
        help="Diagnostic: append leftover greedy_alts after spatial occupants without extra parking",
    )
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
