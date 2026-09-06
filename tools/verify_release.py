#!/usr/bin/env python3
"""Fail-closed source/APK checks shared by F-Droid and GrapheneOS release gates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import build_context_tokenizer
import build_onnxruntime_android
import context_model_contract
import evaluate_engine
import model_sources
import prepare_context_dataset
import prepare_swipe_dataset
import prepare_tap_evaluation
import score_context_teacher
import swipe_model_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
EXPECTED_APPLICATION_IDS = {
    "org.libreboard.keyboard",
    "org.libreboard.keyboard.debug",
}
EXPECTED_MIN_SDK = "26"
EXPECTED_TARGET_SDK = "36"
EXPECTED_APP_VERSION_CODE = 1
EXPECTED_NATIVE_ABIS = {"armeabi-v7a", "arm64-v8a", "x86", "x86_64"}
LATIN_IME_CLASS = "helium314.keyboard.latin.LatinIME"
CLIPBOARD_PROVIDER_CLASS = "helium314.keyboard.latin.database.ClipboardContentProvider"
FORBIDDEN_PERMISSIONS = {
    "android.permission.INTERNET",
    "android.permission.ACCESS_NETWORK_STATE",
}
FORBIDDEN_DEPENDENCY_MARKERS = (
    "com.google.android.gms",
    "com.google.firebase",
    "com.google.mlkit",
    "io.sentry",
    "com.amplitude",
    "com.appsflyer",
)
SENSITIVE_LOG_IDENTIFIERS = (
    "appLabel",
    "applicationSpecifiedCompletions",
    "candidate",
    "chosenWord",
    "clipboard",
    "commitWord",
    "consideredWord",
    "key",
    "keyEvent",
    "mAutoCorrectionWord",
    "mComposingText",
    "mEnteredText",
    "mSuggestions",
    "mTypedWord",
    "mWord",
    "ngramContext",
    "phrase",
    "pointer",
    "prevWordsContext",
    "pseudoTypedWordInfo",
    "rawWord",
    "rawWordBeforeCommit",
    "splitText",
    "suggestion",
    "targetWord",
    "tempWord",
    "text",
    "typedWord",
    "word",
    "wordProperty",
)
SENSITIVE_LOG_IDENTIFIER_PATTERN = "|".join(
    sorted((re.escape(value) for value in SENSITIVE_LOG_IDENTIFIERS), key=len, reverse=True)
)
SENSITIVE_LOG_VALUE = re.compile(
    rf"\b(?:{SENSITIVE_LOG_IDENTIFIER_PATTERN})\b|"
    r"\bgetTextBeforeCursor\s*\(|\bprintableCode\s*\(",
)
SAFE_LOG_METADATA_ACCESS = re.compile(
    rf"\b(?:{SENSITIVE_LOG_IDENTIFIER_PATTERN})\b\s*(?:\?\.)?\.\s*"
    r"(?:length|size|count)\b(?:\s*\(\s*\))?",
)
SAFE_LOG_NULL_CHECK = re.compile(
    rf"(?:\b(?:{SENSITIVE_LOG_IDENTIFIER_PATTERN})\b\s*(?:==|!=)\s*null|"
    rf"null\s*(?:==|!=)\s*\b(?:{SENSITIVE_LOG_IDENTIFIER_PATTERN})\b)",
)
LOG_CALL_START = re.compile(r"\b(?:android\.util\.)?Log\.[A-Za-z_]\w*\s*\(")
KOTLIN_LOG_TEMPLATE = re.compile(r"\$\{([^{}]*)\}|\$([A-Za-z_]\w*)")
QUOTED_SOURCE_STRING = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
NATIVE_LOGGING_LOCK_MARKERS = (
    "ifneq ($(strip $(FLAG_DBG)),false)",
    "$(error FLAG_DBG is disabled by LibreBoard's no-typed-text-logs policy)",
    "ifneq ($(strip $(FLAG_DO_PROFILE)),false)",
    "$(error FLAG_DO_PROFILE is disabled by LibreBoard's no-typed-text-logs policy)",
)
ALLOWED_NATIVE_LIBRARIES = {
    "libjni_latinime.so",
    # Apache-2.0 AndroidX dependency, pinned by the Compose BOM and verified for 16 KiB pages.
    "libandroidx.graphics.path.so",
    # Produced only by tools/build_onnxruntime_android.py from the pinned source submodule.
    "libonnxruntime.so",
    "libonnxruntime4j_jni.so",
}
BACKUP_DOMAINS = {"root", "file", "database", "sharedpref", "external"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
SECURITY_PATCH = re.compile(r"^\d{4}-\d{2}-\d{2}$")
UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
PHASE0_CHECKS = {
    "tap_relative_error_reduction",
    "neural_valid_word_relative_error_reduction",
    "neural_valid_word_absolute_gain",
    "false_correction_ceiling",
    "swipe_top1",
    "swipe_top3",
    "swipe_short_top3",
    "swipe_return_trip_top3",
    "swipe_geometric_relative_error_reduction",
    "tap_p95_latency",
    "swipe_p95_latency",
    "peak_neural_memory",
}
MINIMUM_PHASE0_COUNTS = {
    "tap_error": 3_000,
    "valid_word": 1_000,
    "spacing": 500,
    "lexical": 500,
    "swipe": 5_000,
}
MINIMUM_PHASE0_SWIPE_STRATA_COUNTS = dict(evaluate_engine.MINIMUM_SWIPE_STRATA_COUNTS)
GRAPHENEOS_CHECKS = {
    "apk_verified",
    "keyboard_enable_select",
    "direct_boot",
    "plain_text_correction",
    "url_email_policy",
    "password_pin_policy",
    "incognito_policy",
    "no_suggestions_policy",
    "terminal_commit",
    "english_german_lock",
    "model_fallbacks",
    "clipboard_backup_wipe",
    "window_and_animation",
    "performance_budgets",
    "crash_anr_stale_result_free",
}
PHASE0_ENVIRONMENTS = {
    "stock_android_hardware",
    "grapheneos_hardware",
    "low_ram_emulator",
}
ONNXRUNTIME_COMMIT = "8c546c37b43caaca1fa25db430dab94b901cf277"
MODEL_PACK_APPLICATION_ID = "org.libreboard.model.en_de"
MODEL_PACK_AUTHORITY = "org.libreboard.model.en_de"
MODEL_PACK_PROVIDER_CLASS = "org.libreboard.model.en_de.ContextModelProvider"
MODEL_PACK_ASSET = "assets/model.lbmodel"
MODEL_PACK_MAXIMUM_BYTES = 28 * 1024 * 1024
CONTEXT_MODEL_MAXIMUM_BYTES = 24 * 1024 * 1024
SWIPE_MODEL_MAXIMUM_BYTES = 2_621_440
SWIPE_ARCHIVE_MAXIMUM_BYTES = 3 * 1024 * 1024
CORE_SWIPE_MODEL_ASSET = "assets/models/swipe-latin-v1.lbmodel"
CORE_MODEL_PUBLIC_KEY_ASSET = "assets/models/libreboard-model-signing-public.der"
MODEL_TOKENIZER_MAXIMUM_BYTES = 2 * 1024 * 1024
MODEL_MANIFEST_MAXIMUM_BYTES = 256 * 1024
MODEL_SIGNATURE_MAXIMUM_BYTES = 16 * 1024
MODEL_ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MODEL_ARCHIVE_EXTERNAL_ATTRIBUTES = 0o100644 << 16
CONTEXT_MODEL_ARCHIVE_ENTRIES = {
    "manifest.json",
    "model.onnx",
    "tokenizer.json",
    "signature.der",
}
SWIPE_MODEL_ARCHIVE_ENTRIES = {
    "manifest.json",
    "model.onnx",
    "signature.der",
}
MODEL_MANIFEST_FIELDS = {
    "schemaVersion",
    "engineAbi",
    "modelKind",
    "tensorAbi",
    "locales",
    "architecture",
    "parameterCount",
    "quantization",
    "modelSha256",
    "tokenizerSha256",
    "requiredOnnxOperators",
    "license",
    "provenance",
    "minimumAppVersionCode",
}
STORE_METADATA_LOCALES = {"en-US", "de-DE"}
STORE_METADATA_FILES = {"title.txt", "short_description.txt", "full_description.txt"}
FORBIDDEN_STORE_METADATA_MARKERS = (
    "swypelibs",
    "erkserkserks/openboard",
    "only with closed source",
    "only with closed-source",
    "nur mit proprietärer",
)
GRADLE_VERIFICATION_NS = "{https://schema.gradle.org/dependency-verification}"
REQUIRED_GRADLE_COMPONENTS = {
    ("com.android.tools.build", "gradle", "8.13.2"),
    ("com.android.tools", "desugar_jdk_libs", "2.1.5"),
    ("org.jetbrains.kotlin", "kotlin-gradle-plugin", "2.3.20"),
    ("org.jetbrains.kotlin", "kotlin-stdlib", "2.3.20"),
    ("org.jetbrains.kotlin", "kotlin-test", "2.3.20"),
    ("org.jetbrains.kotlinx", "kotlinx-serialization-json", "1.11.0"),
    ("androidx.compose", "compose-bom", "2025.11.01"),
    ("androidx.compose.material3", "material3", "1.4.0"),
    ("androidx.compose.ui", "ui-tooling", "1.9.5"),
    ("androidx.compose.ui", "ui-tooling-preview", "1.9.5"),
    ("androidx.autofill", "autofill", "1.3.0"),
    ("androidx.core", "core-ktx", "1.17.0"),
    ("androidx.navigation", "navigation-compose", "2.9.8"),
    ("androidx.recyclerview", "recyclerview", "1.4.0"),
    ("androidx.test", "core", "1.7.0"),
    ("androidx.test", "runner", "1.7.0"),
    ("androidx.viewpager2", "viewpager2", "1.1.0"),
    ("com.github.skydoves", "colorpicker-compose", "1.1.3"),
    ("sh.calvin.reorderable", "reorderable", "3.1.0"),
    ("junit", "junit", "4.13.2"),
    ("org.mockito", "mockito-core", "5.23.0"),
    ("org.robolectric", "robolectric", "4.16.1"),
}


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
CONTEXT_DISTILLATION_MANIFEST_FIELDS = {
    "schemaVersion",
    "modelId",
    "releaseEligible",
    "appCommit",
    "dataManifestSha256",
    "distillationPolicySha256",
    "tokenizerSha256",
    "teacher",
    "toolSha256",
    "tokenizerContractToolSha256",
    "toolchain",
    "generationCounts",
    "teacherMetrics",
    "outputs",
}
CONTEXT_DISTILLATION_OUTPUT_FIELDS = {"bytes", "sha256", "records", "languages"}
CONTEXT_DISTILLATION_METRIC_FIELDS = {
    "records", "candidateRows", "observedTop1", "observedTop1Rate", "slateSizes",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _log_calls(source: str):
    """Yield complete android.util.Log calls without requiring Java/Kotlin parsing."""
    for match in LOG_CALL_START.finditer(source):
        opening = source.find("(", match.start())
        depth = 0
        quote = None
        escaped = False
        index = opening
        while index < len(source):
            character = source[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in {'"', "'"}:
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    yield match.start(), source[match.start():index + 1]
                    break
            index += 1


def validate_no_content_bearing_logs(
        errors: list[str], source_root: pathlib.Path) -> None:
    """Reject log calls that expose text, candidates, key events, or gesture state."""
    for path in sorted(source_root.rglob("*")):
        if path.suffix not in {".java", ".kt"}:
            continue
        source = path.read_text(encoding="utf-8")
        for offset, call in _log_calls(source):
            template_expressions = "\n".join(
                group
                for match in KOTLIN_LOG_TEMPLATE.finditer(call)
                for group in (match.group(1) or match.group(2),)
            )
            outside_strings = QUOTED_SOURCE_STRING.sub('""', call)
            values = template_expressions + "\n" + outside_strings
            values = SAFE_LOG_METADATA_ACCESS.sub("", values)
            values = SAFE_LOG_NULL_CHECK.sub("", values)
            if SENSITIVE_LOG_VALUE.search(values):
                line = source.count("\n", 0, offset) + 1
                try:
                    display_path = path.relative_to(ROOT)
                except ValueError:
                    display_path = path
                fail(errors, f"content-bearing log call is forbidden: {display_path}:{line}")


def validate_native_logging_lock(errors: list[str], makefile: pathlib.Path) -> None:
    try:
        source = makefile.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        fail(errors, f"cannot read native build privacy policy: {exc}")
        return
    if any(marker not in source for marker in NATIVE_LOGGING_LOCK_MARKERS):
        fail(errors, "native debug/profile logging must remain fail-closed")


def android_attribute(node: ET.Element, name: str) -> str | None:
    return node.attrib.get(ANDROID_NS + name)


def validate_hash_locked_requirements(
    errors: list[str],
    path: pathlib.Path,
    label: str,
    required_packages: tuple[str, ...],
) -> None:
    try:
        lock_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(errors, f"cannot read hash-locked {label} dependencies: {exc}")
        return
    if any(package not in lock_text for package in required_packages):
        fail(errors, f"{label} lock does not contain the audited direct dependencies")
    requirement_blocks = re.split(r"\n(?=[a-z0-9][a-z0-9_.-]*==)", lock_text)
    unhashed = [
        block.split("==", 1)[0]
        for block in requirement_blocks
        if "==" in block and "--hash=sha256:" not in block
    ]
    if unhashed:
        fail(errors, f"{label} lock contains unhashed packages: " + ", ".join(unhashed))
    if "http://" in lock_text:
        fail(errors, f"{label} lock contains an insecure package source")


def validate_store_metadata(
    errors: list[str],
    root: pathlib.Path = ROOT / "fastlane/metadata/android",
) -> None:
    """Keep Phase 1 store copy honest and free of inherited proprietary-swipe instructions."""
    try:
        locales = {path.name for path in root.iterdir() if path.is_dir()}
    except OSError as exc:
        fail(errors, f"cannot read store metadata: {exc}")
        return
    if locales != STORE_METADATA_LOCALES:
        fail(errors, "store metadata must contain exactly the reviewed English and German locales")

    searchable_text: list[str] = []
    for locale in sorted(STORE_METADATA_LOCALES):
        locale_root = root / locale
        locale_values: dict[str, str] = {}
        for filename in sorted(STORE_METADATA_FILES):
            path = locale_root / filename
            try:
                value = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                fail(errors, f"cannot read {locale} store metadata {filename}: {exc}")
                continue
            locale_values[filename] = value
            searchable_text.append(value.lower())
            if not value or any(ord(character) < 32 and character not in "\n\t" for character in value):
                fail(errors, f"{locale} store metadata {filename} is empty or contains control characters")
            if "LibreBoard" not in value and filename != "short_description.txt":
                fail(errors, f"{locale} store metadata {filename} does not identify LibreBoard")
            maximum = {
                "title.txt": 30,
                "short_description.txt": 80,
                "full_description.txt": 4_000,
            }[filename]
            if len(value) > maximum:
                fail(errors, f"{locale} store metadata {filename} exceeds its length limit")
        if locale_values.get("title.txt") not in {None, "LibreBoard"}:
            fail(errors, f"{locale} store title must be LibreBoard")

    combined = "\n".join(searchable_text)
    for marker in FORBIDDEN_STORE_METADATA_MARKERS:
        if marker in combined:
            fail(errors, f"store metadata contains obsolete proprietary-swipe guidance: {marker}")
    for required in ("internet", "access_network_state", "geometric"):
        if required not in combined:
            fail(errors, f"store metadata omits the release contract marker: {required}")


def validate_gradle_dependency_verification(
    errors: list[str],
    path: pathlib.Path = ROOT / "gradle/verification-metadata.xml",
) -> None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        fail(errors, f"cannot read Gradle dependency verification metadata: {exc}")
        return
    if root.tag != GRADLE_VERIFICATION_NS + "verification-metadata":
        fail(errors, "Gradle dependency verification metadata has an unexpected root")
        return
    configuration = root.find(GRADLE_VERIFICATION_NS + "configuration")
    components = root.find(GRADLE_VERIFICATION_NS + "components")
    if configuration is None or components is None:
        fail(errors, "Gradle dependency verification metadata is incomplete")
        return
    expected_configuration = {
        GRADLE_VERIFICATION_NS + "verify-metadata",
        GRADLE_VERIFICATION_NS + "verify-signatures",
    }
    configuration_children = list(configuration)
    if any(node.tag not in expected_configuration for node in configuration_children):
        fail(errors, "Gradle dependency verification must not contain trust bypasses")
    if (
        len(configuration_children) != len(expected_configuration)
        or {node.tag for node in configuration_children} != expected_configuration
        or configuration.findtext(GRADLE_VERIFICATION_NS + "verify-metadata") != "true"
        or configuration.findtext(GRADLE_VERIFICATION_NS + "verify-signatures") != "false"
    ):
        fail(errors, "Gradle dependency checksum verification must remain enabled")

    coordinates: set[tuple[str, str, str]] = set()
    artifact_count = 0
    for component in components:
        if component.tag != GRADLE_VERIFICATION_NS + "component":
            fail(errors, "Gradle verification components contain an unexpected element")
            continue
        coordinate = tuple(component.attrib.get(name, "") for name in ("group", "name", "version"))
        if (
            set(component.attrib) != {"group", "name", "version"}
            or any(not value for value in coordinate)
            or coordinate in coordinates
        ):
            fail(errors, "Gradle verification metadata contains an invalid or duplicate component")
        coordinates.add(coordinate)
        for artifact in component:
            artifact_count += 1
            if (
                artifact.tag != GRADLE_VERIFICATION_NS + "artifact"
                or set(artifact.attrib) != {"name"}
                or not artifact.attrib["name"]
                or "/" in artifact.attrib["name"]
                or "\\" in artifact.attrib["name"]
                or artifact.attrib["name"] in {".", ".."}
            ):
                fail(errors, "Gradle verification metadata contains an invalid artifact")
                continue
            hashes = list(artifact)
            if (
                len(hashes) != 1
                or hashes[0].tag != GRADLE_VERIFICATION_NS + "sha256"
                or not SHA256.fullmatch(hashes[0].attrib.get("value", ""))
                or set(hashes[0].attrib) - {"value", "origin"}
            ):
                fail(errors, "Gradle verification artifact does not have exactly one SHA-256 checksum")
    if artifact_count < 100:
        fail(errors, "Gradle dependency verification metadata is implausibly incomplete")
    missing = REQUIRED_GRADLE_COMPONENTS - coordinates
    if missing:
        rendered = ", ".join(":".join(coordinate) for coordinate in sorted(missing))
        fail(errors, f"Gradle verification metadata omits required components: {rendered}")


def validate_context_distillation_manifest(
    errors: list[str],
    manifest: dict,
    *,
    context_corpus: dict,
    policy: score_context_teacher.DistillationPolicy,
    teacher_source: model_sources.Source,
) -> None:
    """Validate the immutable release-sized teacher-scoring result without the large JSONL files."""
    if set(manifest) != CONTEXT_DISTILLATION_MANIFEST_FIELDS:
        fail(errors, "context distillation manifest has an unexpected schema")
        return
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("modelId") != "context-en-de-v1"
        or manifest.get("releaseEligible") is not True
        or not isinstance(manifest.get("appCommit"), str)
        or not GIT_COMMIT.fullmatch(manifest["appCommit"])
    ):
        fail(errors, "context distillation manifest is not a release-eligible v1 result")

    expected_teacher = {
        "sourceId": teacher_source.identifier,
        "revision": teacher_source.revision,
        "license": teacher_source.license,
        "sourceUrl": teacher_source.source_url,
        "modelSha256": teacher_source.artifact("model.safetensors").sha256,
    }
    if manifest.get("teacher") != expected_teacher:
        fail(errors, "context distillation manifest has unexpected teacher provenance")
    if manifest.get("dataManifestSha256") != model_sources.file_sha256(
        prepare_context_dataset.DEFAULT_CORPUS_MANIFEST
    ):
        fail(errors, "context distillation manifest is not bound to the current corpus")
    if manifest.get("distillationPolicySha256") != policy.sha256:
        fail(errors, "context distillation manifest is not bound to the current policy")
    if manifest.get("tokenizerSha256") != policy.tokenizer_sha256:
        fail(errors, "context distillation manifest is not bound to the approved tokenizer")
    if manifest.get("toolSha256") != model_sources.file_sha256(
        ROOT / "tools/score_context_teacher.py"
    ):
        fail(errors, "context distillation manifest is not bound to the current scoring tool")
    if manifest.get("tokenizerContractToolSha256") != model_sources.file_sha256(
        ROOT / "tools/context_tokenizer_contract.py"
    ):
        fail(errors, "context distillation manifest is not bound to the current tokenizer verifier")
    if manifest.get("toolchain") != {"torch": "2.8.0", "transformers": "4.57.6"}:
        fail(errors, "context distillation manifest has an unexpected toolchain")

    outputs = manifest.get("outputs")
    expected_output_names = {
        "train.scored.jsonl", "validation.scored.jsonl", "test.scored.jsonl",
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_output_names:
        fail(errors, "context distillation manifest has incomplete outputs")
        return

    prepared_outputs = context_corpus.get("outputs")
    generation_counts: dict[str, int] = {}
    output_records: dict[str, int] = {}
    output_languages: dict[str, dict[str, int]] = {}
    for split in prepare_context_dataset.SPLITS:
        filename = f"{split}.scored.jsonl"
        details = outputs[filename]
        prepared = (
            prepared_outputs.get(f"{split}.sentences.jsonl")
            if isinstance(prepared_outputs, dict) else None
        )
        if (
            not isinstance(details, dict)
            or set(details) != CONTEXT_DISTILLATION_OUTPUT_FIELDS
            or isinstance(details.get("bytes"), bool)
            or not isinstance(details.get("bytes"), int)
            or details["bytes"] <= 0
            or not isinstance(details.get("sha256"), str)
            or not SHA256.fullmatch(details["sha256"])
            or isinstance(details.get("records"), bool)
            or not isinstance(details.get("records"), int)
            or details["records"] <= 0
            or not isinstance(details.get("languages"), dict)
            or set(details["languages"]) != {"en-US", "de"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in details["languages"].values()
            )
            or sum(details["languages"].values()) != details["records"]
        ):
            fail(errors, f"context distillation output metadata is invalid: {filename}")
            continue
        if not isinstance(prepared, dict) or (
            details["records"] != prepared.get("records")
            or details["languages"] != prepared.get("languages")
        ):
            fail(errors, f"context distillation output does not match the prepared corpus: {filename}")
        records = details["records"]
        output_records[split] = records
        output_languages[split] = details["languages"]
        generation_counts[f"source:{split}"] = records
        generation_counts[f"scored:{split}"] = records
        for language, count in details["languages"].items():
            generation_counts[f"language:{split}:{language}"] = count

    if manifest.get("generationCounts") != generation_counts:
        fail(errors, "context distillation generation counts do not exactly reconcile")
    if (
        sum(output_records.values()) < policy.minimum_scored_records
        or sum(output_languages.get(split, {}).get("de", 0)
               for split in prepare_context_dataset.SPLITS) < policy.minimum_german_records
        or output_records.get("test", 0) < policy.minimum_held_out_records
        or output_languages.get("test", {}).get("en-US", 0)
        < policy.minimum_held_out_english_records
        or output_languages.get("test", {}).get("de", 0)
        < policy.minimum_held_out_german_records
    ):
        fail(errors, "context distillation manifest does not satisfy the release data floors")

    metrics = manifest.get("teacherMetrics")
    if not isinstance(metrics, dict) or set(metrics) != set(prepare_context_dataset.SPLITS):
        fail(errors, "context distillation manifest has incomplete teacher metrics")
        return
    for split in prepare_context_dataset.SPLITS:
        metric = metrics[split]
        records = output_records.get(split)
        if not isinstance(metric, dict) or set(metric) != CONTEXT_DISTILLATION_METRIC_FIELDS:
            fail(errors, f"context distillation teacher metrics are invalid: {split}")
            continue
        candidate_rows = metric.get("candidateRows")
        observed_top1 = metric.get("observedTop1")
        rate = metric.get("observedTop1Rate")
        slate_sizes = metric.get("slateSizes")
        valid_slates = isinstance(slate_sizes, dict) and bool(slate_sizes)
        slate_records = slate_rows = 0
        if valid_slates:
            for size, count in slate_sizes.items():
                if (
                    not isinstance(size, str)
                    or not size.isdecimal()
                    or not policy.minimum_candidates <= int(size) <= policy.maximum_candidates
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    valid_slates = False
                    break
                slate_records += count
                slate_rows += int(size) * count
        valid_metric = (
            records is not None
            and metric.get("records") == records
            and isinstance(candidate_rows, int)
            and not isinstance(candidate_rows, bool)
            and isinstance(observed_top1, int)
            and not isinstance(observed_top1, bool)
            and 0 <= observed_top1 <= records
            and finite_number(rate)
            and math.isclose(float(rate), observed_top1 / records, rel_tol=0.0, abs_tol=1e-15)
            and valid_slates
            and slate_records == records
            and slate_rows == candidate_rows
        )
        if not valid_metric:
            fail(errors, f"context distillation teacher metrics do not reconcile: {split}")


def validate_backup_exclusions(errors: list[str]) -> None:
    backup_path = ROOT / "app/src/main/res/xml/backup_rules.xml"
    extraction_path = ROOT / "app/src/main/res/xml/data_extraction_rules.xml"

    try:
        backup = ET.parse(backup_path).getroot()
    except (OSError, ET.ParseError) as exc:
        fail(errors, f"cannot read legacy backup exclusions: {exc}")
    else:
        exclusions = {
            node.attrib.get("domain")
            for node in backup.findall("exclude")
            if node.attrib.get("path") == "."
        }
        if not BACKUP_DOMAINS.issubset(exclusions):
            fail(errors, "legacy backup rules must exclude every app-data domain")

    try:
        extraction = ET.parse(extraction_path).getroot()
    except (OSError, ET.ParseError) as exc:
        fail(errors, f"cannot read Android 12+ data-extraction exclusions: {exc}")
    else:
        for section_name in ("cloud-backup", "device-transfer"):
            section = extraction.find(section_name)
            exclusions = set() if section is None else {
                node.attrib.get("domain")
                for node in section.findall("exclude")
                if node.attrib.get("path") == "."
            }
            if not BACKUP_DOMAINS.issubset(exclusions):
                fail(errors, f"{section_name} rules must exclude every app-data domain")


def validate_model_pack_source(errors: list[str]) -> None:
    manifest_path = ROOT / "modelpack-en-de/src/main/AndroidManifest.xml"
    build_path = ROOT / "modelpack-en-de/build.gradle.kts"
    provider_path = (
        ROOT / "modelpack-en-de/src/main/java/org/libreboard/model/en_de/ContextModelProvider.java"
    )
    settings_path = ROOT / "settings.gradle"
    try:
        manifest = ET.parse(manifest_path).getroot()
    except (OSError, ET.ParseError) as exc:
        fail(errors, f"cannot read model-pack source manifest: {exc}")
        return

    if any(node.tag.startswith("uses-permission") for node in manifest):
        fail(errors, "model-pack source manifest must request no permissions")
    application = manifest.find("application")
    if application is None:
        fail(errors, "model-pack source manifest has no application element")
    else:
        required_attributes = {
            "allowBackup": "false",
            "usesCleartextTraffic": "false",
            "directBootAware": "false",
            "hasCode": "true",
            "fullBackupContent": "@xml/backup_rules",
            "dataExtractionRules": "@xml/data_extraction_rules",
        }
        for name, expected in required_attributes.items():
            if android_attribute(application, name) != expected:
                fail(errors, f"model-pack application android:{name} must be {expected}")
        components = [
            node for node in application
            if node.tag in {"activity", "activity-alias", "service", "receiver", "provider"}
        ]
        providers = [node for node in components if node.tag == "provider"]
        if len(components) != 1 or len(providers) != 1:
            fail(errors, "model-pack source must expose exactly one provider and no active components")
        else:
            provider = providers[0]
            expected = {
                "name": MODEL_PACK_PROVIDER_CLASS,
                "authorities": MODEL_PACK_AUTHORITY,
                "exported": "true",
                "grantUriPermissions": "false",
                "directBootAware": "false",
            }
            for name, value in expected.items():
                if android_attribute(provider, name) != value:
                    fail(errors, f"model-pack provider android:{name} must be {value}")

    try:
        settings_text = settings_path.read_text(encoding="utf-8")
        build_text = build_path.read_text(encoding="utf-8")
        provider_text = provider_path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(errors, f"cannot read model-pack build/provider source: {exc}")
        return
    if ("libreboardIncludeModelPack" not in settings_text
            or "include ':modelpack-en-de'" not in settings_text):
        fail(errors, "model-pack module must remain explicitly opt-in")
    required_build_markers = (
        'applicationId = "org.libreboard.model.en_de"',
        'minSdk = 26',
        'targetSdk = 36',
        'libreboardContextModelArchive',
        'noCompress += "lbmodel"',
    )
    if any(marker not in build_text for marker in required_build_markers):
        fail(errors, "model-pack build does not retain its fixed ID, SDKs, archive input, and storage rule")
    provider_markers = (
        'content://" + AUTHORITY + "/model.lbmodel',
        'application/vnd.org.libreboard.model',
        '"r".equals(mode)',
    )
    if any(marker not in provider_text for marker in provider_markers):
        fail(errors, "model-pack provider no longer exposes one fixed read-only archive contract")
    committed_assets = [
        path for path in (ROOT / "modelpack-en-de/src").rglob("*")
        if path.is_file() and path.suffix.lower() in {".lbmodel", ".onnx", ".aar", ".so"}
    ]
    if committed_assets:
        fail(errors, "generated model-pack binaries must not be committed as source")


def sdk_path() -> pathlib.Path | None:
    configured = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if configured:
        return pathlib.Path(configured)
    props = ROOT / "local.properties"
    if props.exists():
        for line in props.read_text(encoding="utf-8").splitlines():
            if line.startswith("sdk.dir="):
                return pathlib.Path(line.split("=", 1)[1].replace("\\:", ":").replace("\\\\", "\\"))
    return None


def newest_tool(name: str, pattern: str) -> pathlib.Path | None:
    sdk = sdk_path()
    if sdk is None:
        return None
    matches = sorted(sdk.glob(pattern), reverse=True)
    return next((path for path in matches if path.name == name and os.access(path, os.X_OK)), None)


def source_checks(errors: list[str]) -> None:
    manifest_path = ROOT / "app/src/main/AndroidManifest.xml"
    manifest = ET.parse(manifest_path).getroot()
    permissions = {node.attrib.get(ANDROID_NS + "name") for node in manifest.findall("uses-permission")}
    forbidden = sorted(FORBIDDEN_PERMISSIONS & permissions)
    if forbidden:
        fail(errors, f"source manifest contains forbidden permissions: {', '.join(forbidden)}")

    application = manifest.find("application")
    if application is None:
        fail(errors, "source manifest has no application element")
    else:
        required_attributes = {
            "allowBackup": "false",
            "usesCleartextTraffic": "false",
            "directBootAware": "true",
            "fullBackupContent": "@xml/backup_rules",
            "dataExtractionRules": "@xml/data_extraction_rules",
        }
        for name, expected in required_attributes.items():
            if android_attribute(application, name) != expected:
                fail(errors, f"application android:{name} must be {expected}")

        latin_ime = next(
            (service for service in application.findall("service")
             if android_attribute(service, "name") in {"LatinIME", LATIN_IME_CLASS}),
            None,
        )
        if latin_ime is None:
            fail(errors, "source manifest has no LibreBoard input-method service")
        else:
            if android_attribute(latin_ime, "permission") != "android.permission.BIND_INPUT_METHOD":
                fail(errors, "input-method service must require BIND_INPUT_METHOD")
            if android_attribute(latin_ime, "directBootAware") != "true":
                fail(errors, "input-method service must remain Direct Boot aware")

        clipboard_provider = next(
            (provider for provider in application.findall("provider")
             if android_attribute(provider, "name") == CLIPBOARD_PROVIDER_CLASS),
            None,
        )
        if clipboard_provider is None:
            fail(errors, "source manifest has no clipboard content provider")
        elif android_attribute(clipboard_provider, "exported") != "false":
            fail(errors, "clipboard content provider must not be exported")

    queries = manifest.find("queries")
    queried_model_providers = [] if queries is None else [
        provider for provider in queries.findall("provider")
        if android_attribute(provider, "authorities") == MODEL_PACK_AUTHORITY
    ]
    if len(queried_model_providers) != 1:
        fail(errors, "core manifest must query exactly the official model-pack provider")

    validate_backup_exclusions(errors)
    validate_model_pack_source(errors)
    validate_store_metadata(errors)
    validate_gradle_dependency_verification(errors)

    source_root = ROOT / "app/src/main/java"
    validate_no_content_bearing_logs(errors, source_root)
    validate_native_logging_lock(errors, ROOT / "app/src/main/jni/Android.mk")
    dynamic_load = re.compile(r"System\s*\.\s*load\s*\(")
    load_library = re.compile(r"System\s*\.\s*loadLibrary\s*\(\s*([^)]*)\)")
    for path in source_root.rglob("*"):
        if path.suffix not in {".kt", ".java"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "sHaveGestureLib" in text:
            fail(errors, "obsolete external gesture-library availability switch was reintroduced")
        if dynamic_load.search(text):
            fail(errors, f"dynamic native path loading is forbidden: {path.relative_to(ROOT)}")
        for match in load_library.finditer(text):
            if "JNI_LIB_NAME" not in match.group(1):
                fail(errors, f"unapproved System.loadLibrary call: {path.relative_to(ROOT)}")

    removed_paths = (
        source_root / "helium314/keyboard/settings/preferences/LoadGestureLibPreference.kt",
        source_root / "helium314/keyboard/settings/screens/gesturedata",
        source_root / "helium314/keyboard/latin/utils/GestureDataGathering.kt",
        ROOT / "tools/release.py",
    )
    for path in removed_paths:
        if path.is_file() or (path.is_dir() and any(child.is_file() for child in path.rglob("*"))):
            fail(errors, f"removed executable/data-collection path was reintroduced: {path.relative_to(ROOT)}")

    build_files = [
        path
        for pattern in ("*.gradle", "*.gradle.kts", "*.toml")
        for path in ROOT.rglob(pattern)
        if path.is_file() and not {"build", ".gradle"}.intersection(path.relative_to(ROOT).parts)
    ]
    build_text = "\n".join(path.read_text(encoding="utf-8") for path in build_files).lower()
    for marker in FORBIDDEN_DEPENDENCY_MARKERS:
        if marker in build_text:
            fail(errors, f"forbidden network/proprietary dependency marker: {marker}")

    try:
        runtime_settings = build_onnxruntime_android.load_settings()
    except build_onnxruntime_android.BuildConfigurationError as exc:
        fail(errors, f"ONNX Runtime build settings are invalid: {exc}")
    else:
        if runtime_settings.get("sourceCommit") != ONNXRUNTIME_COMMIT:
            fail(errors, "ONNX Runtime build settings do not pin the approved commit")
        if runtime_settings.get("ndkRevision") != "28.0.13004108":
            fail(errors, "ONNX Runtime build settings do not pin the application NDK")
    gitmodules = ROOT / ".gitmodules"
    if not gitmodules.is_file():
        fail(errors, "pinned ONNX Runtime source submodule is missing")
    else:
        module_text = gitmodules.read_text(encoding="utf-8")
        if ("path = third_party/onnxruntime" not in module_text
                or "url = https://github.com/microsoft/onnxruntime.git" not in module_text):
            fail(errors, "ONNX Runtime submodule declaration is not approved")
    if (ROOT / ".git").exists():
        submodule_entry = run([
            "git", "-C", str(ROOT), "ls-files", "--stage", "--", "third_party/onnxruntime",
        ])
        expected = f"160000 {ONNXRUNTIME_COMMIT} 0\tthird_party/onnxruntime"
        if submodule_entry.returncode != 0 or submodule_entry.stdout.strip() != expected:
            fail(errors, "ONNX Runtime gitlink does not pin the approved commit")

    committed_runtime_binaries = [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and "third_party" not in path.relative_to(ROOT).parts
        and path.suffix.lower() in {".aar", ".so"}
        and "onnxruntime" in path.name.lower()
        and "build" not in path.relative_to(ROOT).parts
    ]
    if committed_runtime_binaries:
        fail(errors, "generated ONNX Runtime binaries must not be committed as source")

    try:
        source_manifest = model_sources.load_manifest()
        swipe_policy = prepare_swipe_dataset.load_policy()
        prepare_tap_evaluation.load_policy()
        context_policy = prepare_context_dataset.load_policy()
        context_project_path = (ROOT / context_policy.project_authored_data).resolve()
        context_project = prepare_context_dataset.load_project_corpus(
            context_project_path,
            context_policy,
        )
        context_tokenizer_policy = build_context_tokenizer.load_policy()
        context_distillation_policy = score_context_teacher.load_policy()
        swipe_spec = swipe_model_contract.load_spec()
        context_spec = context_model_contract.load_spec()
        swipe_source = source_manifest.source(swipe_policy.source_id)
        context_source = source_manifest.source(context_policy.source_id)
        teacher_source = source_manifest.source("hanse2-100m-base-teacher-v1")
    except (
        build_context_tokenizer.ContextTokenizerBuildError,
        score_context_teacher.ContextTeacherError,
        model_sources.ModelSourceError,
        prepare_context_dataset.ContextDataError,
        prepare_swipe_dataset.SwipeDataError,
        prepare_tap_evaluation.TapEvaluationDataError,
        swipe_model_contract.SwipeModelContractError,
        context_model_contract.ContextModelContractError,
    ) as exc:
        fail(errors, f"model source/training contract is invalid: {exc}")
    else:
        if swipe_source.repository != "futo-org/swipe.futo.org" or swipe_source.license != "MIT":
            fail(errors, "swipe training must remain pinned to the MIT FUTO gesture dataset")
        if teacher_source.repository != "Evicka/Hanse2-100M-Base" or teacher_source.license != "Apache-2.0":
            fail(errors, "context distillation teacher must remain the Apache-2.0 Hanse2 base model")
        if (
            context_source.identifier != swipe_source.identifier
            or context_source.repository != "futo-org/swipe.futo.org"
            or context_source.license != "MIT"
        ):
            fail(errors, "context sentences must remain pinned to the MIT FUTO gesture dataset")
        if any(
            source.repository.lower() == "futo-org/futo-swipe"
            or (source.repository.lower().startswith("futo-org/") and source.kind == "teacher-model")
            for source in source_manifest.sources
        ):
            fail(errors, "FUTO model weights or outputs are forbidden; only the MIT gesture dataset is approved")
        if swipe_spec.raw.get("parameterCount") > 1_000_000:
            fail(errors, "swipe model exceeds the audited one-million-parameter architecture ceiling")
        if context_spec.raw.get("parameterCount") != 35_662_848:
            fail(errors, "context model has drifted from the audited parameter budget")
        if context_spec.export.get("quantization") != "INT4_BLOCK128":
            fail(errors, "context model must retain the audited blockwise INT4 export")
        if context_spec.training != {
            "seed": 24_012_026,
            "epochs": 4,
            "batchSize": 128,
            "learningRate": 0.0003,
            "weightDecay": 0.1,
            "gradientClip": 1.0,
            "teacherTemperature": 1.0,
            "rankingMargin": 0.2,
            "teacherLossWeight": 1.0,
            "observedLossWeight": 0.5,
            "rankingLossWeight": 0.25,
            "shuffleBufferRecords": 4_096,
            "checkpointEveryExamples": 4_096,
        }:
            fail(errors, "context model training recipe has drifted from the audited distillation contract")
        if context_spec.export.get("maximumModelBytes", CONTEXT_MODEL_MAXIMUM_BYTES + 1) > CONTEXT_MODEL_MAXIMUM_BYTES:
            fail(errors, "context model spec exceeds the sidecar payload ceiling")
        corpus_manifest_path = swipe_model_contract.DEFAULT_CORPUS_MANIFEST
        try:
            corpus_manifest = json.loads(corpus_manifest_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            fail(errors, f"committed swipe corpus manifest is invalid: {exc}")
        else:
            if corpus_manifest.get("sourceManifestSha256") != source_manifest.sha256:
                fail(errors, "swipe corpus manifest is not bound to the current source manifest")
            if corpus_manifest.get("policySha256") != swipe_policy.sha256:
                fail(errors, "swipe corpus manifest is not bound to the current data policy")
            if corpus_manifest.get("toolSha256") != model_sources.file_sha256(ROOT / "tools/prepare_swipe_dataset.py"):
                fail(errors, "swipe corpus manifest is not bound to the current preparation tool")
            if corpus_manifest.get("source") != {
                "id": swipe_source.identifier,
                "license": swipe_source.license,
                "revision": swipe_source.revision,
            }:
                fail(errors, "swipe corpus manifest has unexpected source provenance")
            counts = corpus_manifest.get("counts")
            outputs = corpus_manifest.get("outputs")
            strata = corpus_manifest.get("strata")
            if not isinstance(counts, dict) or counts.get("acceptedRows", 0) < 100_000:
                fail(errors, "swipe corpus manifest has insufficient accepted training data")
            elif counts.get("rejectedFraction", 1.0) > swipe_policy.maximum_rejected_fraction:
                fail(errors, "swipe corpus manifest exceeds the rejection ceiling")
            if not isinstance(outputs, dict) or set(outputs) != {
                "train.jsonl", "validation.jsonl", "test.jsonl", "layout.json",
            }:
                fail(errors, "swipe corpus manifest has incomplete outputs")
            else:
                held_out = outputs.get("test.jsonl")
                if not isinstance(held_out, dict) or held_out.get("records", 0) < 5_000:
                    fail(errors, "swipe corpus manifest has fewer than 5000 held-out gestures")
            test_strata = strata.get("test") if isinstance(strata, dict) else None
            if not isinstance(test_strata, dict) or any(
                test_strata.get(name, 0) < 500
                for name in ("short", "return_trip", "double_letter", "sloppy", "very_sloppy")
            ):
                fail(errors, "swipe corpus test split is not adequately stratified")

        if context_project_path != ROOT / "models/context/project-authored-de-v1.json":
            fail(errors, "context data policy must retain the reviewed project-authored German corpus")
        if context_policy.data_artifacts != ("train.jsonl", "dev.jsonl", "test.jsonl"):
            fail(errors, "context data policy must retain every pinned FUTO sentence split")
        if context_policy.split_basis_points != {"train": 9_000, "validation": 500, "test": 500}:
            fail(errors, "context data policy must retain the session-separated 90/5/5 split")
        if (
            context_policy.minimum_accepted_sentences < 50_000
            or context_policy.minimum_german_sentences < 10_000
            or context_policy.maximum_rejected_fraction > 0.25
        ):
            fail(errors, "context data policy has weakened its corpus quality floors")
        if (
            context_tokenizer_policy.vocabulary_size != 16_384
            or context_tokenizer_policy.minimum_frequency != 2
            or context_tokenizer_policy.special_tokens
            != ("<pad>", "<bos>", "<unk>", "<lang:en>", "<lang:de>")
            or context_tokenizer_policy.languages != {"en": "<lang:en>", "de": "<lang:de>"}
        ):
            fail(errors, "context tokenizer policy has drifted from the Android model ABI")
        if (
            context_distillation_policy.teacher_source_id != teacher_source.identifier
            or context_distillation_policy.tokenizer_sha256
            != "1395e285927bfbfa5888dc7c83e4f57dfcfbfeb54f29a3cf2437f5db3d71d6a2"
            or context_distillation_policy.maximum_prefix_student_tokens != 22
            or context_distillation_policy.maximum_candidate_student_tokens != 8
            or context_distillation_policy.maximum_candidates > 32
            or context_distillation_policy.minimum_scored_records < 50_000
            or context_distillation_policy.minimum_german_records < 10_000
            or context_distillation_policy.minimum_held_out_records < 3_000
        ):
            fail(errors, "context distillation policy has drifted from the audited teacher/student contract")

        context_corpus_path = prepare_context_dataset.DEFAULT_CORPUS_MANIFEST
        try:
            context_corpus = json.loads(context_corpus_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(errors, f"committed context corpus manifest is invalid: {exc}")
        else:
            if not isinstance(context_corpus, dict) or set(context_corpus) != prepare_context_dataset.PREPARED_MANIFEST_KEYS:
                fail(errors, "context corpus manifest has an unexpected schema")
            elif (
                context_corpus.get("schemaVersion") != 1
                or context_corpus.get("modelId") != "context-en-de-v1"
                or context_corpus.get("minimumsEnforced") is not True
            ):
                fail(errors, "context corpus manifest is not a release-eligible v1 corpus")
            else:
                expected_sources = {
                    "external": {
                        "id": context_source.identifier,
                        "license": context_source.license,
                        "revision": context_source.revision,
                        "sourceUrl": context_source.source_url,
                    },
                    "projectAuthored": {
                        "id": context_project.raw["id"],
                        "license": context_project.raw["license"],
                        "path": context_policy.project_authored_data,
                        "sha256": context_project.sha256,
                    },
                }
                if context_corpus.get("sourceManifestSha256") != source_manifest.sha256:
                    fail(errors, "context corpus manifest is not bound to the current source manifest")
                if context_corpus.get("policySha256") != context_policy.sha256:
                    fail(errors, "context corpus manifest is not bound to the current data policy")
                if context_corpus.get("projectDataSha256") != context_project.sha256:
                    fail(errors, "context corpus manifest is not bound to the project-authored data")
                if context_corpus.get("toolSha256") != model_sources.file_sha256(
                    ROOT / "tools/prepare_context_dataset.py"
                ):
                    fail(errors, "context corpus manifest is not bound to the current preparation tool")
                if context_corpus.get("sources") != expected_sources:
                    fail(errors, "context corpus manifest has unexpected source provenance")

                counts = context_corpus.get("counts")
                outputs = context_corpus.get("outputs")
                if not isinstance(counts, dict):
                    fail(errors, "context corpus manifest has invalid counts")
                else:
                    accepted = counts.get("acceptedSentences")
                    german = counts.get("language:de")
                    invalid_fraction = counts.get("invalidSourceFraction")
                    if (
                        isinstance(accepted, bool)
                        or not isinstance(accepted, int)
                        or accepted < context_policy.minimum_accepted_sentences
                        or isinstance(german, bool)
                        or not isinstance(german, int)
                        or german < context_policy.minimum_german_sentences
                        or not finite_number(invalid_fraction)
                        or float(invalid_fraction) > context_policy.maximum_rejected_fraction
                    ):
                        fail(errors, "context corpus manifest does not satisfy its data quality floors")
                expected_outputs = {
                    "train.sentences.jsonl", "validation.sentences.jsonl", "test.sentences.jsonl",
                }
                if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
                    fail(errors, "context corpus manifest has incomplete outputs")
                else:
                    total_records = 0
                    for filename, details in outputs.items():
                        if (
                            not isinstance(details, dict)
                            or set(details) != prepare_context_dataset.PREPARED_OUTPUT_KEYS
                            or isinstance(details.get("bytes"), bool)
                            or not isinstance(details.get("bytes"), int)
                            or details["bytes"] <= 0
                            or not isinstance(details.get("sha256"), str)
                            or not SHA256.fullmatch(details["sha256"])
                            or isinstance(details.get("records"), bool)
                            or not isinstance(details.get("records"), int)
                            or details["records"] <= 0
                            or isinstance(details.get("sessions"), bool)
                            or not isinstance(details.get("sessions"), int)
                            or details["sessions"] <= 0
                            or not isinstance(details.get("languages"), dict)
                            or set(details["languages"]) != {"en-US", "de"}
                            or any(
                                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                                for value in details["languages"].values()
                            )
                            or sum(details["languages"].values()) != details["records"]
                        ):
                            fail(errors, f"context corpus output metadata is invalid: {filename}")
                            continue
                        total_records += details["records"]
                    if isinstance(counts, dict) and total_records != counts.get("acceptedSentences"):
                        fail(errors, "context corpus output counts do not match accepted sentences")
                    held_out = outputs.get("test.sentences.jsonl")
                    if isinstance(held_out, dict) and (
                        held_out.get("languages", {}).get("en-US", 0) < 2_000
                        or held_out.get("languages", {}).get("de", 0) < 1_000
                    ):
                        fail(errors, "context corpus has insufficient bilingual held-out sentences")

            if isinstance(context_corpus, dict):
                distillation_manifest_path = ROOT / "models/context/distillation-manifest.json"
                try:
                    distillation_manifest = json.loads(distillation_manifest_path.read_bytes())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    fail(errors, f"committed context distillation manifest is invalid: {exc}")
                else:
                    if not isinstance(distillation_manifest, dict):
                        fail(errors, "context distillation manifest must be a JSON object")
                    else:
                        validate_context_distillation_manifest(
                            errors,
                            distillation_manifest,
                            context_corpus=context_corpus,
                            policy=context_distillation_policy,
                            teacher_source=teacher_source,
                        )

    validate_hash_locked_requirements(
        errors,
        ROOT / "models/training/requirements-linux-x86_64.lock",
        "model training",
        ("numpy==2.2.6", "onnx==1.19.0", "safetensors==0.6.2", "torch==2.8.0+cpu"),
    )
    validate_hash_locked_requirements(
        errors,
        ROOT / "models/training/requirements-context-linux-x86_64.lock",
        "context model",
        (
            "numpy==2.2.6",
            "onnx==1.19.0",
            "onnx-ir==1.0.0",
            "onnxruntime==1.26.0",
            "safetensors==0.6.2",
            "tokenizers==0.22.2",
            "torch==2.8.0+cpu",
            "transformers==4.57.6",
        ),
    )
    validate_hash_locked_requirements(
        errors,
        ROOT / "models/training/requirements-onnxruntime-build-linux-x86_64.lock",
        "ONNX Runtime build",
        (
            "flatbuffers==25.12.19",
            "numpy==2.2.6",
            "onnxruntime==1.26.0",
            "packaging==26.3",
            "protobuf==7.36.1",
        ),
    )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(command, 126, "", f"cannot start command: {exc}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(errors: list[str], path: pathlib.Path, label: str) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(errors, f"cannot read {label}: {exc}")
        return None
    if not isinstance(value, dict):
        fail(errors, f"{label} must be a JSON object")
        return None
    return value


def validate_phase0_report(errors: list[str], report: dict, apk_hash: str) -> dict | None:
    if report.get("schemaVersion") != evaluate_engine.SCHEMA_VERSION:
        fail(errors, "Phase 0 report has an unsupported schema")
    if report.get("passed") is not True:
        fail(errors, "Phase 0 report did not pass")
    checks = report.get("checks")
    if (not isinstance(checks, dict)
            or set(checks) != PHASE0_CHECKS
            or any(value is not True for value in checks.values())):
        fail(errors, "Phase 0 report does not contain every passing release check")
    counts = report.get("counts")
    if not isinstance(counts, dict) or any(
        isinstance(counts.get(category), bool)
        or not isinstance(counts.get(category), int)
        or counts[category] < minimum
        for category, minimum in MINIMUM_PHASE0_COUNTS.items()
    ):
        fail(errors, "Phase 0 report does not satisfy held-out dataset minimums")
    swipe_strata = report.get("swipeStrataCounts")
    if not isinstance(swipe_strata, dict) or any(
        isinstance(swipe_strata.get(stratum), bool)
        or not isinstance(swipe_strata.get(stratum), int)
        or swipe_strata[stratum] < minimum
        for stratum, minimum in MINIMUM_PHASE0_SWIPE_STRATA_COUNTS.items()
    ):
        fail(errors, "Phase 0 report does not satisfy swipe stratum minimums")
    valid_word_counts = report.get("validWordCounts")
    if (
        not isinstance(valid_word_counts, dict)
        or set(valid_word_counts) != {"correct", "keep"}
        or not isinstance(valid_word_counts.get("correct"), int)
        or isinstance(valid_word_counts.get("correct"), bool)
        or valid_word_counts["correct"] < evaluate_engine.MINIMUM_VALID_WORD_CORRECTIONS
        or not isinstance(valid_word_counts.get("keep"), int)
        or isinstance(valid_word_counts.get("keep"), bool)
        or valid_word_counts["keep"] < evaluate_engine.MINIMUM_VALID_WORD_KEEPS
    ):
        fail(errors, "Phase 0 report does not satisfy valid-word correction and keep minimums")
    environment_counts = report.get("environmentCounts")
    if (
        not isinstance(environment_counts, dict)
        or set(environment_counts) != PHASE0_ENVIRONMENTS
        or any(
            not isinstance(environment_counts.get(kind), dict)
            or not isinstance(environment_counts[kind].get("tap"), int)
            or isinstance(environment_counts[kind].get("tap"), bool)
            or environment_counts[kind]["tap"] < evaluate_engine.MINIMUM_ENVIRONMENT_TAP_SAMPLES
            or not isinstance(environment_counts[kind].get("swipe"), int)
            or isinstance(environment_counts[kind].get("swipe"), bool)
            or environment_counts[kind]["swipe"] < evaluate_engine.MINIMUM_ENVIRONMENT_SWIPE_SAMPLES
            for kind in PHASE0_ENVIRONMENTS
        )
    ):
        fail(errors, "Phase 0 report does not bind sufficient measurements to every environment")
    environment_latency = report.get("environmentLatencyMs")
    if (
        not isinstance(environment_latency, dict)
        or set(environment_latency) != PHASE0_ENVIRONMENTS
        or any(
            not isinstance(environment_latency.get(kind), dict)
            or not isinstance(environment_latency[kind].get(path), dict)
            or not finite_number(environment_latency[kind][path].get("p95"))
            for kind in PHASE0_ENVIRONMENTS
            for path in ("tap", "swipe")
        )
    ):
        fail(errors, "Phase 0 report has invalid per-environment latency evidence")

    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        fail(errors, "Phase 0 report has no artifact or environment evidence")
        return None
    if evidence.get("coreApkSha256") != apk_hash:
        fail(errors, "Phase 0 report was produced with a different APK")
    if not isinstance(evidence.get("appCommit"), str) or not GIT_COMMIT.fullmatch(evidence["appCommit"]):
        fail(errors, "Phase 0 report has an invalid app commit")
    for field in ("swipeModelSha256", "contextModelSha256"):
        if not isinstance(evidence.get(field), str) or not SHA256.fullmatch(evidence[field]):
            fail(errors, f"Phase 0 report has an invalid {field}")
    if (
        not isinstance(evidence.get("measurementDatasetSha256"), str)
        or not SHA256.fullmatch(evidence["measurementDatasetSha256"])
    ):
        fail(errors, "Phase 0 report has an invalid measurementDatasetSha256")
    environments = evidence.get("environments")
    graphene = None
    if isinstance(environments, list):
        kinds = [item.get("kind") for item in environments if isinstance(item, dict)]
        if len(environments) != 3 or set(kinds) != PHASE0_ENVIRONMENTS or len(kinds) != len(set(kinds)):
            fail(errors, "Phase 0 report does not contain the three reference environments")
        matches = [item for item in environments
                   if isinstance(item, dict) and item.get("kind") == "grapheneos_hardware"]
        if len(matches) == 1:
            graphene = matches[0]
    if graphene is None:
        fail(errors, "Phase 0 report has no unique GrapheneOS hardware run")
    elif (graphene.get("physicalDevice") is not True
          or graphene.get("sandboxedGooglePlayInstalled") is not False):
        fail(errors, "Phase 0 GrapheneOS run is not qualifying physical hardware")
    return evidence


def finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def validate_grapheneos_evidence(errors: list[str], report: dict, apk: pathlib.Path, apk_hash: str) -> None:
    if report.get("schemaVersion") != 2 or report.get("status") != "PASS":
        fail(errors, "GrapheneOS evidence must use schema 2 with PASS status")
    if report.get("apkFilename") != apk.name or report.get("apkSha256") != apk_hash:
        fail(errors, "GrapheneOS evidence does not identify the verified APK")
    if not isinstance(report.get("appCommit"), str) or not GIT_COMMIT.fullmatch(report["appCommit"]):
        fail(errors, "GrapheneOS evidence has an invalid app commit")
    for field in (
        "deviceModel",
        "grapheneOsBuildNumber",
        "buildFingerprint",
        "testerId",
        "phase0TestRunId",
    ):
        if not isinstance(report.get(field), str) or not report[field] or len(report[field]) > 512:
            fail(errors, f"GrapheneOS evidence requires {field}")
    if (not isinstance(report.get("securityPatchLevel"), str)
            or not SECURITY_PATCH.fullmatch(report["securityPatchLevel"])):
        fail(errors, "GrapheneOS evidence has an invalid security patch level")
    if not isinstance(report.get("testedAtUtc"), str) or not UTC_TIMESTAMP.fullmatch(report["testedAtUtc"]):
        fail(errors, "GrapheneOS evidence has an invalid UTC test timestamp")
    api_level = report.get("apiLevel")
    if isinstance(api_level, bool) or not isinstance(api_level, int) or api_level < 35 or api_level > 100:
        fail(errors, "GrapheneOS evidence must come from Android 15+ hardware")
    if report.get("physicalDevice") is not True:
        fail(errors, "GrapheneOS evidence must come from a physical device")
    if report.get("sandboxedGooglePlayInstalled") is not False:
        fail(errors, "GrapheneOS evidence must run without sandboxed Google Play")
    if report.get("exploitProtectionCompatibilityModeEnabled") is not False:
        fail(errors, "GrapheneOS evidence must keep exploit protection compatibility mode disabled")

    checks = report.get("checks")
    if (not isinstance(checks, dict)
            or set(checks) != GRAPHENEOS_CHECKS
            or any(value is not True for value in checks.values())):
        fail(errors, "GrapheneOS evidence does not contain every passing device check")

    measurements = report.get("measurements")
    if not isinstance(measurements, dict):
        fail(errors, "GrapheneOS evidence has no measurements")
    else:
        for name, budget in (("tapLatencyMs", 80.0), ("swipeLatencyMs", 200.0)):
            latency = measurements.get(name)
            if (not isinstance(latency, dict)
                    or any(not finite_number(latency.get(key)) for key in ("p50", "p95", "p99"))):
                fail(errors, f"GrapheneOS evidence has invalid {name}")
            elif not (latency["p50"] <= latency["p95"] <= latency["p99"]):
                fail(errors, f"GrapheneOS {name} percentiles are not ordered")
            elif latency["p95"] > budget:
                fail(errors, f"GrapheneOS {name} exceeds the p95 budget")
        for field in (
            "coldStartMs",
            "warmStartMs",
            "peakRssMiB",
            "neuralTimeoutCount",
            "circuitBreakerActivationCount",
        ):
            if not finite_number(measurements.get(field)):
                fail(errors, f"GrapheneOS evidence has invalid {field}")

    for field in (
        "instrumentationOutputSha256",
        "phase0ReportSha256",
        "swipeModelSha256",
        "contextModelSha256",
    ):
        if not isinstance(report.get(field), str) or not SHA256.fullmatch(report[field]):
            fail(errors, f"GrapheneOS evidence requires {field}")


def evidence_checks(
    errors: list[str],
    apk: pathlib.Path,
    rebuilt_apk: pathlib.Path,
    phase0_path: pathlib.Path,
    phase0_measurements_path: pathlib.Path,
    grapheneos_path: pathlib.Path,
    instrumentation_path: pathlib.Path,
) -> None:
    if not apk.is_file() or not rebuilt_apk.is_file():
        fail(errors, "both reproducibility APKs must exist")
        return
    apk_hash = sha256_file(apk)
    rebuilt_hash = sha256_file(rebuilt_apk)
    if apk_hash != rebuilt_hash:
        fail(errors, "clean rebuild APK is not byte-identical")

    phase0 = read_json_object(errors, phase0_path, "Phase 0 report")
    grapheneos = read_json_object(errors, grapheneos_path, "GrapheneOS evidence")
    phase0_evidence = validate_phase0_report(errors, phase0, apk_hash) if phase0 is not None else None
    if grapheneos is None:
        return
    validate_grapheneos_evidence(errors, grapheneos, apk, apk_hash)
    if phase0_evidence is None:
        return

    if not phase0_measurements_path.is_file():
        fail(errors, "Phase 0 measurement dataset does not exist")
    else:
        measurement_hash = sha256_file(phase0_measurements_path)
        if phase0_evidence.get("measurementDatasetSha256") != measurement_hash:
            fail(errors, "Phase 0 report references a different measurement dataset")
        try:
            measurements = evaluate_engine.read_jsonl(phase0_measurements_path)
            recomputed = evaluate_engine.evaluate(
                measurements,
                {"schemaVersion": evaluate_engine.SCHEMA_VERSION, **phase0_evidence},
                measurement_sha256=measurement_hash,
                enforce_minimum_counts=True,
            )
        except (OSError, evaluate_engine.EvaluationError) as exc:
            fail(errors, f"Phase 0 measurements cannot be independently evaluated: {exc}")
        else:
            if recomputed != phase0:
                fail(errors, "Phase 0 report does not exactly match its measurement dataset")

    if grapheneos.get("phase0ReportSha256") != sha256_file(phase0_path):
        fail(errors, "GrapheneOS evidence references a different Phase 0 report")
    if not instrumentation_path.is_file():
        fail(errors, "GrapheneOS instrumentation output does not exist")
    elif grapheneos.get("instrumentationOutputSha256") != sha256_file(instrumentation_path):
        fail(errors, "GrapheneOS evidence references different instrumentation output")
    for field in ("appCommit", "swipeModelSha256", "contextModelSha256"):
        if grapheneos.get(field) != phase0_evidence.get(field):
            fail(errors, f"GrapheneOS and Phase 0 evidence disagree on {field}")
    environments = phase0_evidence.get("environments", [])
    phase0_graphene = next(
        (item for item in environments if isinstance(item, dict) and item.get("kind") == "grapheneos_hardware"),
        None,
    )
    if phase0_graphene is not None:
        matches = {
            "deviceModel": "deviceModel",
            "grapheneOsBuildNumber": "grapheneOsBuildNumber",
            "buildFingerprint": "buildFingerprint",
            "phase0TestRunId": "testRunId",
        }
        for report_field, environment_field in matches.items():
            if grapheneos.get(report_field) != phase0_graphene.get(environment_field):
                fail(errors, f"GrapheneOS device evidence disagrees with Phase 0 on {report_field}")
        phase0_latency = phase0.get("environmentLatencyMs", {}).get("grapheneos_hardware", {})
        graphene_measurements = grapheneos.get("measurements", {})
        for report_field, phase0_field in (("tapLatencyMs", "tap"), ("swipeLatencyMs", "swipe")):
            if graphene_measurements.get(report_field) != phase0_latency.get(phase0_field):
                fail(errors, f"GrapheneOS {report_field} disagrees with its Phase 0 device run")


def verify_model_signature(
    errors: list[str],
    public_key: pathlib.Path,
    manifest: bytes,
    signature: bytes,
    label: str,
) -> None:
    if (
        not public_key.is_file()
        or public_key.is_symlink()
        or public_key.stat().st_size <= 0
        or public_key.stat().st_size > 8 * 1024
    ):
        fail(errors, f"{label} public key is missing, symlinked, empty, or oversized")
        return
    description = run([
        "openssl", "pkey", "-pubin", "-inform", "DER", "-in", str(public_key),
        "-text", "-noout",
    ])
    key_match = re.search(r"Public-Key:\s*\((\d+) bit", description.stdout)
    exponent_match = re.search(r"Exponent:\s*(\d+)", description.stdout)
    if (
        description.returncode != 0
        or key_match is None
        or int(key_match.group(1)) < 3072
        or "modulus:" not in description.stdout.lower()
        or exponent_match is None
        or int(exponent_match.group(1)) != 65_537
    ):
        fail(errors, f"{label} public key must be a valid RSA key of at least 3072 bits with exponent 65537")
        return
    with tempfile.TemporaryDirectory(prefix="libreboard-model-signature-") as temporary:
        root = pathlib.Path(temporary)
        manifest_path = root / "manifest.json"
        signature_path = root / "signature.der"
        manifest_path.write_bytes(manifest)
        signature_path.write_bytes(signature)
        verification = run([
            "openssl", "dgst", "-sha256", "-verify", str(public_key), "-keyform", "DER",
            "-signature", str(signature_path), str(manifest_path),
        ])
    if verification.returncode != 0 or verification.stdout.strip() != "Verified OK":
        fail(errors, f"{label} signature is not trusted by the supplied project key")


def model_archive_checks(
    errors: list[str],
    archive_bytes: bytes,
    label: str = "model archive",
    public_key: pathlib.Path | None = None,
    model_id: str = "context-en-de-v1",
) -> None:
    if model_id == "context-en-de-v1":
        expected_entries = CONTEXT_MODEL_ARCHIVE_ENTRIES
        maximum_archive_bytes = MODEL_PACK_MAXIMUM_BYTES
        maximum_model_bytes = CONTEXT_MODEL_MAXIMUM_BYTES
        expected_kind = "context-rescorer"
        maximum_parameter_count = 40_000_000
        requires_tokenizer = True
    elif model_id == "swipe-latin-v1":
        expected_entries = SWIPE_MODEL_ARCHIVE_ENTRIES
        maximum_archive_bytes = SWIPE_ARCHIVE_MAXIMUM_BYTES
        maximum_model_bytes = SWIPE_MODEL_MAXIMUM_BYTES
        expected_kind = "swipe-ctc"
        maximum_parameter_count = 1_000_000
        requires_tokenizer = False
    else:
        fail(errors, f"{label} has an unsupported model identity")
        return
    if not archive_bytes or len(archive_bytes) > maximum_archive_bytes:
        fail(errors, f"{label} must be non-empty and at most {maximum_archive_bytes} bytes")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (len(entries) != len(expected_entries)
                    or len(names) != len(set(names))
                    or set(names) != expected_entries
                    or any(entry.is_dir() for entry in entries)):
                fail(errors, f"{label} must contain exactly the approved data entries")
                return
            if (
                names != sorted(expected_entries)
                or archive.comment
                or any(
                    entry.date_time != MODEL_ARCHIVE_TIMESTAMP
                    or entry.compress_type != zipfile.ZIP_STORED
                    or entry.create_system != 3
                    or entry.external_attr != MODEL_ARCHIVE_EXTERNAL_ATTRIBUTES
                    or entry.flag_bits != 0
                    or entry.extra
                    or entry.comment
                    for entry in entries
                )
            ):
                fail(errors, f"{label} is not in the canonical deterministic ZIP format")
                return
            sizes = {entry.filename: entry.file_size for entry in entries}
            maximums = {
                "manifest.json": MODEL_MANIFEST_MAXIMUM_BYTES,
                "model.onnx": maximum_model_bytes,
                "signature.der": MODEL_SIGNATURE_MAXIMUM_BYTES,
            }
            if requires_tokenizer:
                maximums["tokenizer.json"] = MODEL_TOKENIZER_MAXIMUM_BYTES
            if any(sizes[name] <= 0 or sizes[name] > maximum for name, maximum in maximums.items()):
                fail(errors, f"{label} contains an empty or oversized entry")
                return
            manifest_bytes = archive.read("manifest.json")
            model_bytes = archive.read("model.onnx")
            tokenizer_bytes = archive.read("tokenizer.json") if requires_tokenizer else None
            signature_bytes = archive.read("signature.der")
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        fail(errors, f"cannot read {label}: {exc}")
        return

    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        fail(errors, f"cannot parse {label} manifest: {exc}")
        return
    expected_manifest_fields = MODEL_MANIFEST_FIELDS if requires_tokenizer else (
        MODEL_MANIFEST_FIELDS - {"tokenizerSha256"}
    )
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        fail(errors, f"{label} manifest must contain exactly the supported schema fields")
        return
    if manifest.get("schemaVersion") != 1 or manifest.get("engineAbi") != 1:
        fail(errors, f"{label} has an unsupported schema or engine ABI")
    if manifest.get("modelKind") != expected_kind or manifest.get("tensorAbi") != model_id:
        fail(errors, f"{label} is not the official {model_id} tensor contract")
    locales = manifest.get("locales")
    if (not isinstance(locales, list) or len(locales) != 2
            or any(not isinstance(locale, str) for locale in locales)
            or set(locales) != {"en-US", "de"}):
        fail(errors, f"{label} must contain exactly the en-US and de locales")
    if (not isinstance(manifest.get("architecture"), str)
            or not manifest["architecture"] or len(manifest["architecture"]) > 128):
        fail(errors, f"{label} has no architecture")
    parameter_count = manifest.get("parameterCount")
    if (isinstance(parameter_count, bool) or not isinstance(parameter_count, int)
            or parameter_count <= 0 or parameter_count > maximum_parameter_count):
        fail(errors, f"{label} has an invalid parameter count")
    if (not isinstance(manifest.get("quantization"), str)
            or not manifest["quantization"] or len(manifest["quantization"]) > 32):
        fail(errors, f"{label} has no quantization")
    minimum_version = manifest.get("minimumAppVersionCode")
    if (isinstance(minimum_version, bool) or not isinstance(minimum_version, int)
            or minimum_version < 1 or minimum_version > EXPECTED_APP_VERSION_CODE):
        fail(errors, f"{label} has an invalid minimum app version")
    if manifest.get("license") != "Apache-2.0":
        fail(errors, f"{label} model license must be Apache-2.0")

    operators = manifest.get("requiredOnnxOperators")
    if (not isinstance(operators, list) or not operators or len(operators) > 256
            or any(not isinstance(value, str) or not value or len(value) > 256
                   or any(character.isspace() for character in value) for value in operators)
            or len(operators) != len(set(operators))):
        fail(errors, f"{label} has an invalid ONNX operator declaration")
    provenance = manifest.get("provenance")
    if (not isinstance(provenance, list) or not provenance or len(provenance) > 64
            or any(not isinstance(item, dict)
                   or set(item) != {"name", "revision", "license", "source_url"}
                   or any(not isinstance(item.get(key), str) or not item[key]
                          for key in ("name", "revision", "license"))
                   or not isinstance(item.get("source_url"), str)
                   or not item["source_url"].startswith("https://")
                   for item in provenance)):
        fail(errors, f"{label} has invalid model provenance")

    model_hash = manifest.get("modelSha256")
    tokenizer_hash = manifest.get("tokenizerSha256")
    if not isinstance(model_hash, str) or not SHA256.fullmatch(model_hash):
        fail(errors, f"{label} has an invalid model hash")
    elif hashlib.sha256(model_bytes).hexdigest() != model_hash:
        fail(errors, f"{label} model hash does not match its payload")
    if requires_tokenizer:
        if not isinstance(tokenizer_hash, str) or not SHA256.fullmatch(tokenizer_hash):
            fail(errors, f"{label} has an invalid tokenizer hash")
        elif hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_hash:
            fail(errors, f"{label} tokenizer hash does not match its payload")
    if public_key is not None:
        verify_model_signature(errors, public_key, manifest_bytes, signature_bytes, label)


def model_pack_apk_checks(errors: list[str], apk: pathlib.Path, public_key: pathlib.Path) -> None:
    if not apk.is_file():
        fail(errors, f"model-pack APK does not exist: {apk}")
        return

    apkanalyzer = newest_tool("apkanalyzer", "cmdline-tools/*/bin/apkanalyzer")
    if apkanalyzer is None:
        fail(errors, "apkanalyzer not found in the configured Android SDK")
    else:
        result = run([str(apkanalyzer), "manifest", "print", str(apk)])
        if result.returncode != 0:
            fail(errors, f"apkanalyzer failed for model-pack APK: {result.stderr.strip()}")
        else:
            try:
                manifest = ET.fromstring(result.stdout)
            except ET.ParseError as exc:
                fail(errors, f"cannot parse merged model-pack manifest: {exc}")
            else:
                if manifest.attrib.get("package") != MODEL_PACK_APPLICATION_ID:
                    fail(errors, "model-pack APK has an unexpected application ID")
                uses_sdk = manifest.find("uses-sdk")
                if (uses_sdk is None
                        or android_attribute(uses_sdk, "minSdkVersion") != EXPECTED_MIN_SDK
                        or android_attribute(uses_sdk, "targetSdkVersion") != EXPECTED_TARGET_SDK):
                    fail(errors, "model-pack APK must use the core min/target SDK contract")
                if any(node.tag.startswith("uses-permission") for node in manifest):
                    fail(errors, "model-pack APK must request no permissions")
                application = manifest.find("application")
                if application is None:
                    fail(errors, "merged model-pack manifest has no application element")
                else:
                    required_attributes = {
                        "allowBackup": "false",
                        "usesCleartextTraffic": "false",
                        "directBootAware": "false",
                        "hasCode": "true",
                    }
                    for name, expected in required_attributes.items():
                        if android_attribute(application, name) != expected:
                            fail(errors, f"merged model-pack application android:{name} must be {expected}")
                    for name in ("fullBackupContent", "dataExtractionRules"):
                        if not android_attribute(application, name):
                            fail(errors, f"merged model-pack application must retain android:{name}")
                    components = [
                        node for node in application
                        if node.tag in {"activity", "activity-alias", "service", "receiver", "provider"}
                    ]
                    providers = [node for node in components if node.tag == "provider"]
                    if len(components) != 1 or len(providers) != 1:
                        fail(errors, "model-pack APK must expose exactly one provider and no active components")
                    else:
                        provider = providers[0]
                        expected = {
                            "name": MODEL_PACK_PROVIDER_CLASS,
                            "authorities": MODEL_PACK_AUTHORITY,
                            "exported": "true",
                            "grantUriPermissions": "false",
                            "directBootAware": "false",
                        }
                        for name, value in expected.items():
                            if android_attribute(provider, name) != value:
                                fail(errors, f"merged model-pack provider android:{name} must be {value}")

    zipalign = newest_tool("zipalign", "build-tools/*/zipalign")
    if zipalign is None:
        fail(errors, "zipalign not found in the configured Android SDK")
    else:
        result = run([str(zipalign), "-c", "-P", "16", "4", str(apk)])
        if result.returncode != 0:
            fail(errors, "model-pack APK fails 16 KiB ZIP alignment")

    try:
        with zipfile.ZipFile(apk) as archive:
            model_assets = [name for name in archive.namelist() if name.endswith(".lbmodel")]
            if model_assets != [MODEL_PACK_ASSET]:
                fail(errors, "model-pack APK must contain exactly assets/model.lbmodel")
            else:
                asset = archive.getinfo(MODEL_PACK_ASSET)
                if asset.compress_type != zipfile.ZIP_STORED:
                    fail(errors, "model-pack archive asset must be stored uncompressed")
                if asset.file_size <= 0 or asset.file_size > MODEL_PACK_MAXIMUM_BYTES:
                    fail(errors, "model-pack archive asset is empty or oversized")
                else:
                    model_archive_checks(
                        errors,
                        archive.read(asset),
                        "packaged model archive",
                        public_key,
                    )
            dex_entries = [name for name in archive.namelist() if name.endswith(".dex")]
            if dex_entries != ["classes.dex"]:
                fail(errors, "model-pack APK must contain exactly one provider DEX")
            native_entries = [name for name in archive.namelist() if name.endswith(".so")]
            if native_entries:
                fail(errors, "model-pack APK must not contain native code")
            for dex_entry in dex_entries:
                dex = archive.read(dex_entry).lower()
                for marker in FORBIDDEN_DEPENDENCY_MARKERS:
                    if (marker.encode("ascii") in dex
                            or marker.replace(".", "/").encode("ascii") in dex):
                        fail(errors, f"model-pack DEX contains forbidden dependency marker {marker}")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        fail(errors, f"cannot inspect model-pack APK: {exc}")


def onnx_runtime_apk_entry_checks(
    errors: list[str],
    native_entries: list[str],
    signed_model_packaged: bool,
) -> None:
    ort_names = {"libonnxruntime.so", "libonnxruntime4j_jni.so"}
    ort_by_abi: dict[str, set[str]] = {}
    for entry in native_entries:
        path = pathlib.PurePosixPath(entry)
        if path.name not in ort_names:
            continue
        if len(path.parts) < 3:
            fail(errors, f"ONNX Runtime library has an invalid APK path: {entry}")
            continue
        ort_by_abi.setdefault(path.parts[-2], set()).add(path.name)
    for abi, libraries in ort_by_abi.items():
        if libraries != ort_names:
            fail(errors, f"ONNX Runtime native pair is incomplete for {abi}")
    if ort_by_abi and set(ort_by_abi) != EXPECTED_NATIVE_ABIS:
        fail(errors, "ONNX Runtime native libraries must cover exactly the four application ABIs")
    if signed_model_packaged and (
        set(ort_by_abi) != EXPECTED_NATIVE_ABIS
        or any(libraries != ort_names for libraries in ort_by_abi.values())
    ):
        fail(errors, "core APK contains a signed swipe model without the complete ONNX Runtime")


def apk_checks(errors: list[str], apk: pathlib.Path) -> None:
    if not apk.is_file():
        fail(errors, f"APK does not exist: {apk}")
        return

    apkanalyzer = newest_tool("apkanalyzer", "cmdline-tools/*/bin/apkanalyzer")
    if apkanalyzer is None:
        fail(errors, "apkanalyzer not found in the configured Android SDK")
    else:
        result = run([str(apkanalyzer), "manifest", "print", str(apk)])
        if result.returncode != 0:
            fail(errors, f"apkanalyzer failed: {result.stderr.strip()}")
        else:
            try:
                manifest = ET.fromstring(result.stdout)
            except ET.ParseError as exc:
                fail(errors, f"cannot parse merged APK manifest: {exc}")
            else:
                package_name = manifest.attrib.get("package")
                if package_name not in EXPECTED_APPLICATION_IDS:
                    fail(errors, f"unexpected APK application ID: {package_name}")

                uses_sdk = manifest.find("uses-sdk")
                if uses_sdk is None:
                    fail(errors, "merged APK manifest has no uses-sdk element")
                else:
                    if android_attribute(uses_sdk, "minSdkVersion") != EXPECTED_MIN_SDK:
                        fail(errors, f"APK minSdkVersion must be {EXPECTED_MIN_SDK}")
                    if android_attribute(uses_sdk, "targetSdkVersion") != EXPECTED_TARGET_SDK:
                        fail(errors, f"APK targetSdkVersion must be {EXPECTED_TARGET_SDK}")

                permissions = {
                    android_attribute(node, "name")
                    for node in manifest.findall("uses-permission")
                }
                forbidden = sorted(FORBIDDEN_PERMISSIONS & permissions)
                if forbidden:
                    fail(errors, f"APK contains forbidden permissions: {', '.join(forbidden)}")

                application = manifest.find("application")
                if application is None:
                    fail(errors, "merged APK manifest has no application element")
                else:
                    required_attributes = {
                        "allowBackup": "false",
                        "usesCleartextTraffic": "false",
                        "directBootAware": "true",
                    }
                    for name, expected in required_attributes.items():
                        if android_attribute(application, name) != expected:
                            fail(errors, f"merged application android:{name} must be {expected}")
                    for name in ("fullBackupContent", "dataExtractionRules"):
                        if not android_attribute(application, name):
                            fail(errors, f"merged application must retain android:{name}")

                    latin_ime = next(
                        (service for service in application.findall("service")
                         if android_attribute(service, "name") == LATIN_IME_CLASS),
                        None,
                    )
                    if latin_ime is None:
                        fail(errors, "merged APK has no LibreBoard input-method service")
                    else:
                        if android_attribute(latin_ime, "permission") != "android.permission.BIND_INPUT_METHOD":
                            fail(errors, "merged input-method service must require BIND_INPUT_METHOD")
                        if android_attribute(latin_ime, "directBootAware") != "true":
                            fail(errors, "merged input-method service must remain Direct Boot aware")

                    clipboard_provider = next(
                        (provider for provider in application.findall("provider")
                         if android_attribute(provider, "name") == CLIPBOARD_PROVIDER_CLASS),
                        None,
                    )
                    if clipboard_provider is None:
                        fail(errors, "merged APK has no clipboard content provider")
                    elif android_attribute(clipboard_provider, "exported") != "false":
                        fail(errors, "merged clipboard content provider must not be exported")

                queries = manifest.find("queries")
                queried_model_providers = [] if queries is None else [
                    provider for provider in queries.findall("provider")
                    if android_attribute(provider, "authorities") == MODEL_PACK_AUTHORITY
                ]
                if len(queried_model_providers) != 1:
                    fail(errors, "merged core APK must query exactly the official model-pack provider")

    zipalign = newest_tool("zipalign", "build-tools/*/zipalign")
    if zipalign is None:
        fail(errors, "zipalign not found in the configured Android SDK")
    else:
        result = run([str(zipalign), "-c", "-P", "16", "4", str(apk)])
        if result.returncode != 0:
            fail(errors, "APK fails 16 KiB ZIP alignment")

    readobj = newest_tool("llvm-readobj", "ndk/*/toolchains/llvm/prebuilt/*/bin/llvm-readobj")
    with zipfile.ZipFile(apk) as archive, tempfile.TemporaryDirectory(prefix="libreboard-native-") as temp:
        signed_model_assets = {
            name for name in archive.namelist()
            if name in {CORE_SWIPE_MODEL_ASSET, CORE_MODEL_PUBLIC_KEY_ASSET}
        }
        if signed_model_assets and signed_model_assets != {
            CORE_SWIPE_MODEL_ASSET,
            CORE_MODEL_PUBLIC_KEY_ASSET,
        }:
            fail(errors, "core APK must package the signed swipe model and project key together")
        elif signed_model_assets:
            public_key_entry = archive.getinfo(CORE_MODEL_PUBLIC_KEY_ASSET)
            if public_key_entry.file_size <= 0 or public_key_entry.file_size > 8 * 1024:
                fail(errors, "core APK model public key is empty or oversized")
            else:
                public_key_path = pathlib.Path(temp) / "libreboard-model-signing-public.der"
                public_key_path.write_bytes(archive.read(CORE_MODEL_PUBLIC_KEY_ASSET))
                model_archive_checks(
                    errors,
                    archive.read(CORE_SWIPE_MODEL_ASSET),
                    "core swipe model archive",
                    public_key_path,
                    "swipe-latin-v1",
                )
        for dex_entry in (name for name in archive.namelist() if name.endswith(".dex")):
            dex = archive.read(dex_entry).lower()
            for marker in FORBIDDEN_DEPENDENCY_MARKERS:
                dotted = marker.encode("ascii")
                slashed = marker.replace(".", "/").encode("ascii")
                if dotted in dex or slashed in dex:
                    fail(errors, f"APK DEX contains forbidden dependency marker {marker}: {dex_entry}")

        native_entries = [name for name in archive.namelist() if name.endswith(".so")]
        onnx_runtime_apk_entry_checks(errors, native_entries, bool(signed_model_assets))
        for entry in native_entries:
            name = pathlib.PurePosixPath(entry).name
            if name not in ALLOWED_NATIVE_LIBRARIES:
                fail(errors, f"APK contains unapproved native library: {entry}")
                continue
            if readobj is None:
                fail(errors, "llvm-readobj not found; cannot verify native page alignment")
                break
            output = pathlib.Path(temp) / name
            output.write_bytes(archive.read(entry))
            result = run([str(readobj), "--program-headers", str(output)])
            if result.returncode != 0:
                fail(errors, f"llvm-readobj failed for {entry}: {result.stderr.strip()}")
                continue
            alignments = [int(value) for value in re.findall(
                r"Type:\s+PT_LOAD(?:(?!Type:).)*?Alignment:\s+(\d+)", result.stdout, re.DOTALL
            )]
            if not alignments or any(value < 16_384 for value in alignments):
                fail(errors, f"native library lacks 16 KiB ELF alignment: {entry}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="store_true", help="verify source privacy and dependency rules")
    parser.add_argument("--apk", action="append", type=pathlib.Path, default=[], help="verify a built APK")
    parser.add_argument(
        "--model-pack-apk",
        action="append",
        type=pathlib.Path,
        default=[],
        help="verify the separately built English/German context-model APK",
    )
    parser.add_argument(
        "--model-public-key",
        type=pathlib.Path,
        help="X.509 DER RSA public key trusted for signed model archives",
    )
    parser.add_argument("--rebuilt-apk", type=pathlib.Path, help="independent clean rebuild of the release APK")
    parser.add_argument("--phase0-report", type=pathlib.Path, help="passing Phase 0 JSON report for the APK")
    parser.add_argument(
        "--phase0-measurements",
        type=pathlib.Path,
        help="raw session-separated Phase 0 measurement JSONL",
    )
    parser.add_argument("--grapheneos-evidence", type=pathlib.Path, help="physical GrapheneOS JSON evidence")
    parser.add_argument(
        "--instrumentation-output",
        type=pathlib.Path,
        help="raw output from the GrapheneOS device run",
    )
    args = parser.parse_args()
    evidence_values = (
        args.rebuilt_apk,
        args.phase0_report,
        args.phase0_measurements,
        args.grapheneos_evidence,
        args.instrumentation_output,
    )
    has_evidence = any(value is not None for value in evidence_values)
    if not args.source and not args.apk and not args.model_pack_apk and not has_evidence:
        parser.error("select --source, --apk, and/or --model-pack-apk")
    if has_evidence and (len(args.apk) != 1 or any(value is None for value in evidence_values)):
        parser.error(
            "GrapheneOS evidence requires exactly one --apk plus --rebuilt-apk, "
            "--phase0-report, --phase0-measurements, --grapheneos-evidence, and "
            "--instrumentation-output"
        )
    if args.model_pack_apk and args.model_public_key is None:
        parser.error("--model-pack-apk requires --model-public-key")

    errors: list[str] = []
    if args.source:
        source_checks(errors)
    for apk in args.apk:
        apk_checks(errors, apk.resolve())
    for model_pack_apk in args.model_pack_apk:
        model_pack_apk_checks(errors, model_pack_apk.resolve(), args.model_public_key.resolve())
    if has_evidence:
        evidence_checks(
            errors,
            args.apk[0].resolve(),
            args.rebuilt_apk.resolve(),
            args.phase0_report.resolve(),
            args.phase0_measurements.resolve(),
            args.grapheneos_evidence.resolve(),
            args.instrumentation_output.resolve(),
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("LibreBoard release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
