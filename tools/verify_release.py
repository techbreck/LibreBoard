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

import context_model_contract
import model_sources
import prepare_swipe_dataset
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
MODEL_TOKENIZER_MAXIMUM_BYTES = 2 * 1024 * 1024
MODEL_MANIFEST_MAXIMUM_BYTES = 256 * 1024
MODEL_SIGNATURE_MAXIMUM_BYTES = 16 * 1024
MODEL_ARCHIVE_ENTRIES = {
    "manifest.json",
    "model.onnx",
    "tokenizer.json",
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


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


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

    source_root = ROOT / "app/src/main/java"
    dynamic_load = re.compile(r"System\s*\.\s*load\s*\(")
    load_library = re.compile(r"System\s*\.\s*loadLibrary\s*\(\s*([^)]*)\)")
    for path in source_root.rglob("*"):
        if path.suffix not in {".kt", ".java"}:
            continue
        text = path.read_text(encoding="utf-8")
        if dynamic_load.search(text):
            fail(errors, f"dynamic native path loading is forbidden: {path.relative_to(ROOT)}")
        for match in load_library.finditer(text):
            if "JNI_LIB_NAME" not in match.group(1):
                fail(errors, f"unapproved System.loadLibrary call: {path.relative_to(ROOT)}")

    removed_paths = (
        source_root / "helium314/keyboard/settings/preferences/LoadGestureLibPreference.kt",
        source_root / "helium314/keyboard/settings/screens/gesturedata",
        source_root / "helium314/keyboard/latin/utils/GestureDataGathering.kt",
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

    runtime_settings = read_json_object(
        errors,
        ROOT / "runtime/onnxruntime/build-settings.json",
        "ONNX Runtime build settings",
    )
    if runtime_settings is not None and runtime_settings.get("sourceCommit") != ONNXRUNTIME_COMMIT:
        fail(errors, "ONNX Runtime build settings do not pin the approved commit")
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
        swipe_spec = swipe_model_contract.load_spec()
        context_spec = context_model_contract.load_spec()
        swipe_source = source_manifest.source(swipe_policy.source_id)
        teacher_source = source_manifest.source("hanse2-100m-base-teacher-v1")
    except (
        model_sources.ModelSourceError,
        prepare_swipe_dataset.SwipeDataError,
        swipe_model_contract.SwipeModelContractError,
        context_model_contract.ContextModelContractError,
    ) as exc:
        fail(errors, f"model source/training contract is invalid: {exc}")
    else:
        if swipe_source.repository != "futo-org/swipe.futo.org" or swipe_source.license != "MIT":
            fail(errors, "swipe training must remain pinned to the MIT FUTO gesture dataset")
        if teacher_source.repository != "Evicka/Hanse2-100M-Base" or teacher_source.license != "Apache-2.0":
            fail(errors, "context distillation teacher must remain the Apache-2.0 Hanse2 base model")
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
            "torch==2.8.0+cpu",
            "transformers==4.57.6",
        ),
    )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


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
    if report.get("schemaVersion") != 1:
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
    if report.get("schemaVersion") != 1 or report.get("status") != "PASS":
        fail(errors, "GrapheneOS evidence must use schema 1 with PASS status")
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
    if report.get("compatibilityChangesEnabled") is not False:
        fail(errors, "GrapheneOS evidence must not require compatibility changes")

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


def model_archive_checks(errors: list[str], archive_bytes: bytes, label: str = "model archive") -> None:
    if not archive_bytes or len(archive_bytes) > MODEL_PACK_MAXIMUM_BYTES:
        fail(errors, f"{label} must be non-empty and at most {MODEL_PACK_MAXIMUM_BYTES} bytes")
        return
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (len(entries) != len(MODEL_ARCHIVE_ENTRIES)
                    or len(names) != len(set(names))
                    or set(names) != MODEL_ARCHIVE_ENTRIES
                    or any(entry.is_dir() for entry in entries)):
                fail(errors, f"{label} must contain exactly the four approved data entries")
                return
            sizes = {entry.filename: entry.file_size for entry in entries}
            maximums = {
                "manifest.json": MODEL_MANIFEST_MAXIMUM_BYTES,
                "model.onnx": CONTEXT_MODEL_MAXIMUM_BYTES,
                "tokenizer.json": MODEL_TOKENIZER_MAXIMUM_BYTES,
                "signature.der": MODEL_SIGNATURE_MAXIMUM_BYTES,
            }
            if any(sizes[name] <= 0 or sizes[name] > maximum for name, maximum in maximums.items()):
                fail(errors, f"{label} contains an empty or oversized entry")
                return
            manifest_bytes = archive.read("manifest.json")
            model_bytes = archive.read("model.onnx")
            tokenizer_bytes = archive.read("tokenizer.json")
            archive.read("signature.der")  # CRC-check the signature before accepting the container.
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        fail(errors, f"cannot read {label}: {exc}")
        return

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(errors, f"cannot parse {label} manifest: {exc}")
        return
    if not isinstance(manifest, dict) or set(manifest) != MODEL_MANIFEST_FIELDS:
        fail(errors, f"{label} manifest must contain exactly the supported schema fields")
        return
    if manifest.get("schemaVersion") != 1 or manifest.get("engineAbi") != 1:
        fail(errors, f"{label} has an unsupported schema or engine ABI")
    if (manifest.get("modelKind") != "context-rescorer"
            or manifest.get("tensorAbi") != "context-en-de-v1"):
        fail(errors, f"{label} is not the official context-en-de tensor contract")
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
            or parameter_count <= 0 or parameter_count > 50_000_000):
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
    if not isinstance(tokenizer_hash, str) or not SHA256.fullmatch(tokenizer_hash):
        fail(errors, f"{label} has an invalid tokenizer hash")
    elif hashlib.sha256(tokenizer_bytes).hexdigest() != tokenizer_hash:
        fail(errors, f"{label} tokenizer hash does not match its payload")


def model_pack_apk_checks(errors: list[str], apk: pathlib.Path) -> None:
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
                    model_archive_checks(errors, archive.read(asset), "packaged model archive")
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
        for dex_entry in (name for name in archive.namelist() if name.endswith(".dex")):
            dex = archive.read(dex_entry).lower()
            for marker in FORBIDDEN_DEPENDENCY_MARKERS:
                dotted = marker.encode("ascii")
                slashed = marker.replace(".", "/").encode("ascii")
                if dotted in dex or slashed in dex:
                    fail(errors, f"APK DEX contains forbidden dependency marker {marker}: {dex_entry}")

        native_entries = [name for name in archive.namelist() if name.endswith(".so")]
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
    parser.add_argument("--rebuilt-apk", type=pathlib.Path, help="independent clean rebuild of the release APK")
    parser.add_argument("--phase0-report", type=pathlib.Path, help="passing Phase 0 JSON report for the APK")
    parser.add_argument("--grapheneos-evidence", type=pathlib.Path, help="physical GrapheneOS JSON evidence")
    parser.add_argument(
        "--instrumentation-output",
        type=pathlib.Path,
        help="raw output from the GrapheneOS device run",
    )
    args = parser.parse_args()
    evidence_values = (args.rebuilt_apk, args.phase0_report, args.grapheneos_evidence, args.instrumentation_output)
    has_evidence = any(value is not None for value in evidence_values)
    if not args.source and not args.apk and not args.model_pack_apk and not has_evidence:
        parser.error("select --source, --apk, and/or --model-pack-apk")
    if has_evidence and (len(args.apk) != 1 or any(value is None for value in evidence_values)):
        parser.error(
            "GrapheneOS evidence requires exactly one --apk plus --rebuilt-apk, "
            "--phase0-report, --grapheneos-evidence, and --instrumentation-output"
        )

    errors: list[str] = []
    if args.source:
        source_checks(errors)
    for apk in args.apk:
        apk_checks(errors, apk.resolve())
    for model_pack_apk in args.model_pack_apk:
        model_pack_apk_checks(errors, model_pack_apk.resolve())
    if has_evidence:
        evidence_checks(
            errors,
            args.apk[0].resolve(),
            args.rebuilt_apk.resolve(),
            args.phase0_report.resolve(),
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
