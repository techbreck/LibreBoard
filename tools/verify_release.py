#!/usr/bin/env python3
"""Fail-closed source/APK checks shared by F-Droid and GrapheneOS release gates."""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = pathlib.Path(__file__).resolve().parents[1]
ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
EXPECTED_APPLICATION_IDS = {
    "org.libreboard.keyboard",
    "org.libreboard.keyboard.debug",
}
EXPECTED_MIN_SDK = "26"
EXPECTED_TARGET_SDK = "36"
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


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def android_attribute(node: ET.Element, name: str) -> str | None:
    return node.attrib.get(ANDROID_NS + name)


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

    validate_backup_exclusions(errors)

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
    if not args.source and not args.apk and not has_evidence:
        parser.error("select --source and/or at least one --apk")
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
