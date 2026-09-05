#!/usr/bin/env python3
"""Fail-closed source/APK checks shared by F-Droid and GrapheneOS release gates."""

from __future__ import annotations

import argparse
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
    # Reserved for the pinned, reduced-operator source build after its model gate passes.
    "liblibreboard_onnxruntime.so",
}
BACKUP_DOMAINS = {"root", "file", "database", "sharedpref", "external"}


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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


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
    args = parser.parse_args()
    if not args.source and not args.apk:
        parser.error("select --source and/or at least one --apk")

    errors: list[str] = []
    if args.source:
        source_checks(errors)
    for apk in args.apk:
        apk_checks(errors, apk.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("LibreBoard release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
