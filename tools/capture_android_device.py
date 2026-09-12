#!/usr/bin/env python3
"""Capture bounded, read-only Android device prerequisites for a GrapheneOS release run.

This tool deliberately does not issue a PASS record. GrapheneOS identity, exploit-protection mode,
and the behavioral acceptance matrix still require operator verification on the physical device.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from typing import Any


SCHEMA_VERSION = 1
MAXIMUM_COMMAND_OUTPUT_BYTES = 1024 * 1024
SERIAL = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
PACKAGE_NAME = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
SECURITY_PATCH = re.compile(r"^20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])$")
MEM_TOTAL = re.compile(r"^MemTotal:\s+([1-9][0-9]*)\s+kB$", re.MULTILINE)
PLAY_PACKAGES = frozenset({
    "com.android.vending",
    "com.google.android.gms",
    "com.google.android.gsf",
})
LIBREBOARD_PACKAGE = "org.libreboard.keyboard"
MANUAL_CHECKS = (
    "Confirm Settings identifies the current stable GrapheneOS release.",
    "Confirm LibreBoard exploit protection compatibility mode is disabled.",
    "Run the complete physical-device behavior matrix in docs/grapheneos.md.",
    "Bind raw instrumentation, Phase 0, model, APK, and reproducible-build hashes.",
)


class DeviceCaptureError(ValueError):
    """The connected-device inventory could not be captured safely."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _command(
    adb: pathlib.Path,
    arguments: Sequence[str],
    *,
    serial: str | None,
    runner: Runner,
) -> str:
    command = [str(adb)]
    if serial is not None:
        command.extend(("-s", serial))
    command.extend(arguments)
    environment = os.environ.copy()
    environment.update({"LANG": "C", "LC_ALL": "C"})
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as failure:
        raise DeviceCaptureError(f"adb command could not run: {failure}") from failure
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if len(stdout.encode("utf-8")) > MAXIMUM_COMMAND_OUTPUT_BYTES:
        raise DeviceCaptureError("adb command output exceeded the bounded size")
    if result.returncode != 0:
        detail = stderr.strip()[:512] or f"exit {result.returncode}"
        raise DeviceCaptureError(f"adb command failed: {detail}")
    return stdout.strip()


def _connected_devices(output: str) -> dict[str, str]:
    devices: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("List of devices attached") or line.startswith("*"):
            continue
        fields = line.split()
        if len(fields) < 2 or not SERIAL.fullmatch(fields[0]):
            raise DeviceCaptureError("adb returned an invalid device inventory")
        if fields[0] in devices:
            raise DeviceCaptureError("adb returned a duplicate device serial")
        devices[fields[0]] = fields[1]
    return devices


def _select_device(devices: dict[str, str], requested: str | None) -> str:
    if requested is not None:
        if not SERIAL.fullmatch(requested):
            raise DeviceCaptureError("requested device serial is invalid")
        if requested not in devices:
            raise DeviceCaptureError("requested device is not connected")
        serial = requested
    else:
        ready = sorted(serial for serial, state in devices.items() if state == "device")
        if len(ready) != 1:
            raise DeviceCaptureError("connect exactly one ready device or pass --serial")
        serial = ready[0]
    if devices[serial] != "device":
        raise DeviceCaptureError(f"selected device is not ready: {devices[serial]}")
    return serial


def _parse_packages(output: str) -> set[str]:
    packages = set()
    for line in output.splitlines():
        if not line.startswith("package:"):
            raise DeviceCaptureError("adb returned an invalid package inventory")
        package = line.removeprefix("package:").strip()
        # Android's own framework package is the valid single-segment name `android`.
        # Overlay packages may also contain underscores, including trailing underscores.
        if not PACKAGE_NAME.fullmatch(package):
            raise DeviceCaptureError("adb returned an invalid package name")
        packages.add(package)
    return packages


def inspect_device(
    adb: pathlib.Path,
    requested_serial: str | None = None,
    *,
    device_user: int | None = None,
    runner: Runner = subprocess.run,
    now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(datetime.UTC),
) -> dict[str, Any]:
    devices = _connected_devices(_command(adb, ("devices", "-l"), serial=None, runner=runner))
    serial = _select_device(devices, requested_serial)

    def shell(*arguments: str) -> str:
        return _command(adb, ("shell", *arguments), serial=serial, runner=runner)

    properties = {
        "deviceModel": "ro.product.model",
        "manufacturer": "ro.product.manufacturer",
        "apiLevel": "ro.build.version.sdk",
        "securityPatchLevel": "ro.build.version.security_patch",
        "buildFingerprint": "ro.build.fingerprint",
        "osBuildIncremental": "ro.build.version.incremental",
        "kernelQemu": "ro.kernel.qemu",
        "bootQemu": "ro.boot.qemu",
        "lowRamFlag": "ro.config.low_ram",
        "debuggableFlag": "ro.debuggable",
        "forcedLowRamFlag": "debug.force_low_ram",
    }
    values = {name: shell("getprop", prop) for name, prop in properties.items()}
    if any(not values[name] for name in (
        "deviceModel", "manufacturer", "apiLevel", "securityPatchLevel",
        "buildFingerprint", "osBuildIncremental",
    )):
        raise DeviceCaptureError("device identity properties are incomplete")
    try:
        api_level = int(values["apiLevel"])
    except ValueError as failure:
        raise DeviceCaptureError("device API level is invalid") from failure
    if values["lowRamFlag"] not in {"", "false", "true"}:
        raise DeviceCaptureError("device low-RAM property is invalid")

    if values["debuggableFlag"] not in {"", "0", "1"} or values["forcedLowRamFlag"] not in {"", "false", "true"}:
        raise DeviceCaptureError("device debug low-RAM properties are invalid")
    forced_low_ram = values["debuggableFlag"] == "1" and values["forcedLowRamFlag"] == "true"

    memory_match = MEM_TOTAL.search(shell("cat", "/proc/meminfo"))
    if memory_match is None:
        raise DeviceCaptureError("device memory total is unavailable")
    memory_mib = (int(memory_match.group(1)) + 1023) // 1024

    if device_user is not None:
        if not 0 <= device_user <= 9999:
            raise DeviceCaptureError("requested device-user id is invalid")
        current_user = str(device_user)
    else:
        current_user = shell("am", "get-current-user")
    if not current_user.isdigit() or int(current_user) > 9999:
        raise DeviceCaptureError("device current-user id is invalid")
    packages = _parse_packages(shell("pm", "list", "packages", "--user", current_user))
    installed_play = sorted(PLAY_PACKAGES & packages)
    physical = values["kernelQemu"] not in {"1", "true"} and values["bootQemu"] not in {"1", "true"}

    blockers = []
    if not physical:
        blockers.append("selected target reports an emulator/qemu environment")
    if api_level < 35:
        blockers.append("selected target is older than Android 15/API 35")
    if not SECURITY_PATCH.fullmatch(values["securityPatchLevel"]):
        blockers.append("selected target has an invalid security patch level")
    if installed_play:
        blockers.append("sandboxed Google Play packages are installed for the inspected user")
    if LIBREBOARD_PACKAGE not in packages:
        blockers.append("LibreBoard is not installed for the inspected user")

    captured = now().astimezone(datetime.UTC).replace(microsecond=0)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "READY_FOR_MANUAL_MATRIX" if not blockers else "BLOCKED",
        "capturedAtUtc": captured.isoformat().replace("+00:00", "Z"),
        "serial": serial,
        "currentUserId": int(current_user),
        "deviceModel": values["deviceModel"],
        "manufacturer": values["manufacturer"],
        "apiLevel": api_level,
        "securityPatchLevel": values["securityPatchLevel"],
        "buildFingerprint": values["buildFingerprint"],
        "osBuildIncremental": values["osBuildIncremental"],
        "physicalDevice": physical,
        "isLowRamDevice": values["lowRamFlag"] == "true" or forced_low_ram,
        "lowRamConfiguration": {
            "ro.config.low_ram": values["lowRamFlag"],
            "ro.debuggable": values["debuggableFlag"],
            "debug.force_low_ram": values["forcedLowRamFlag"],
            "mode": "product" if values["lowRamFlag"] == "true" else "debug-forced" if forced_low_ram else "normal",
            "requiresAppApiVerification": True,
        },
        "memoryMiB": memory_mib,
        "libreBoardInstalled": LIBREBOARD_PACKAGE in packages,
        "sandboxedGooglePlayInstalled": bool(installed_play),
        "installedGooglePlayPackages": installed_play,
        "blockers": blockers,
        "manualChecksRequired": list(MANUAL_CHECKS),
        "isReleaseEvidence": False,
    }


def write_report(path: pathlib.Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    try:
        with temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", type=pathlib.Path, help="path to the Android Debug Bridge")
    parser.add_argument("--serial", help="specific adb device serial")
    parser.add_argument("--device-user", type=int,
                        help="inspect this Android user id instead of the current foreground user")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    adb_value = str(args.adb) if args.adb is not None else shutil.which("adb")
    if adb_value is None:
        print("device capture error: adb was not found; pass --adb", file=sys.stderr)
        return 1
    adb = pathlib.Path(adb_value).resolve()
    if not adb.is_file() or not os.access(adb, os.X_OK):
        print("device capture error: adb is not an executable file", file=sys.stderr)
        return 1
    try:
        report = inspect_device(adb, args.serial, device_user=args.device_user)
        write_report(args.output, report)
    except DeviceCaptureError as failure:
        print(f"device capture error: {failure}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
