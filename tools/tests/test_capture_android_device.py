# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import datetime
import pathlib
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import capture_android_device as capture  # noqa: E402


class FakeAdb:
    def __init__(self, *, devices: str = "List of devices attached\npixel-1 device product:akita\n"):
        self.devices = devices
        self.properties = {
            "ro.product.model": "Pixel 8a",
            "ro.product.manufacturer": "Google",
            "ro.build.version.sdk": "36",
            "ro.build.version.security_patch": "2026-09-05",
            "ro.build.fingerprint": "google/akita/akita:16/build/123:user/release-keys",
            "ro.build.version.incremental": "2026090500",
            "ro.kernel.qemu": "0",
            "ro.boot.qemu": "0",
        }
        self.packages = {capture.LIBREBOARD_PACKAGE}

    def __call__(self, command, **_kwargs):
        arguments = command[1:]
        if arguments == ["devices", "-l"]:
            output = self.devices
        else:
            self.assert_serial(arguments)
            operation = arguments[2:]
            if operation[:2] == ["shell", "getprop"]:
                output = self.properties[operation[2]] + "\n"
            elif operation == ["shell", "am", "get-current-user"]:
                output = "0\n"
            elif operation == ["shell", "pm", "list", "packages", "--user", "0"]:
                output = "".join(f"package:{value}\n" for value in sorted(self.packages))
            else:
                raise AssertionError(f"unexpected adb command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    @staticmethod
    def assert_serial(arguments):
        if arguments[:2] != ["-s", "pixel-1"]:
            raise AssertionError(f"missing selected serial: {arguments}")


class CaptureAndroidDeviceTest(unittest.TestCase):
    def inspect(self, fake: FakeAdb):
        return capture.inspect_device(
            pathlib.Path("/fixture/adb"),
            runner=fake,
            now=lambda: datetime.datetime(2026, 9, 6, 4, 0, tzinfo=datetime.UTC),
        )

    def test_physical_play_free_device_is_ready_only_for_manual_matrix(self):
        fake = FakeAdb()
        fake.packages.add("android")

        report = self.inspect(fake)

        self.assertEqual("READY_FOR_MANUAL_MATRIX", report["status"])
        self.assertEqual("2026-09-06T04:00:00Z", report["capturedAtUtc"])
        self.assertEqual("Pixel 8a", report["deviceModel"])
        self.assertEqual(36, report["apiLevel"])
        self.assertTrue(report["physicalDevice"])
        self.assertTrue(report["libreBoardInstalled"])
        self.assertFalse(report["sandboxedGooglePlayInstalled"])
        self.assertFalse(report["isReleaseEvidence"])
        self.assertEqual([], report["blockers"])
        self.assertEqual(4, len(report["manualChecksRequired"]))

    def test_package_inventory_accepts_android_and_rejects_unsafe_names(self):
        self.assertEqual({"android", "org.libreboard.keyboard"}, capture._parse_packages(
            "package:android\npackage:org.libreboard.keyboard\n",
        ))
        for invalid in ("package:", "package:bad-name", "package:.leading", "package:trailing."):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(capture.DeviceCaptureError, "invalid package name"):
                    capture._parse_packages(invalid)

    def test_emulator_play_or_missing_keyboard_remains_blocked(self):
        fake = FakeAdb()
        fake.properties["ro.kernel.qemu"] = "1"
        fake.packages = set(capture.PLAY_PACKAGES)

        report = self.inspect(fake)

        self.assertEqual("BLOCKED", report["status"])
        self.assertFalse(report["physicalDevice"])
        self.assertFalse(report["libreBoardInstalled"])
        self.assertTrue(report["sandboxedGooglePlayInstalled"])
        self.assertEqual(sorted(capture.PLAY_PACKAGES), report["installedGooglePlayPackages"])
        self.assertEqual(3, len(report["blockers"]))

    def test_multiple_or_unready_devices_require_an_explicit_ready_serial(self):
        multiple = FakeAdb(devices=(
            "List of devices attached\n"
            "pixel-1 device product:akita\n"
            "pixel-2 device product:husky\n"
        ))
        with self.assertRaisesRegex(capture.DeviceCaptureError, "exactly one ready device"):
            self.inspect(multiple)

        offline = FakeAdb(devices="List of devices attached\npixel-1 unauthorized\n")
        with self.assertRaisesRegex(capture.DeviceCaptureError, "exactly one ready device"):
            self.inspect(offline)
        with self.assertRaisesRegex(capture.DeviceCaptureError, "not ready"):
            capture.inspect_device(pathlib.Path("/fixture/adb"), "pixel-1", runner=offline)

    def test_malformed_device_output_and_identity_fail_closed(self):
        with self.assertRaisesRegex(capture.DeviceCaptureError, "invalid device inventory"):
            self.inspect(FakeAdb(devices="List of devices attached\nbad/serial device\n"))

        fake = FakeAdb()
        fake.properties["ro.build.version.sdk"] = "latest"
        with self.assertRaisesRegex(capture.DeviceCaptureError, "API level is invalid"):
            self.inspect(fake)

        fake = FakeAdb()
        fake.packages.add("not a package")
        with self.assertRaisesRegex(capture.DeviceCaptureError, "invalid package name"):
            self.inspect(fake)

    def test_report_write_is_canonical_and_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "nested" / "device.json"
            path.parent.mkdir()
            path.write_text("old", encoding="utf-8")
            report = self.inspect(FakeAdb())

            capture.write_report(path, report)

            payload = path.read_text(encoding="utf-8")
            self.assertTrue(payload.endswith("\n"))
            self.assertIn('"isReleaseEvidence": false', payload)
            self.assertFalse(any(path.parent.glob(".device.json.*")))


if __name__ == "__main__":
    unittest.main()
