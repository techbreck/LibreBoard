# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
import zipfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify_release import (  # noqa: E402
    GRAPHENEOS_CHECKS,
    PHASE0_CHECKS,
    evidence_checks,
    model_archive_checks,
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def context_model_archive(**manifest_overrides) -> bytes:
    model = b"fixture onnx"
    tokenizer = b'{"fixture":true}'
    manifest = {
        "schemaVersion": 1,
        "engineAbi": 1,
        "modelKind": "context-rescorer",
        "tensorAbi": "context-en-de-v1",
        "locales": ["en-US", "de"],
        "architecture": "fixture",
        "parameterCount": 1,
        "quantization": "fixture",
        "modelSha256": sha256(model),
        "tokenizerSha256": sha256(tokenizer),
        "requiredOnnxOperators": ["MatMul"],
        "license": "Apache-2.0",
        "provenance": [{
            "name": "fixture",
            "revision": "1",
            "license": "Apache-2.0",
            "source_url": "https://example.invalid/model",
        }],
        "minimumAppVersionCode": 1,
    }
    manifest.update(manifest_overrides)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        archive.writestr("model.onnx", model)
        archive.writestr("tokenizer.json", tokenizer)
        archive.writestr("signature.der", b"fixture signature")
    return output.getvalue()


class VerifyModelArchiveTest(unittest.TestCase):
    def test_accepts_bounded_context_pack_contract(self):
        errors = []
        model_archive_checks(errors, context_model_archive())
        self.assertEqual([], errors)

    def test_rejects_wrong_model_kind_and_payload_hash(self):
        errors = []
        model_archive_checks(
            errors,
            context_model_archive(modelKind="swipe-ctc", modelSha256="a" * 64),
        )
        self.assertIn("model archive is not the official context-en-de tensor contract", errors)
        self.assertIn("model archive model hash does not match its payload", errors)

    def test_rejects_extra_archive_entry(self):
        original = context_model_archive()
        input_archive = zipfile.ZipFile(io.BytesIO(original))
        output = io.BytesIO()
        with input_archive, zipfile.ZipFile(output, "w") as archive:
            for info in input_archive.infolist():
                archive.writestr(info.filename, input_archive.read(info))
            archive.writestr("unexpected.bin", b"no")
        errors = []
        model_archive_checks(errors, output.getvalue())
        self.assertEqual(
            ["model archive must contain exactly the four approved data entries"],
            errors,
        )

    def test_malformed_operator_and_locale_values_fail_closed(self):
        errors = []
        model_archive_checks(
            errors,
            context_model_archive(
                locales=[{"not": "a locale"}, "de"],
                requiredOnnxOperators=[["not hashable"]],
            ),
        )
        self.assertIn("model archive must contain exactly the en-US and de locales", errors)
        self.assertIn("model archive has an invalid ONNX operator declaration", errors)


class VerifyReleaseEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="libreboard-release-evidence-")
        self.root = pathlib.Path(self.temporary.name)
        self.apk = self.root / "LibreBoard_release.apk"
        self.rebuilt = self.root / "rebuilt.apk"
        self.apk.write_bytes(b"reproducible apk")
        self.rebuilt.write_bytes(self.apk.read_bytes())
        self.apk_hash = sha256(self.apk.read_bytes())
        self.phase0_path = self.root / "phase0.json"
        self.grapheneos_path = self.root / "grapheneos.json"
        self.instrumentation_path = self.root / "instrumentation.txt"
        self.instrumentation_path.write_bytes(b"passing device instrumentation")

    def tearDown(self):
        self.temporary.cleanup()

    def phase0(self):
        return {
            "schemaVersion": 1,
            "passed": True,
            "checks": {name: True for name in PHASE0_CHECKS},
            "counts": {
                "tap_error": 3_000,
                "valid_word": 1_000,
                "spacing": 500,
                "lexical": 500,
                "swipe": 5_000,
            },
            "evidence": {
                "appCommit": "a" * 40,
                "coreApkSha256": self.apk_hash,
                "swipeModelSha256": "b" * 64,
                "contextModelSha256": "c" * 64,
                "environments": [
                    {
                        "kind": "stock_android_hardware",
                        "deviceModel": "Pixel 8a",
                        "buildFingerprint": "stock/fingerprint",
                        "testRunId": "stock-run",
                        "apiLevel": 36,
                        "physicalDevice": True,
                    },
                    {
                        "kind": "grapheneos_hardware",
                        "deviceModel": "Pixel 8a",
                        "grapheneOsBuildNumber": "2026090100",
                        "buildFingerprint": "graphene/fingerprint",
                        "testRunId": "graphene-run",
                        "apiLevel": 36,
                        "physicalDevice": True,
                        "sandboxedGooglePlayInstalled": False,
                    },
                    {
                        "kind": "low_ram_emulator",
                        "deviceModel": "AOSP low RAM",
                        "buildFingerprint": "aosp/fingerprint",
                        "testRunId": "low-ram-run",
                        "apiLevel": 36,
                        "physicalDevice": False,
                        "isLowRamDevice": True,
                        "memoryMiB": 1_024,
                    },
                ],
            },
        }

    def grapheneos(self, phase0_bytes: bytes):
        return {
            "schemaVersion": 1,
            "status": "PASS",
            "appCommit": "a" * 40,
            "apkFilename": self.apk.name,
            "apkSha256": self.apk_hash,
            "swipeModelSha256": "b" * 64,
            "contextModelSha256": "c" * 64,
            "deviceModel": "Pixel 8a",
            "grapheneOsBuildNumber": "2026090100",
            "apiLevel": 36,
            "securityPatchLevel": "2026-09-01",
            "buildFingerprint": "graphene/fingerprint",
            "physicalDevice": True,
            "sandboxedGooglePlayInstalled": False,
            "compatibilityChangesEnabled": False,
            "testerId": "release-test-1",
            "testedAtUtc": "2026-09-05T12:00:00Z",
            "phase0TestRunId": "graphene-run",
            "phase0ReportSha256": sha256(phase0_bytes),
            "instrumentationOutputSha256": sha256(self.instrumentation_path.read_bytes()),
            "checks": {name: True for name in GRAPHENEOS_CHECKS},
            "measurements": {
                "tapLatencyMs": {"p50": 20.0, "p95": 70.0, "p99": 90.0},
                "swipeLatencyMs": {"p50": 80.0, "p95": 180.0, "p99": 210.0},
                "coldStartMs": 300.0,
                "warmStartMs": 80.0,
                "peakRssMiB": 120.0,
                "neuralTimeoutCount": 2,
                "circuitBreakerActivationCount": 1,
            },
        }

    def write_reports(self, phase0=None, grapheneos_mutator=None):
        phase0 = phase0 or self.phase0()
        phase0_bytes = (json.dumps(phase0, sort_keys=True) + "\n").encode()
        self.phase0_path.write_bytes(phase0_bytes)
        grapheneos = self.grapheneos(phase0_bytes)
        if grapheneos_mutator:
            grapheneos_mutator(grapheneos)
        self.grapheneos_path.write_text(json.dumps(grapheneos), encoding="utf-8")

    def verify(self):
        errors = []
        evidence_checks(
            errors,
            self.apk,
            self.rebuilt,
            self.phase0_path,
            self.grapheneos_path,
            self.instrumentation_path,
        )
        return errors

    def test_accepts_matching_reproducible_grapheneos_evidence(self):
        self.write_reports()
        self.assertEqual([], self.verify())

    def test_rejects_non_reproducible_apk(self):
        self.write_reports()
        self.rebuilt.write_bytes(b"different apk")
        self.assertIn("clean rebuild APK is not byte-identical", self.verify())

    def test_rejects_google_play_or_missing_device_check(self):
        def mutate(report):
            report["sandboxedGooglePlayInstalled"] = True
            report["checks"].pop("direct_boot")

        self.write_reports(grapheneos_mutator=mutate)
        errors = self.verify()
        self.assertIn("GrapheneOS evidence must run without sandboxed Google Play", errors)
        self.assertIn("GrapheneOS evidence does not contain every passing device check", errors)

    def test_rejects_phase0_failure_and_artifact_mismatch(self):
        phase0 = self.phase0()
        phase0["passed"] = False
        phase0["evidence"]["coreApkSha256"] = "e" * 64
        self.write_reports(phase0=phase0)
        errors = self.verify()
        self.assertIn("Phase 0 report did not pass", errors)
        self.assertIn("Phase 0 report was produced with a different APK", errors)

    def test_rejects_phase0_report_without_full_environment_matrix(self):
        phase0 = self.phase0()
        phase0["evidence"]["environments"] = phase0["evidence"]["environments"][1:2]
        self.write_reports(phase0=phase0)
        self.assertIn(
            "Phase 0 report does not contain the three reference environments",
            self.verify(),
        )

    def test_rejects_device_identity_mismatch_and_slow_typing(self):
        def mutate(report):
            report["buildFingerprint"] = "different/fingerprint"
            report["measurements"]["tapLatencyMs"]["p95"] = 81.0

        self.write_reports(grapheneos_mutator=mutate)
        errors = self.verify()
        self.assertIn("GrapheneOS tapLatencyMs exceeds the p95 budget", errors)
        self.assertIn("GrapheneOS device evidence disagrees with Phase 0 on buildFingerprint", errors)

    def test_rejects_different_instrumentation_output(self):
        self.write_reports()
        self.instrumentation_path.write_bytes(b"different output")
        self.assertIn("GrapheneOS evidence references different instrumentation output", self.verify())


if __name__ == "__main__":
    unittest.main()
