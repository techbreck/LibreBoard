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
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify_release import (  # noqa: E402
    GRAPHENEOS_CHECKS,
    evidence_checks,
    model_archive_checks,
    validate_context_distillation_manifest,
    validate_hash_locked_requirements,
)
import model_sources  # noqa: E402
import prepare_context_dataset  # noqa: E402
import score_context_teacher  # noqa: E402
import evaluate_engine  # noqa: E402
import verify_release  # noqa: E402


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


class VerifyDependencyLockTest(unittest.TestCase):
    def test_process_exhaustion_is_a_failed_command_not_a_traceback(self):
        with mock.patch.object(
            verify_release.subprocess,
            "run",
            side_effect=BlockingIOError(35, "Resource temporarily unavailable"),
        ):
            result = verify_release.run(["fixture"])
        self.assertEqual(126, result.returncode)
        self.assertIn("Resource temporarily unavailable", result.stderr)

    def test_accepts_hash_locked_direct_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "requirements.lock"
            path.write_text(
                "first==1.0 \\\n"
                "    --hash=sha256:" + "a" * 64 + "\n"
                "second==2.0 \\\n"
                "    --hash=sha256:" + "b" * 64 + "\n",
                encoding="utf-8",
            )
            errors = []
            validate_hash_locked_requirements(errors, path, "fixture", ("first==1.0", "second==2.0"))
            self.assertEqual([], errors)

    def test_rejects_missing_unhashed_and_insecure_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "requirements.lock"
            path.write_text("first==1.0\n# http://example.invalid/simple\n", encoding="utf-8")
            errors = []
            validate_hash_locked_requirements(errors, path, "fixture", ("first==1.0", "second==2.0"))
            self.assertEqual(
                [
                    "fixture lock does not contain the audited direct dependencies",
                    "fixture lock contains unhashed packages: first",
                    "fixture lock contains an insecure package source",
                ],
                errors,
            )


class VerifyContextDistillationManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = pathlib.Path(__file__).resolve().parents[2]
        cls.manifest = json.loads(
            (root / "models/context/distillation-manifest.json").read_text(encoding="utf-8")
        )
        cls.context_corpus = json.loads(
            prepare_context_dataset.DEFAULT_CORPUS_MANIFEST.read_text(encoding="utf-8")
        )
        cls.policy = score_context_teacher.load_policy()
        cls.teacher_source = model_sources.load_manifest().source(cls.policy.teacher_source_id)

    def validate(self, manifest):
        errors = []
        validate_context_distillation_manifest(
            errors,
            manifest,
            context_corpus=self.context_corpus,
            policy=self.policy,
            teacher_source=self.teacher_source,
        )
        return errors

    def test_accepts_committed_release_sized_result(self):
        self.assertEqual([], self.validate(json.loads(json.dumps(self.manifest))))

    def test_rejects_unreconciled_counts_and_teacher_provenance(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["generationCounts"]["rejection:train:no_candidates"] = 1
        manifest["teacher"]["modelSha256"] = "a" * 64
        errors = self.validate(manifest)
        self.assertIn("context distillation generation counts do not exactly reconcile", errors)
        self.assertIn("context distillation manifest has unexpected teacher provenance", errors)

    def test_rejects_invalid_output_without_crashing(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["outputs"]["test.scored.jsonl"]["languages"] = ["en-US", "de"]
        self.assertIn(
            "context distillation output metadata is invalid: test.scored.jsonl",
            self.validate(manifest),
        )

    def test_rejects_inconsistent_teacher_metrics(self):
        manifest = json.loads(json.dumps(self.manifest))
        manifest["teacherMetrics"]["validation"]["observedTop1Rate"] = 0.5
        self.assertIn(
            "context distillation teacher metrics do not reconcile: validation",
            self.validate(manifest),
        )


class VerifyReleaseEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        documents = []

        def add(number, category, target, raw, predictions, *, strata=None, should_correct=None):
            document = {
                "schemaVersion": 1,
                "id": f"release-example-{number}",
                "sessionId": f"release-session-{number}",
                "split": "test",
                "category": category,
                "target": target,
                "raw": raw,
                "predictions": predictions,
                "latencyMs": {system: 20.0 for system in predictions},
                "strata": strata or [],
            }
            if should_correct is not None:
                document["shouldCorrect"] = should_correct
            documents.append(document)

        number = 0
        for category, count in (("tap_error", 3_000), ("spacing", 500), ("lexical", 500)):
            for index in range(count):
                number += 1
                raw = f"raw-{category}-{index}"
                target = f"target-{category}-{index}"
                add(number, category, target, raw, {
                    "heliboard": [raw],
                    "fused": [target, raw],
                    "fused_personal": [target, raw],
                    "fused_neural": [target, raw],
                })
        for index in range(1_000):
            number += 1
            should_correct = index < 500
            raw = f"their-{index}" if should_correct else f"word-{index}"
            target = f"there-{index}" if should_correct else raw
            add(number, "valid_word", target, raw, {
                "heliboard": [raw],
                "fused": [raw],
                "fused_personal": [raw],
                "fused_neural": [target, raw] if should_correct else [raw],
            }, should_correct=should_correct)
        for index in range(5_000):
            number += 1
            length = "short" if index < 1_667 else "medium" if index < 3_334 else "long"
            quality = "clean" if 500 <= index < 2_500 else "sloppy"
            strata = [length, quality]
            if index < 500:
                strata.extend(("very_sloppy", "double_letter"))
            if 500 <= index < 1_000:
                strata.append("return_trip")
            target = f"swipe-{index}"
            add(number, "swipe", target, "", {
                "geometric": [f"wrong-{index}"],
                "ctc": [target],
                "fused_swipe": [target],
            }, strata=strata)

        cls.measurement_payload = b"".join(
            (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for document in documents
        )
        cls.measurement_examples = [
            evaluate_engine.parse_example(document, index)
            for index, document in enumerate(documents, 1)
        ]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="libreboard-release-evidence-")
        self.root = pathlib.Path(self.temporary.name)
        self.apk = self.root / "LibreBoard_release.apk"
        self.rebuilt = self.root / "rebuilt.apk"
        self.apk.write_bytes(b"reproducible apk")
        self.rebuilt.write_bytes(self.apk.read_bytes())
        self.apk_hash = sha256(self.apk.read_bytes())
        self.phase0_path = self.root / "phase0.json"
        self.phase0_measurements_path = self.root / "phase0-measurements.jsonl"
        self.phase0_measurements_path.write_bytes(self.measurement_payload)
        self.grapheneos_path = self.root / "grapheneos.json"
        self.instrumentation_path = self.root / "instrumentation.txt"
        self.instrumentation_path.write_bytes(b"passing device instrumentation")
        self.phase0_result = evaluate_engine.evaluate(
            self.measurement_examples,
            self.phase0_metadata(),
            measurement_sha256=sha256(self.measurement_payload),
            enforce_minimum_counts=True,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def phase0(self):
        return json.loads(json.dumps(self.phase0_result))

    def phase0_metadata(self):
        return {
            "schemaVersion": 1,
            "appCommit": "a" * 40,
            "coreApkSha256": self.apk_hash,
            "swipeModelSha256": "b" * 64,
            "contextModelSha256": "c" * 64,
            "peakAddedNeuralMemoryMiB": 40,
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
            self.phase0_measurements_path,
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

    def test_rejects_different_phase0_measurement_dataset(self):
        self.write_reports()
        self.phase0_measurements_path.write_bytes(self.measurement_payload + b"\n")
        self.assertIn("Phase 0 report references a different measurement dataset", self.verify())


if __name__ == "__main__":
    unittest.main()
