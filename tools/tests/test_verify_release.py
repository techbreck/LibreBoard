# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify_release import (  # noqa: E402
    GRAPHENEOS_CHECKS,
    evidence_checks,
    model_archive_checks,
    onnx_runtime_apk_entry_checks,
    validate_context_distillation_manifest,
    validate_hash_locked_requirements,
    verify_model_signature,
)
import model_sources  # noqa: E402
import prepare_context_dataset  # noqa: E402
import score_context_teacher  # noqa: E402
import evaluate_engine  # noqa: E402
import verify_release  # noqa: E402


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_archive(entries: dict[str, bytes], *, canonical: bool = True) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in sorted(entries):
            if canonical:
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, entries[name])
            else:
                archive.writestr(name, entries[name])
    return output.getvalue()


def context_model_archive(*, canonical: bool = True, **manifest_overrides) -> bytes:
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
    return _model_archive({
        "manifest.json": json.dumps(manifest, separators=(",", ":")).encode(),
        "model.onnx": model,
        "tokenizer.json": tokenizer,
        "signature.der": b"fixture signature",
    }, canonical=canonical)


def swipe_model_archive() -> bytes:
    model = b"fixture swipe onnx"
    manifest = {
        "schemaVersion": 1,
        "engineAbi": 1,
        "modelKind": "swipe-ctc",
        "tensorAbi": "swipe-latin-v1",
        "locales": ["en-US", "de"],
        "architecture": "fixture",
        "parameterCount": 1,
        "quantization": "FP16",
        "modelSha256": sha256(model),
        "requiredOnnxOperators": ["MatMul"],
        "license": "Apache-2.0",
        "provenance": [{
            "name": "fixture",
            "revision": "1",
            "license": "MIT",
            "source_url": "https://example.invalid/swipe",
        }],
        "minimumAppVersionCode": 1,
    }
    return _model_archive({
        "manifest.json": json.dumps(manifest, separators=(",", ":")).encode(),
        "model.onnx": model,
        "signature.der": b"fixture signature",
    })


class VerifyModelArchiveTest(unittest.TestCase):
    def test_accepts_bounded_context_pack_contract(self):
        errors = []
        model_archive_checks(errors, context_model_archive())
        self.assertEqual([], errors)

    def test_accepts_bounded_swipe_pack_contract_without_a_tokenizer(self):
        errors = []
        model_archive_checks(errors, swipe_model_archive(), model_id="swipe-latin-v1")
        self.assertEqual([], errors)

    def test_rejects_wrong_model_kind_and_payload_hash(self):
        errors = []
        model_archive_checks(
            errors,
            context_model_archive(modelKind="swipe-ctc", modelSha256="a" * 64),
        )
        self.assertIn("model archive is not the official context-en-de-v1 tensor contract", errors)
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
            ["model archive must contain exactly the approved data entries"],
            errors,
        )

    def test_rejects_noncanonical_archive_metadata(self):
        errors = []
        model_archive_checks(errors, context_model_archive(canonical=False))
        self.assertEqual(
            ["model archive is not in the canonical deterministic ZIP format"],
            errors,
        )

    def test_rejects_duplicate_manifest_keys(self):
        original = context_model_archive()
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        entries["manifest.json"] = b'{"schemaVersion":1,"schemaVersion":1}'
        errors = []
        model_archive_checks(errors, _model_archive(entries))
        self.assertEqual(
            ["cannot parse model archive manifest: duplicate JSON key: schemaVersion"],
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

    def test_model_signature_requires_strong_rsa_key_and_verified_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            key = pathlib.Path(temporary) / "public.der"
            key.write_bytes(b"public key fixture")
            accepted = subprocess.CompletedProcess(
                ["openssl"], 0,
                "Public-Key: (3072 bit)\nModulus:\n  01\nExponent: 65537 (0x10001)", "",
            )
            verified = subprocess.CompletedProcess(["openssl"], 0, "Verified OK\n", "")
            with mock.patch.object(verify_release, "run", side_effect=[accepted, verified]):
                errors = []
                verify_model_signature(errors, key, b"manifest", b"signature", "fixture")
            self.assertEqual([], errors)

            weak = subprocess.CompletedProcess(
                ["openssl"], 0,
                "Public-Key: (2048 bit)\nModulus:\n  01\nExponent: 65537 (0x10001)", "",
            )
            with mock.patch.object(verify_release, "run", return_value=weak):
                errors = []
                verify_model_signature(errors, key, b"manifest", b"signature", "fixture")
            self.assertEqual(
                ["fixture public key must be a valid RSA key of at least 3072 bits with exponent 65537"],
                errors,
            )

            wrong_exponent = subprocess.CompletedProcess(
                ["openssl"], 0,
                "Public-Key: (3072 bit)\nModulus:\n  01\nExponent: 3 (0x3)", "",
            )
            with mock.patch.object(verify_release, "run", return_value=wrong_exponent):
                errors = []
                verify_model_signature(errors, key, b"manifest", b"signature", "fixture")
            self.assertEqual(
                ["fixture public key must be a valid RSA key of at least 3072 bits with exponent 65537"],
                errors,
            )


class VerifyOnnxRuntimeApkEntriesTest(unittest.TestCase):
    def test_requires_complete_pairs_for_all_application_abis(self):
        entries = [
            f"lib/{abi}/{library}"
            for abi in ("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
            for library in ("libonnxruntime.so", "libonnxruntime4j_jni.so")
        ]
        errors = []
        onnx_runtime_apk_entry_checks(errors, entries, signed_model_packaged=True)
        self.assertEqual([], errors)

        errors = []
        onnx_runtime_apk_entry_checks(errors, entries[:-2], signed_model_packaged=True)
        self.assertIn(
            "ONNX Runtime native libraries must cover exactly the four application ABIs",
            errors,
        )
        self.assertIn(
            "core APK contains a signed swipe model without the complete ONNX Runtime",
            errors,
        )

    def test_signed_model_rejects_absent_or_partial_runtime(self):
        errors = []
        onnx_runtime_apk_entry_checks(errors, [], signed_model_packaged=True)
        self.assertEqual(
            ["core APK contains a signed swipe model without the complete ONNX Runtime"],
            errors,
        )

        errors = []
        onnx_runtime_apk_entry_checks(
            errors,
            ["lib/arm64-v8a/libonnxruntime.so"],
            signed_model_packaged=False,
        )
        self.assertIn("ONNX Runtime native pair is incomplete for arm64-v8a", errors)
        self.assertIn(
            "ONNX Runtime native libraries must cover exactly the four application ABIs",
            errors,
        )


class VerifyStoreMetadataTest(unittest.TestCase):
    def write_locale(self, root: pathlib.Path, locale: str, full_description: str) -> None:
        locale_root = root / locale
        locale_root.mkdir(parents=True)
        (locale_root / "title.txt").write_text("LibreBoard\n", encoding="utf-8")
        (locale_root / "short_description.txt").write_text("Private offline keyboard\n", encoding="utf-8")
        (locale_root / "full_description.txt").write_text(full_description + "\n", encoding="utf-8")

    def test_accepts_reviewed_english_and_german_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            description = (
                "LibreBoard does not request INTERNET or ACCESS_NETWORK_STATE and includes a geometric fallback."
            )
            self.write_locale(root, "en-US", description)
            self.write_locale(root, "de-DE", description)
            errors = []
            verify_release.validate_store_metadata(errors, root)
            self.assertEqual([], errors)

    def test_rejects_stale_locale_and_proprietary_swipe_instructions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            description = (
                "LibreBoard does not request INTERNET or ACCESS_NETWORK_STATE and includes a geometric fallback."
            )
            self.write_locale(root, "en-US", description + " Install swypelibs for swipe typing.")
            self.write_locale(root, "de-DE", description)
            self.write_locale(root, "fr-FR", description)
            errors = []
            verify_release.validate_store_metadata(errors, root)
            self.assertIn(
                "store metadata must contain exactly the reviewed English and German locales",
                errors,
            )
            self.assertIn(
                "store metadata contains obsolete proprietary-swipe guidance: swypelibs",
                errors,
            )

    def test_malformed_metadata_fails_closed_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            description = (
                "LibreBoard does not request INTERNET or ACCESS_NETWORK_STATE and includes a geometric fallback."
            )
            self.write_locale(root, "en-US", description)
            self.write_locale(root, "de-DE", description)
            (root / "de-DE/title.txt").write_bytes(b"\xff")
            errors = []
            verify_release.validate_store_metadata(errors, root)
            self.assertTrue(any(error.startswith("cannot read de-DE store metadata title.txt") for error in errors))


class VerifyGradleDependenciesTest(unittest.TestCase):
    def setUp(self):
        self.source = pathlib.Path(__file__).resolve().parents[2] / "gradle/verification-metadata.xml"

    def test_accepts_committed_checksum_inventory(self):
        errors = []
        verify_release.validate_gradle_dependency_verification(errors, self.source)
        self.assertEqual([], errors)

    def test_rejects_trust_bypass_and_missing_required_component(self):
        tree = ET.parse(self.source)
        root = tree.getroot()
        namespace = verify_release.GRADLE_VERIFICATION_NS
        configuration = root.find(namespace + "configuration")
        components = root.find(namespace + "components")
        ET.SubElement(configuration, namespace + "trusted-artifacts")
        required = next(
            component
            for component in components
            if component.attrib == {
                "group": "junit",
                "name": "junit",
                "version": "4.13.2",
            }
        )
        components.remove(required)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "verification-metadata.xml"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            errors = []
            verify_release.validate_gradle_dependency_verification(errors, path)
        self.assertIn("Gradle dependency verification must not contain trust bypasses", errors)
        self.assertIn(
            "Gradle verification metadata omits required components: junit:junit:4.13.2",
            errors,
        )

    def test_rejects_malformed_checksum_and_artifact_path(self):
        tree = ET.parse(self.source)
        namespace = verify_release.GRADLE_VERIFICATION_NS
        artifacts = tree.getroot().findall(f".//{namespace}artifact")
        artifacts[0][0].set("value", "0" * 63)
        artifacts[1].set("name", "../escaped.module")
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "verification-metadata.xml"
            tree.write(path, encoding="utf-8", xml_declaration=True)
            errors = []
            verify_release.validate_gradle_dependency_verification(errors, path)
        self.assertIn("Gradle verification metadata contains an invalid artifact", errors)
        self.assertIn(
            "Gradle verification artifact does not have exactly one SHA-256 checksum",
            errors,
        )


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
