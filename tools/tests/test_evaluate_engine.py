# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluate_engine import EvaluationError, evaluate, parse_example, read_jsonl, validate_swipe_strata  # noqa: E402


def example(
    number: int,
    category: str,
    target: str,
    raw: str,
    predictions: dict[str, list[str]],
    *,
    split: str = "test",
    session: str | None = None,
    strata: list[str] | None = None,
    should_correct: bool | None = None,
    environment_kind: str = "stock_android_hardware",
    test_run_id: str = "stock-run",
    latency_overrides: dict[str, float] | None = None,
):
    latency = {system: 20.0 for system in predictions}
    latency.update(latency_overrides or {})
    value = {
        "schemaVersion": 3,
        "id": f"example-{number}",
        "sessionId": session or f"session-{number}",
        "split": split,
        "category": category,
        "target": target,
        "raw": raw,
        "predictions": predictions,
        "latencyMs": latency,
        "strata": strata or [],
    }
    if split == "test":
        value["environmentKind"] = environment_kind
        value["testRunId"] = test_run_id
    if category == "lexical":
        value["lexicalKind"] = "contraction"
    if should_correct is not None:
        value["shouldCorrect"] = should_correct
    return parse_example(value, number)


def metadata():
    return {
        "schemaVersion": 3,
        "appCommit": "a" * 40,
        "coreApkSha256": "b" * 64,
        "swipeModelSha256": "c" * 64,
        "contextModelSha256": "d" * 64,
        "environments": [
            {
                "kind": "stock_android_hardware",
                "deviceModel": "Pixel 8a",
                "buildFingerprint": "stock/fingerprint",
                "testRunId": "stock-run",
                "apiLevel": 36,
                "physicalDevice": True,
                "peakAddedNeuralMemoryMiB": 40,
            },
            {
                "kind": "grapheneos_hardware",
                "deviceModel": "Pixel 8a",
                "buildFingerprint": "graphene/fingerprint",
                "testRunId": "graphene-run",
                "apiLevel": 36,
                "physicalDevice": True,
                "grapheneOsBuildNumber": "2026090100",
                "sandboxedGooglePlayInstalled": False,
                "peakAddedNeuralMemoryMiB": 42,
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
                "peakAddedNeuralMemoryMiB": 18,
            },
        ],
    }


class EvaluateEngineTest(unittest.TestCase):
    def test_lexical_kind_is_required_and_category_scoped(self):
        row = {"schemaVersion": 3, "id": "lexical", "sessionId": "session", "split": "train",
               "category": "lexical", "raw": "dont", "target": "don't"}
        for invalid in (None, "unknown", True):
            with self.subTest(kind=invalid), self.assertRaisesRegex(EvaluationError, "valid lexicalKind"):
                parse_example({**row, "lexicalKind": invalid}, 1)
        with self.assertRaisesRegex(EvaluationError, "valid lexicalKind"):
            parse_example(row, 1)
        self.assertEqual("contraction", parse_example({**row, "lexicalKind": "contraction"}, 1).lexical_kind)
        with self.assertRaisesRegex(EvaluationError, "only for lexical"):
            parse_example({**row, "category": "tap_error", "lexicalKind": "contraction"}, 1)
        with self.assertRaisesRegex(EvaluationError, "unsupported schemaVersion"):
            parse_example({**row, "schemaVersion": 2, "lexicalKind": "contraction"}, 1)

    def test_valid_word_labels_cannot_reclassify_keeps_as_corrections(self):
        row = {"schemaVersion": 3, "id": "valid", "sessionId": "session", "split": "train",
               "category": "valid_word", "raw": "their", "target": "there"}
        with self.assertRaisesRegex(EvaluationError, "contradicts"):
            parse_example({**row, "shouldCorrect": False}, 1)
        with self.assertRaisesRegex(EvaluationError, "contradicts"):
            parse_example({**row, "target": "THEIR", "shouldCorrect": True}, 1)
        self.assertFalse(parse_example({**row, "target": "THEIR", "shouldCorrect": False}, 1).should_correct)

    def test_passing_measurements_satisfy_every_gate(self):
        tap_systems = {
            "heliboard": ["wrong"],
            "fused": ["target", "targte", "inthe", "dont"],
            "fused_personal": ["target", "targte", "inthe", "dont"],
            "fused_neural": ["target", "targte", "inthe", "dont"],
        }
        rows = [example(1, "tap_error", "target", "targte", tap_systems)]
        rows += [example(2, "spacing", "in the", "inthe", tap_systems)]
        rows += [example(3, "lexical", "don't", "dont", tap_systems)]
        rows += [example(4, "valid_word", "there", "their", {
            "heliboard": ["their"], "fused": ["their"], "fused_personal": ["their"],
            "fused_neural": ["there", "their"],
        }, should_correct=True)]
        rows += [example(5, "valid_word", "their", "their", {
            "heliboard": ["their"], "fused": ["their"], "fused_personal": ["their"],
            "fused_neural": ["their"],
        }, should_correct=False)]
        swipe_systems = {
            "geometric": ["wrong"], "ctc": ["target"], "fused_swipe": ["target"],
        }
        rows += [example(6, "swipe", "target", "", swipe_systems, strata=["short"])]
        rows += [example(7, "swipe", "target", "", swipe_systems, strata=["return_trip"])]
        rows += [example(8, "tap_error", "target", "targte", tap_systems,
                         environment_kind="grapheneos_hardware", test_run_id="graphene-run")]
        rows += [example(9, "swipe", "target", "", swipe_systems, strata=["medium"],
                         environment_kind="grapheneos_hardware", test_run_id="graphene-run")]
        rows += [example(10, "tap_error", "target", "targte", tap_systems,
                         environment_kind="low_ram_emulator", test_run_id="low-ram-run")]
        rows += [example(11, "swipe", "target", "", swipe_systems, strata=["long"],
                         environment_kind="low_ram_emulator", test_run_id="low-ram-run")]

        result = evaluate(
            rows,
            metadata(),
            measurement_sha256="e" * 64,
            enforce_minimum_counts=False,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual("b" * 64, result["evidence"]["coreApkSha256"])
        self.assertEqual(1, result["swipeStrataCounts"]["short"])
        self.assertEqual(1.0, result["gates"]["neuralContextRelativeErrorReduction"])
        self.assertEqual({"correct": 1, "keep": 1}, result["validWordCounts"])

    def test_session_crossing_splits_is_rejected(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        rows = [
            example(1, "tap_error", "target", "raw", systems, split="train", session="same"),
            example(2, "tap_error", "target", "raw", systems, session="same"),
        ]
        with self.assertRaisesRegex(EvaluationError, "sessions cross"):
            evaluate(
                rows,
                metadata(),
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_missing_valid_word_label_is_rejected(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        with self.assertRaisesRegex(EvaluationError, "shouldCorrect"):
            example(1, "valid_word", "target", "raw", systems)

    def test_measured_tap_must_preserve_exact_raw_candidate(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "Raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        with self.assertRaisesRegex(EvaluationError, "fused does not preserve the exact raw"):
            example(1, "tap_error", "target", "raw", systems)

    def test_measurement_rows_reject_cross_path_systems_and_duplicate_candidates(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
            "ctc": ["target"],
        }
        with self.assertRaisesRegex(EvaluationError, "exactly the applicable systems"):
            example(1, "tap_error", "target", "raw", systems)

        systems.pop("ctc")
        systems["fused"] = ["target", "raw", "RAW"]
        with self.assertRaisesRegex(EvaluationError, "normalization-distinct"):
            example(2, "tap_error", "target", "raw", systems)

    def test_should_correct_is_rejected_outside_valid_word_rows(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        with self.assertRaisesRegex(EvaluationError, "only for valid_word"):
            example(1, "tap_error", "target", "raw", systems, should_correct=True)

    def test_jsonl_reader_rejects_unbounded_or_invalid_utf8_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "measurements.jsonl"
            path.write_bytes(b"x" * (1024 * 1024 + 1) + b"\n")
            with self.assertRaisesRegex(EvaluationError, "byte limit"):
                read_jsonl(path)

            path.write_bytes(b"\xff\n")
            with self.assertRaisesRegex(EvaluationError, "invalid UTF-8"):
                read_jsonl(path)

    def test_release_swipe_strata_require_substantial_coverage(self):
        swipe = {system: ["target"] for system in ("geometric", "ctc", "fused_swipe")}
        rows = [
            example(100 + index, "swipe", "target", "", swipe, strata=["short", "clean"])
            for index in range(500)
        ]
        with self.assertRaisesRegex(EvaluationError, "medium=0<500"):
            validate_swipe_strata(rows, {
                "short": 500,
                "medium": 500,
                "clean": 500,
            })

    def test_missing_reference_environment_is_rejected(self):
        value = metadata()
        value["environments"] = value["environments"][:-1]
        with self.assertRaisesRegex(EvaluationError, "three reference environments"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_grapheneos_run_with_google_play_is_rejected(self):
        value = metadata()
        value["environments"][1]["sandboxedGooglePlayInstalled"] = True
        with self.assertRaisesRegex(EvaluationError, "without sandboxed Google Play"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_unbound_artifact_hash_is_rejected(self):
        value = metadata()
        value["coreApkSha256"] = "not-a-hash"
        with self.assertRaisesRegex(EvaluationError, "coreApkSha256"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_emulator_must_be_reported_as_low_ram(self):
        value = metadata()
        value["environments"][2]["isLowRamDevice"] = False
        with self.assertRaisesRegex(EvaluationError, "low-RAM emulator"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_measurement_run_must_match_declared_environment(self):
        rows = self._minimum_rows()
        rows[0] = example(
            11,
            "tap_error",
            "target",
            "raw",
            {system: list(values) for system, values in rows[0].predictions.items()},
            environment_kind="grapheneos_hardware",
            test_run_id="stock-run",
        )
        with self.assertRaisesRegex(EvaluationError, "environmentKind disagrees"):
            evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False)

    def test_release_evidence_requires_substantial_measurements_from_each_environment(self):
        with self.assertRaisesRegex(EvaluationError, "measurement coverage for stock_android_hardware"):
            evaluate(
                self._minimum_rows(),
                metadata(),
                measurement_sha256="e" * 64,
                enforce_minimum_counts=True,
            )

    def test_metadata_rejects_duplicate_run_ids(self):
        value = metadata()
        value["environments"][1]["testRunId"] = "stock-run"
        with self.assertRaisesRegex(EvaluationError, "duplicate testRunId"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_peak_memory_is_derived_from_the_slowest_environment(self):
        value = metadata()
        value["environments"][1]["peakAddedNeuralMemoryMiB"] = 65
        result = evaluate(
            self._minimum_rows(),
            value,
            measurement_sha256="e" * 64,
            enforce_minimum_counts=False,
        )
        self.assertEqual(65.0, result["evidence"]["peakAddedNeuralMemoryMiB"])
        self.assertFalse(result["checks"]["peak_neural_memory"])

    def test_each_environment_controls_its_own_latency_gate(self):
        rows = self._minimum_rows()
        tap = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        rows.extend(
            example(100 + index, "tap_error", "target", "raw", tap,
                    latency_overrides={"fused_neural": 1.0})
            for index in range(30)
        )
        rows[3] = example(
            14,
            "tap_error",
            "target",
            "raw",
            tap,
            environment_kind="grapheneos_hardware",
            test_run_id="graphene-run",
            latency_overrides={"fused_neural": 81.0},
        )
        result = evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False)
        self.assertLessEqual(result["systems"]["fused_neural"]["latencyMs"]["p95"], 80.0)
        self.assertEqual(81.0, result["environmentLatencyMs"]["grapheneos_hardware"]["tap"]["p95"])
        self.assertFalse(result["checks"]["tap_p95_latency"])

    @staticmethod
    def _minimum_rows():
        tap = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        swipe = {system: ["target"] for system in ("geometric", "ctc", "fused_swipe")}
        return [
            example(11, "tap_error", "target", "raw", tap),
            example(12, "valid_word", "target", "target", tap, should_correct=False),
            example(13, "swipe", "target", "", swipe, strata=["short", "return_trip"]),
            example(14, "tap_error", "target", "raw", tap,
                    environment_kind="grapheneos_hardware", test_run_id="graphene-run"),
            example(15, "swipe", "target", "", swipe, strata=["medium"],
                    environment_kind="grapheneos_hardware", test_run_id="graphene-run"),
            example(16, "tap_error", "target", "raw", tap,
                    environment_kind="low_ram_emulator", test_run_id="low-ram-run"),
            example(17, "swipe", "target", "", swipe, strata=["long"],
                    environment_kind="low_ram_emulator", test_run_id="low-ram-run"),
            example(18, "valid_word", "there", "their", {
                "heliboard": ["their"],
                "fused": ["their"],
                "fused_personal": ["their"],
                "fused_neural": ["there", "their"],
            }, should_correct=True),
        ]


if __name__ == "__main__":
    unittest.main()
