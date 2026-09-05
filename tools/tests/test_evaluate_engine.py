# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluate_engine import EvaluationError, evaluate, parse_example, validate_swipe_strata  # noqa: E402


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
):
    latency = {system: 20.0 for system in predictions}
    value = {
        "schemaVersion": 1,
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
    if should_correct is not None:
        value["shouldCorrect"] = should_correct
    return parse_example(value, number)


def metadata():
    return {
        "schemaVersion": 1,
        "appCommit": "a" * 40,
        "coreApkSha256": "b" * 64,
        "swipeModelSha256": "c" * 64,
        "contextModelSha256": "d" * 64,
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
                "buildFingerprint": "graphene/fingerprint",
                "testRunId": "graphene-run",
                "apiLevel": 36,
                "physicalDevice": True,
                "grapheneOsBuildNumber": "2026090100",
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


class EvaluateEngineTest(unittest.TestCase):
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
        ]


if __name__ == "__main__":
    unittest.main()
