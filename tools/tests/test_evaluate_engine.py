# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluate_engine import EvaluationError, evaluate, parse_example  # noqa: E402


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
            "fused": ["target"],
            "fused_personal": ["target"],
            "fused_neural": ["target"],
        }
        rows = [example(1, "tap_error", "target", "targte", tap_systems)]
        rows += [example(2, "spacing", "in the", "inthe", tap_systems)]
        rows += [example(3, "lexical", "don't", "dont", tap_systems)]
        rows += [example(4, "valid_word", "there", "their", {
            "heliboard": ["their"], "fused": ["their"], "fused_personal": ["their"],
            "fused_neural": ["there"],
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

        result = evaluate(rows, metadata(), enforce_minimum_counts=False)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual("b" * 64, result["evidence"]["coreApkSha256"])

    def test_session_crossing_splits_is_rejected(self):
        systems = {system: ["target"] for system in ("heliboard", "fused", "fused_personal", "fused_neural")}
        rows = [
            example(1, "tap_error", "target", "raw", systems, split="train", session="same"),
            example(2, "tap_error", "target", "raw", systems, session="same"),
        ]
        with self.assertRaisesRegex(EvaluationError, "sessions cross"):
            evaluate(rows, metadata(), enforce_minimum_counts=False)

    def test_missing_valid_word_label_is_rejected(self):
        systems = {system: ["target"] for system in ("heliboard", "fused", "fused_personal", "fused_neural")}
        with self.assertRaisesRegex(EvaluationError, "shouldCorrect"):
            example(1, "valid_word", "target", "raw", systems)

    def test_missing_reference_environment_is_rejected(self):
        value = metadata()
        value["environments"] = value["environments"][:-1]
        with self.assertRaisesRegex(EvaluationError, "three reference environments"):
            evaluate(self._minimum_rows(), value, enforce_minimum_counts=False)

    def test_grapheneos_run_with_google_play_is_rejected(self):
        value = metadata()
        value["environments"][1]["sandboxedGooglePlayInstalled"] = True
        with self.assertRaisesRegex(EvaluationError, "without sandboxed Google Play"):
            evaluate(self._minimum_rows(), value, enforce_minimum_counts=False)

    def test_unbound_artifact_hash_is_rejected(self):
        value = metadata()
        value["coreApkSha256"] = "not-a-hash"
        with self.assertRaisesRegex(EvaluationError, "coreApkSha256"):
            evaluate(self._minimum_rows(), value, enforce_minimum_counts=False)

    def test_emulator_must_be_reported_as_low_ram(self):
        value = metadata()
        value["environments"][2]["isLowRamDevice"] = False
        with self.assertRaisesRegex(EvaluationError, "low-RAM emulator"):
            evaluate(self._minimum_rows(), value, enforce_minimum_counts=False)

    @staticmethod
    def _minimum_rows():
        tap = {system: ["target"] for system in ("heliboard", "fused", "fused_personal", "fused_neural")}
        swipe = {system: ["target"] for system in ("geometric", "ctc", "fused_swipe")}
        return [
            example(11, "tap_error", "target", "raw", tap),
            example(12, "valid_word", "target", "target", tap, should_correct=False),
            example(13, "swipe", "target", "", swipe, strata=["short", "return_trip"]),
        ]


if __name__ == "__main__":
    unittest.main()
