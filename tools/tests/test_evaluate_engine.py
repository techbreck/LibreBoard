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

        result = evaluate(rows, {"schemaVersion": 1, "peakAddedNeuralMemoryMiB": 40},
                          enforce_minimum_counts=False)
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_session_crossing_splits_is_rejected(self):
        systems = {system: ["target"] for system in ("heliboard", "fused", "fused_personal", "fused_neural")}
        rows = [
            example(1, "tap_error", "target", "raw", systems, split="train", session="same"),
            example(2, "tap_error", "target", "raw", systems, session="same"),
        ]
        with self.assertRaisesRegex(EvaluationError, "sessions cross"):
            evaluate(rows, {"schemaVersion": 1, "peakAddedNeuralMemoryMiB": 1},
                     enforce_minimum_counts=False)

    def test_missing_valid_word_label_is_rejected(self):
        systems = {system: ["target"] for system in ("heliboard", "fused", "fused_personal", "fused_neural")}
        with self.assertRaisesRegex(EvaluationError, "shouldCorrect"):
            example(1, "valid_word", "target", "raw", systems)


if __name__ == "__main__":
    unittest.main()
