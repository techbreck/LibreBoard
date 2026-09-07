# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_context_model as evaluator


class ContextEvaluationTests(unittest.TestCase):
    def test_partial_export_cannot_enter_normal_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "export-report.json"
            path.write_text(json.dumps({"schemaVersion": 1, "modelId": "context-en-de-v1",
                                        "modelSpecSha256": "a" * 64, "releaseEligible": False}))
            with self.assertRaisesRegex(evaluator.ContextEvaluationError, "requires --development"):
                evaluator.load_export(path, SimpleNamespace(sha256="a" * 64), False)

    def test_wrong_model_spec_is_rejected_even_in_development(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "export-report.json"
            path.write_text(json.dumps({"schemaVersion": 1, "modelId": "context-en-de-v1",
                                        "modelSpecSha256": "b" * 64, "releaseEligible": False}))
            with self.assertRaisesRegex(evaluator.ContextEvaluationError, "incompatible"):
                evaluator.load_export(path, SimpleNamespace(sha256="a" * 64), True)

    def test_constant_scores_do_not_win_from_observed_candidate_position(self):
        metrics = evaluator.ranking_metrics([0.0] * 8, [-1.0] * 8)
        self.assertEqual(0, metrics["observedTop1"])
        self.assertEqual(0, metrics["observedTop3"])
        self.assertEqual(0, metrics["teacherTop1Agreement"])
        self.assertEqual(1, metrics["tiedTop1"])

    def test_ranking_and_teacher_agreement_are_independent(self):
        metrics = evaluator.ranking_metrics([-2.0, -1.0, -3.0], [-3.0, -1.0, -2.0])
        self.assertEqual(0, metrics["observedTop1"])
        self.assertEqual(1, metrics["observedTop3"])
        self.assertEqual(1, metrics["teacherTop1Agreement"])

    def test_tie_at_third_place_does_not_reward_label_position(self):
        metrics = evaluator.ranking_metrics([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(0, metrics["observedTop3"])

    def test_invalid_scores_fail_closed(self):
        for values in ([float("nan"), 0.0], [float("inf"), 0.0], [0.0]):
            with self.subTest(values=values), self.assertRaises(evaluator.ContextEvaluationError):
                evaluator.ranking_metrics(values, [0.0, 1.0])

    def test_truncated_run_requires_development_before_loading_artifacts(self):
        with self.assertRaisesRegex(evaluator.ContextEvaluationError, "requires --development"):
            evaluator.evaluate(SimpleNamespace(maximum_examples=1, development=False))

    def test_artifact_tamper_symlink_and_traversal_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"original")
            metadata = {"file": artifact.name, "bytes": 8,
                        "sha256": hashlib.sha256(b"original").hexdigest()}
            self.assertEqual(artifact, evaluator._artifact(root, metadata, 100, "fixture"))
            artifact.write_bytes(b"tampered")
            with self.assertRaises(evaluator.ContextEvaluationError):
                evaluator._artifact(root, metadata, 100, "fixture")
            (root / "link.onnx").symlink_to(artifact)
            for filename in ("../model.onnx", "link.onnx"):
                with self.assertRaises(evaluator.evaluate_swipe_ctc.SwipeEvaluationError):
                    evaluator._artifact(root, {**metadata, "file": filename}, 100, "fixture")

    def test_artifact_size_cannot_be_boolean_or_exceed_limit(self):
        for size in (True, 0, 101):
            with self.subTest(size=size), self.assertRaises(evaluator.ContextEvaluationError):
                evaluator._artifact(pathlib.Path("."), {"file": "model.onnx", "bytes": size,
                    "sha256": "a" * 64}, 100, "fixture")


if __name__ == "__main__":
    unittest.main()
