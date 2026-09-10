# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import inspect
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_swipe_candidate_recall as diagnostic
import evaluate_swipe_ctc as evaluator


class CandidateRecallDiagnosticTest(unittest.TestCase):
    def test_frozen_slate_hash_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "slates.jsonl"
            path.write_bytes(b'{"id":"x"}\n')
            with self.assertRaises(evaluator.SwipeEvaluationError):
                diagnostic.verify_frozen_slates(path)

    def test_metric_tables_count_membership_and_ranking_without_building_candidates_from_targets(self):
        rows = [
            {"target": "cat", "strata": ["short", "clean"]},
            {"target": "dog", "strata": ["long"]},
        ]
        present = [True, False]
        ranked = [["cat", "car"], ["dot", "doe", "dog"]]
        membership, ranking = diagnostic.metric_tables(rows, present, ranked)
        self.assertEqual(1, membership["overall"]["targetPresent"])
        self.assertEqual(0.5, membership["overall"]["recall"])
        self.assertEqual(1, ranking["overall"]["top1"])
        self.assertEqual(2, ranking["overall"]["top3"])
        self.assertNotIn("target", inspect.signature(diagnostic.scored).parameters)
        self.assertNotIn("target", inspect.signature(diagnostic.verify_frozen_slates).parameters)

    def test_scored_round_trip_preserves_frequency_and_spatial(self):
        items = [{"word": "cat", "language": "en", "frequency": 12, "spatial": -0.25}]
        entries = diagnostic.scored(items)
        self.assertEqual("cat", entries[0].word)
        self.assertEqual(12, entries[0].entry.frequency)
        self.assertEqual(-0.25, entries[0].spatial)
        self.assertFalse(entries[0].frequency_free)


if __name__ == "__main__":
    unittest.main()
