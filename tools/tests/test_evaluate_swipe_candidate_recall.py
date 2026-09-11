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

    def test_parse_reserved_sources_rejects_unknown_names(self):
        self.assertEqual(("greedy", "greedy_alts"), diagnostic.parse_reserved_sources("greedy,greedy_alts"))
        restored = diagnostic.parse_reserved_sources(
            "greedy,greedy_alts,nbest,neighbors,truncated_ctc,truncated_geometry",
        )
        self.assertEqual(
            ("greedy", "greedy_alts", "nbest", "neighbors", "truncated_ctc", "truncated_geometry"),
            restored,
        )
        with self.assertRaises(evaluator.SwipeEvaluationError):
            diagnostic.parse_reserved_sources("greedy,rerank")

    def test_published_31_defaults_are_one_slot_greedy_train_fit_ols(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "out.json"
            args = diagnostic.parse_args(["recall-6000", "--output", str(output)])
        self.assertEqual(1, args.reserved_budget)
        self.assertEqual(1, args.reserved_oov_nbest)
        self.assertEqual(4, args.oov_beam_width)
        self.assertEqual(0.0, args.oov_map_blend)
        self.assertEqual(0, args.park_extra_reserved_min_rank)
        self.assertFalse(args.converting_best_occupant)
        self.assertFalse(args.prefer_converting_greedy_alts)
        self.assertFalse(args.ablation_extra_fill)
        self.assertFalse(args.leftover_inlex_fill)
        self.assertFalse(args.converting_leftover_extras)
        self.assertFalse(args.converting_alt_expand)
        self.assertFalse(args.ablation_first_source_fill)
        self.assertFalse(args.converting_fill_loss_append)
        self.assertFalse(args.leftover_greedy_alts_append)
        self.assertFalse(args.protect_frozen_ranks)
        self.assertFalse(args.oov_conservative_spatial)
        self.assertEqual(
            ("greedy", "greedy_alts"),
            diagnostic.parse_reserved_sources(args.reserved_sources),
        )

    def test_metric_tables_treat_ranked_list_as_the_published_bound(self):
        rows = [{"target": "cax", "strata": ["short"]}, {"target": "dog", "strata": ["long"]}]
        present = [True, False]
        ranked = [["cat", "cax"], ["dot"] * 31]
        membership, ranking = diagnostic.metric_tables(rows, present, ranked)
        self.assertEqual(1, membership["overall"]["targetPresent"])
        self.assertEqual(31, len(ranked[1]))
        self.assertEqual(0, ranking["overall"]["top1"])
        self.assertEqual(1, ranking["overall"]["top3"])
        self.assertNotIn("target", inspect.signature(diagnostic.parse_reserved_sources).parameters)

    def test_conversion_outside_report_counts_ols_that_would_enter_top3(self):
        outside = {
            "count": 3,
            "frequencyFree": 2,
            "inLexicon": 1,
            "bySource": {"greedy": 1, "greedy_alts": 1, "nbest": 0, "neighbors": 1, "truncated_ctc": 0, "truncated_geometry": 0},
            "rank": {"4-6": 2, "7-10": 1, "11-20": 0, "21-31": 0},
            "blendedGaps": [-0.2, -0.05],
            "olsGaps": [0.1, -0.01],
            "inLexiconGaps": [-0.4],
        }
        report = diagnostic._conversion_outside_report(outside)
        self.assertEqual(3, report["count"])
        self.assertEqual(2, report["frequencyFree"])
        self.assertEqual(1, report["olsWouldEnterTop3"])
        self.assertEqual(1, report["blendedGap"]["within0.10"])
        self.assertNotIn("target", inspect.signature(diagnostic._conversion_outside_report).parameters)

    def test_unpublished_competing_labels_fill_loss_without_reading_targets(self):
        lexicon = [
            diagnostic.evaluator.ScoredLexiconEntry(
                diagnostic.evaluator.LexiconEntry("the", "en", (1,), 200), 3.0,
            ),
            diagnostic.evaluator.ScoredLexiconEntry(
                diagnostic.evaluator.LexiconEntry("and", "en", (1,), 180), 2.5,
            ),
            diagnostic.evaluator.ScoredLexiconEntry(
                diagnostic.evaluator.LexiconEntry("for", "en", (1,), 160), 2.2,
            ),
        ]
        dest = [item.spatial for item in lexicon]
        conservative = diagnostic.evaluator.conservative_lexicon_spatial(dest)
        ols = 8.0
        blended = 0.5 * ols + 0.5 * conservative
        match = diagnostic.evaluator.ScoredLexiconEntry(
            diagnostic.evaluator.LexiconEntry("cax", "en", (2,), 0),
            blended,
            frequency_free=True,
            source="greedy_alts",
            oov_map_blend=0.5,
        )
        unpublished = diagnostic._empty_unpublished_competing()
        diagnostic._record_unpublished_competing(
            unpublished,
            match=match,
            fill_rank=18,
            oov_ctc_rank=9,
            in_frozen=False,
            in_merged=False,
            reserved_budget=11,
            extra_oov=7,
            lexicon=lexicon,
            blend=0.5,
            merged_rank=None,
        )
        report = diagnostic._unpublished_competing_report(unpublished)
        self.assertEqual(1, report["count"])
        self.assertEqual(1, report["newUnpublished"])
        self.assertEqual(1, report["firstSource"]["greedy_alts"])
        self.assertEqual(1, report["fillRank"]["gt11"])
        self.assertEqual(1, report["oovCtcRank"]["gt7"])
        self.assertEqual(1, report["olsWouldEnterTop3"])
        self.assertEqual(1, report["convertingFillLoss"])
        self.assertNotIn("target", inspect.signature(diagnostic._record_unpublished_competing).parameters)
        self.assertNotIn("target", inspect.signature(diagnostic._unpublished_competing_report).parameters)
        self.assertNotIn("target", inspect.signature(diagnostic._empty_unpublished_competing).parameters)
        weak = diagnostic.evaluator.ScoredLexiconEntry(
            diagnostic.evaluator.LexiconEntry("caz", "en", (3,), 0),
            -9.0,
            frequency_free=True,
            source="greedy_alts",
            oov_map_blend=0.5,
        )
        unpublished_weak = diagnostic._empty_unpublished_competing()
        diagnostic._record_unpublished_competing(
            unpublished_weak,
            match=weak,
            fill_rank=18,
            oov_ctc_rank=9,
            in_frozen=False,
            in_merged=False,
            reserved_budget=11,
            extra_oov=7,
            lexicon=lexicon,
            blend=0.5,
        )
        weak_report = diagnostic._unpublished_competing_report(unpublished_weak)
        self.assertEqual(1, weak_report["fillRank"]["gt11"])
        self.assertEqual(0, weak_report["olsWouldEnterTop3"])
        self.assertEqual(0, weak_report["convertingFillLoss"])
        frozen_lost = diagnostic._empty_unpublished_competing()
        diagnostic._record_unpublished_competing(
            frozen_lost,
            match=None,
            fill_rank=None,
            oov_ctc_rank=None,
            in_frozen=True,
            in_merged=True,
            reserved_budget=11,
            extra_oov=7,
            lexicon=lexicon,
            blend=0.5,
            merged_rank=22,
        )
        frozen_report = diagnostic._unpublished_competing_report(frozen_lost)
        self.assertEqual(1, frozen_report["frozenLost"])
        self.assertEqual(1, frozen_report["inAdaptiveMergedOnly"])
        self.assertEqual(1, frozen_report["frozenLostMergedRankLe23"])
        self.assertEqual(22, frozen_report["rows"][0]["mergedRank"])
        self.assertNotIn("target", inspect.signature(diagnostic._record_unpublished_competing).parameters)


if __name__ == "__main__":
    unittest.main()
