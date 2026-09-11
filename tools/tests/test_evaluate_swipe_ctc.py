# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import zipfile
import pathlib
import sys
import unittest

try:
    import numpy
except ImportError:
    numpy = None


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_swipe_ctc as evaluator  # noqa: E402


def identifier(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def row(index: int, strata: set[str]) -> evaluator.EvaluationRow:
    return evaluator.EvaluationRow(
        identifier=identifier(str(index)),
        session_id=identifier(f"session-{index}"),
        language="en",
        target="cat",
        labels=(1, 2, 3),
        path=tuple([0.0] * 128),
        strata=frozenset(strata),
    )


def logits(emissions: list[int], classes: int = 8, frames: int = 12) -> list[list[float]]:
    result = []
    for frame in range(frames):
        output_class = emissions[frame] if frame < len(emissions) else 0
        values = [-12.0] * classes
        values[output_class] = 12.0
        result.append(values)
    return result


def trie(entries: list[evaluator.LexiconEntry]) -> evaluator.TrieNode:
    return evaluator.build_trie(entries, approximate_length=4)


class EvaluateSwipeCtcTest(unittest.TestCase):
    def test_validation_vocabulary_excludes_validation_and_test_targets(self):
        layout = {"keyLabels": list("catdog"), "keyMask": [1] * 6}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for split, target in (("train", "cat"), ("validation", "dog"), ("test", "god")):
                (root / f"{split}.jsonl").write_text(json.dumps({
                    "split": split, "language": "en", "target": target,
                }) + "\n")
            validation = evaluator.build_lexicon(root, layout, evaluation_split="validation")
            final = evaluator.build_lexicon(root, layout, evaluation_split="test")
            self.assertEqual({"cat"}, {entry.word for entry in validation["en"]})
            self.assertEqual({"cat", "dog"}, {entry.word for entry in final["en"]})

    def test_native_dictionary_export_requires_matching_apk_and_asset(self):
        layout = {"keyLabels": list("catdog"), "keyMask": [1] * 6}
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            apk, export = root / "test.apk", root / "lexicon.json"
            with zipfile.ZipFile(apk, "w") as archive:
                archive.writestr("assets/dicts/main_en-US.dict", b"dictionary")
            value = {
                "schemaVersion": 1, "source": "bundled-static-dictionary", "maximumWords": 100000,
                "apkSha256": hashlib.sha256(apk.read_bytes()).hexdigest(),
                "dictionaryAsset": "dicts/main_en-US.dict",
                "dictionarySha256": hashlib.sha256(b"dictionary").hexdigest(),
                "words": [{"word": "cat", "languageTag": "en-US", "frequency": 200, "possiblyOffensive": False},
                          {"word": "dog", "languageTag": "en-US", "frequency": 100, "possiblyOffensive": True}],
            }
            export.write_text(json.dumps(value))
            entries, provenance = evaluator.load_dictionary_lexicon(export, apk, layout)
            self.assertEqual(["cat"], [entry.word for entry in entries["en"]])
            self.assertEqual([], provenance["sourceSplits"])
            for field in ("apkSha256", "dictionarySha256"):
                corrupt = dict(value, **{field: "0" * 64})
                export.write_text(json.dumps(corrupt))
                with self.assertRaises(evaluator.SwipeEvaluationError):
                    evaluator.load_dictionary_lexicon(export, apk, layout)
            value["words"].append(dict(value["words"][0], word="CAT"))
            export.write_text(json.dumps(value))
            with self.assertRaises(evaluator.SwipeEvaluationError):
                evaluator.load_dictionary_lexicon(export, apk, layout)

    def test_greedy_and_prefix_beam_handle_blank_separated_double_letters(self):
        entries = [
            evaluator.LexiconEntry("al", "en", (1, 2), 100),
            evaluator.LexiconEntry("all", "en", (1, 2, 2), 90),
        ]
        output = logits([1, 2, 0, 2])

        self.assertEqual((1, 2, 2), evaluator.collapse_greedy(output))
        self.assertEqual("all", evaluator.prefix_beam_decode(output, trie(entries))[0].word)

    def test_repeated_class_without_blank_collapses_and_return_trip_survives(self):
        repeated = [
            evaluator.LexiconEntry("l", "en", (2,), 100),
            evaluator.LexiconEntry("ll", "en", (2, 2), 90),
        ]
        returning = [evaluator.LexiconEntry("pop", "en", (3, 4, 3), 100)]

        self.assertEqual("l", evaluator.prefix_beam_decode(logits([2, 2]), trie(repeated))[0].word)
        self.assertEqual("pop", evaluator.prefix_beam_decode(logits([3, 4, 3]), trie(returning))[0].word)

    def test_apostrophe_free_emissions_retain_contraction_surface(self):
        labels = {character: index + 1 for index, character in enumerate("dont")}
        self.assertEqual([(1, 2, 3, 4)], evaluator.emission_variants("don't", labels, "en"))
        entries = [
            evaluator.LexiconEntry("dont", "en", (1, 2, 3, 4), 10),
            evaluator.LexiconEntry("don't", "en", (1, 2, 3, 4), 100),
        ]

        self.assertEqual("don't", evaluator.prefix_beam_decode(logits([1, 2, 3, 4]), trie(entries))[0].word)

    def test_static_fusion_uses_production_z_scores_and_frequency_weight(self):
        rare = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("rare", "en", (1,), 1),
            spatial=-0.10,
        )
        common = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("common", "en", (2,), 10_000),
            spatial=-0.11,
        )
        distant = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("distant", "en", (3,), 100),
            spatial=-2.0,
        )

        ranked = evaluator.rank_static_fusion([rare, common, distant])

        self.assertEqual("common", ranked[0].word)
        self.assertEqual("rare", ranked[1].word)

    def test_decoder_slate_merge_normalizes_ctc_and_geometric_scales(self):
        ctc = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 10), -0.1),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 9), -0.2),
        ]
        geometric = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("dog", "en", (3,), 8), -100.0),
        ]

        merged = evaluator.merge_swipe_slates(ctc, geometric)

        self.assertEqual({"cat", "car", "dog"}, {candidate.word for candidate in merged})
        self.assertEqual(1.0, next(candidate.spatial for candidate in merged if candidate.word == "dog"))

    @unittest.skipUnless(numpy is not None, "NumPy is part of the optional model toolchain")
    def test_geometric_decoder_ranks_an_exact_live_geometry_trace_first(self):
        labels = ["a", "b", "c", "d"] + [None] * 60
        centers = [0.25, 0.2, 0.75, 0.2, 0.25, 0.8, 0.75, 0.8] + [0.0] * 120
        layout = {
            "keyLabels": labels,
            "keyCenters": centers,
            "keyMask": [1, 1, 1, 1] + [0] * 60,
        }
        lexicon = {"en": [
            evaluator.LexiconEntry("ab", "en", (1, 2), 10),
            evaluator.LexiconEntry("ac", "en", (1, 3), 100),
            evaluator.LexiconEntry("dc", "en", (4, 3), 100),
        ]}
        index = evaluator.build_geometric_index(lexicon, layout, numpy)
        path = evaluator._resample_template(
            numpy.asarray([[0.25, 0.2], [0.75, 0.2]], dtype=numpy.float32),
            numpy,
        ).reshape(-1).tolist()

        decoded = evaluator.geometric_decode(path, "en", index, numpy)

        self.assertEqual("ab", decoded[0].word)

    def test_german_popup_letters_have_scoped_base_key_emissions(self):
        labels = {character: index + 1 for index, character in enumerate("tase")}

        self.assertEqual([], evaluator.emission_variants("tät", labels, "en-US"))
        self.assertEqual([(1, 2, 1)], evaluator.emission_variants("tät", labels, "de-DE"))
        self.assertEqual(
            [(1, 3), (1, 3, 3)],
            evaluator.emission_variants("tß", labels, "de"),
        )

    def test_stratified_selection_is_order_independent_and_meets_every_minimum(self):
        rows = [row(index, {stratum}) for index, stratum in enumerate(evaluator.REQUIRED_STRATA)]
        rows += [row(100 + index, {"short", "clean"}) for index in range(4)]

        selected = evaluator.select_rows(rows, sample_count=10, minimum_per_stratum=1)
        reversed_selected = evaluator.select_rows(list(reversed(rows)), sample_count=10, minimum_per_stratum=1)

        self.assertEqual([item.identifier for item in selected], [item.identifier for item in reversed_selected])
        self.assertEqual(10, len(selected))
        for stratum in evaluator.REQUIRED_STRATA:
            self.assertTrue(any(stratum in item.strata for item in selected))

    def test_selection_fails_when_a_required_stratum_is_missing(self):
        rows = [row(index, {"short"}) for index in range(8)]
        with self.assertRaisesRegex(evaluator.SwipeEvaluationError, "medium"):
            evaluator.select_rows(rows, sample_count=8, minimum_per_stratum=1)

    def test_path_length_estimate_does_not_count_every_crossed_key(self):
        labels = list("qwertyuiopasdfghjklzxcvbnm")
        centers = []
        positions = {}
        for characters, y, offset in (
            ("qwertyuiop", 1 / 6, 0.05),
            ("asdfghjkl", 0.5, 0.1),
            ("zxcvbnm", 5 / 6, 0.2),
        ):
            for index, character in enumerate(characters):
                point = (offset + index * 0.1, y)
                positions[character] = point
                centers.extend(point)
        centers.extend([0.0] * (128 - len(centers)))
        layout = {
            "keyLabels": labels + [None] * (64 - len(labels)),
            "keyCenters": centers,
            "keyMask": [1] * len(labels) + [0] * (64 - len(labels)),
        }
        intended = [positions[character] for character in "both"]
        points = [intended[0]]
        for start, end in zip(intended, intended[1:]):
            for step in range(1, 22):
                fraction = step / 21
                points.append((
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                ))
        flattened = tuple(coordinate for point in points for coordinate in point)

        self.assertEqual(128, len(flattened))
        self.assertEqual(3, evaluator.path_length_estimate(flattened, layout))
        self.assertLessEqual(abs(evaluator.path_length_estimate(flattened, layout) - len("both")), 1)

    def test_ctc_forward_matches_prefix_beam_score_for_a_single_path(self):
        entries = [evaluator.LexiconEntry("ab", "en", (1, 2), 10)]
        output = logits([1, 2, 0, 2], classes=8, frames=8)
        scored = evaluator.prefix_beam_decode_scored(output, trie(entries))
        self.assertEqual("ab", scored[0].word)
        self.assertAlmostEqual(scored[0].spatial, evaluator.ctc_forward_logprob(output, (1, 2)), places=9)

    def test_oov_calibration_maps_onto_the_lexicon_constrained_scale(self):
        forwards = [-0.40, -0.20, -0.10, -0.02]
        spatials = [-0.85, -0.45, -0.25, -0.09]
        calibration = evaluator.fit_oov_score_calibration(forwards, spatials, optimism_gaps=[0.05, 0.07])
        mapped = [calibration.to_lexicon_spatial(value) for value in forwards]
        self.assertTrue(all(min(spatials) - 0.2 <= value <= max(spatials) + 0.2 for value in mapped))
        projected = evaluator.project_onto_lexicon_scale(0.50, spatials)
        self.assertEqual(max(spatials), projected)
        self.assertLess(calibration.optimism_offset, 0.1)
        self.assertIsNone(evaluator.oov_score_calibration_report(calibration)["frequencyPrior"])

    def test_oov_calibration_rejects_a_fabricated_frequency_prior(self):
        calibration = evaluator.fit_oov_score_calibration([-0.2, -0.1], [-0.2, -0.1])
        payload = evaluator.oov_score_calibration_report(calibration)
        payload["frequencyPrior"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "calibration.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(evaluator.SwipeEvaluationError):
                evaluator.load_oov_score_calibration(path)

    def test_reserved_slot_append_does_not_drop_an_existing_candidate(self):
        base = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), -0.1 * index)
            for index in range(32)
        ]
        reserved = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry("oovword", "en", (2, 3), 0), -0.05, frequency_free=True,
            ),
            evaluator.ScoredLexiconEntry(base[0].entry, 9.0, frequency_free=True),
        ]
        merged = evaluator.append_reserved_slots(base, reserved)
        self.assertEqual(33, len(merged))
        self.assertEqual([candidate.word for candidate in base], [candidate.word for candidate in merged[:32]])
        self.assertEqual("oovword", merged[32].word)

    def test_known_offensive_spellings_are_rejected_from_reserved_slots(self):
        layout = {"keyLabels": list("abcdefgh") + [None] * 56}
        output = logits([1, 2], classes=8, frames=6)
        calibration = evaluator.fit_oov_score_calibration([-0.2, -0.1], [-0.2, -0.1])
        existing = [evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("zz", "en", (3,), 8), -0.3)]
        reserved, rejected = evaluator.reserved_oov_from_logits(
            output,
            layout,
            calibration=calibration,
            n_best=4,
            beam_width=8,
            existing=existing,
            known_offensive={"ab"},
            lexicon_by_word={},
            language="en",
            ctc_spatials=[-0.3],
        )
        self.assertGreaterEqual(rejected, 1)
        self.assertNotIn("ab", {candidate.word for candidate in reserved})

    def test_construction_helpers_do_not_take_evaluation_targets(self):
        helpers = (
            evaluator.ctc_forward_logprob,
            evaluator.fit_oov_score_calibration,
            evaluator.reserved_oov_from_logits,
            evaluator.unconstrained_ctc_emissions,
            evaluator.unconstrained_prefix_beam_scored,
            evaluator.emissions_to_spelling,
            evaluator.append_reserved_slots,
            evaluator.collect_oov_calibration_observations,
            evaluator.load_train_paths,
            evaluator.decoder_slate_budgets,
            evaluator.rank_static_fusion_with_reserved,
            evaluator.project_onto_lexicon_scale,
            evaluator.lexicon_neighbors,
            evaluator.competing_slate,
            evaluator.conservative_lexicon_spatial,
            evaluator.median_lexicon_spatial,
            evaluator.map_z_onto_pool,
            evaluator._oov_blend_by_margin,
            evaluator.reserved_from_decoder_slates,
            evaluator.greedy_unconstrained_emissions,
            evaluator.greedy_alt_unconstrained_emissions,
            evaluator.nbest_unconstrained_emissions,
            evaluator.collect_reserved_sources,
            evaluator.flatten_reserved_sources,
            evaluator.score_reserved_candidate,
            evaluator.score_reserved_sources,
            evaluator.publish_reserved_slots,
            evaluator.prioritize_reserved_candidates,
            evaluator.reserved_occupants,
            evaluator.reserved_clears_lexicon_top3,
            evaluator.ols_reserved_candidate,
            evaluator.converting_best_reserved,
            evaluator.extra_reserved_occupant_keys,
            evaluator.park_extra_reserved_below_rank,
            evaluator.park_extra_frequency_free_to_protect_top3,
            evaluator.ablation_extra_fill,
            evaluator.leftover_inlex_after_converting_oov,
            evaluator.converting_leftover_extras,
            evaluator.converting_alt_expand_fill,
            evaluator.ablation_first_source_fill,
            evaluator.leftover_converting_greedy_neighbors,
            evaluator.converting_fill_loss_append_fill,
            evaluator.window_losing_converting_greedy_alts,
            evaluator.prefer_window_losing_converting_alts,
            evaluator.reserved_fill_ranks,
            evaluator.reserved_oov_ctc_ranks,
            evaluator.lift_near_top3_frequency_free,
            evaluator.unblend_greedy_alts_when_greedy_misses_top3,
            evaluator.split_published_slate,
            evaluator.published_ranking,
            evaluator.static_fusion_values,
            evaluator.reserved_fusion_values,
        )
        for helper in helpers:
            self.assertNotIn("target", inspect.signature(helper).parameters)
            self.assertNotIn("targets", inspect.signature(helper).parameters)
        self.assertIn("target", inspect.signature(evaluator.first_recovered_source).parameters)
        self.assertIn("target", inspect.signature(evaluator.cumulative_recovery_stage).parameters)

    def test_reserved_ranking_keeps_lexicon_z_pools_uncontaminated(self):
        rare = evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("rare", "en", (1,), 1), -0.10)
        common = evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("common", "en", (2,), 10_000), -0.11)
        distant = evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("distant", "en", (3,), 100), -2.0)
        lexicon = [rare, common, distant]
        baseline = [entry.word for entry in evaluator.rank_static_fusion(lexicon)]
        weak = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("zzzz", "en", (4,), 0), -3.0, frequency_free=True,
        )
        ranked = [entry.word for entry in evaluator.rank_static_fusion_with_reserved(lexicon, [weak])]
        self.assertEqual(baseline, ranked[:3])
        self.assertEqual(baseline, [entry.word for entry in evaluator.rank_static_fusion_with_reserved(lexicon, [])])

    def test_weak_frequency_free_oov_do_not_steal_lexicon_top3(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ]
        weak = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (4,), 0), -3.0, frequency_free=True,
        )
        extras = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"zz{index}", "en", (5,), 0), -3.0, frequency_free=True,
            )
            for index in range(8)
        ]
        reserved = [weak, *extras]
        published = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=4)
        self.assertLessEqual(len(published), 32)
        ranked = [entry.word for entry in evaluator.published_ranking(published, reserved)]
        self.assertEqual(["cat", "car", "can"], ranked[:3])
        self.assertLessEqual(len(ranked), 31)

    def test_frequency_free_oov_are_not_floor_clamped_below_top3(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ]
        strong = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (4,), 0), 8.0, frequency_free=True,
        )
        ranked = [entry.word for entry in evaluator.rank_static_fusion_with_reserved(lexicon, [strong])]
        self.assertEqual("cax", ranked[0])
        self.assertEqual(31, evaluator.PUBLISHED_RANKING_BOUND)

    def test_recall_is_membership_in_the_published_ranking_list(self):
        merged = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), 1.0 - 0.01 * index)
            for index in range(32)
        ]
        reserved = [evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0), -9.0, frequency_free=True,
        )]
        published = evaluator.publish_reserved_slots(merged, reserved, reserved_budget=1)
        self.assertEqual(31, len(published))
        self.assertIn("cax", {item.word for item in published})
        ranked = evaluator.published_ranking(published, reserved)
        self.assertEqual(31, len(ranked))
        self.assertIn("cax", [entry.word for entry in ranked])

    def test_conservative_oov_spatial_is_below_the_lexicon_median(self):
        spatials = [-0.8, -0.4, -0.1, 0.2]
        conservative = evaluator.conservative_lexicon_spatial(spatials)
        median = sorted(spatials)[len(spatials) // 2]
        self.assertLessEqual(conservative, median)
        self.assertGreaterEqual(conservative, min(spatials))

    def test_frequency_free_reserved_scores_are_on_the_lexicon_spatial_scale(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 0.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), -0.5),
        ]
        reserved = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (4,), 0), 2.0, frequency_free=True,
        )
        ranked = evaluator.rank_static_fusion_with_reserved(lexicon, [reserved])
        self.assertIn("cax", [entry.word for entry in ranked])
        self.assertEqual("cat", ranked[0].word)

    def test_lexicon_neighbors_are_edit_distance_variants_without_targets(self):
        lexicon = {
            "cat": evaluator.LexiconEntry("cat", "en", (1, 2, 3), 10),
            "car": evaluator.LexiconEntry("car", "en", (1, 2, 4), 9),
            "don't": evaluator.LexiconEntry("don't", "en", (1, 2, 3, 4), 8),
        }
        words = {entry.word for entry in evaluator.lexicon_neighbors("cot", lexicon)}
        self.assertIn("cat", words)
        self.assertIn("cat", {entry.word for entry in evaluator.lexicon_neighbors("act", lexicon)})
        self.assertNotIn("don't", words)
        contractions = {entry.word for entry in evaluator.lexicon_neighbors("dont", lexicon)}
        self.assertIn("don't", contractions)
        self.assertNotIn("target", inspect.signature(evaluator.lexicon_neighbors).parameters)

    def test_unconstrained_nbest_includes_the_greedy_spelling(self):
        output = logits([1, 2, 0, 2], classes=8, frames=8)
        greedy = evaluator.collapse_greedy(output)
        nbest = evaluator.unconstrained_ctc_emissions(output, n_best=4, beam_width=8)
        self.assertEqual(greedy, nbest[0][0])
        self.assertGreaterEqual(len(nbest), 1)
        greedy_only = evaluator.unconstrained_ctc_emissions(output, n_best=1, beam_width=8)
        self.assertEqual(1, len(greedy_only))
        self.assertEqual(greedy, greedy_only[0][0])

    def test_stratum_adaptive_budgets_do_not_reduce_short(self):
        self.assertEqual((32, 32), evaluator.decoder_slate_budgets({"short", "very_sloppy"}))
        self.assertEqual((32, 32), evaluator.decoder_slate_budgets({"short", "long", "sloppy"}))
        self.assertEqual((32, 8), evaluator.decoder_slate_budgets({"long", "very_sloppy"}))
        self.assertEqual((32, 8), evaluator.decoder_slate_budgets({"very_sloppy"}))
        self.assertEqual((32, 12), evaluator.decoder_slate_budgets({"sloppy", "medium"}))
        self.assertEqual((32, 32), evaluator.decoder_slate_budgets({"clean", "medium"}))

    def test_train_path_loader_does_not_return_targets(self):
        layout = {"id": "test-layout", "keyLabels": list("abc"), "keyMask": [1, 1, 1]}
        record = {
            "schemaVersion": 1,
            "sessionId": identifier("session"),
            "id": identifier("row"),
            "split": "train",
            "language": "en",
            "target": "secret-target",
            "ctcLabels": [1, 2],
            "pathCoordinates": [0.1] * 128,
            "layoutId": "test-layout",
            "orientation": "portrait",
            "geometricDeviation": 0.0,
            "strata": ["short"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "train.jsonl"
            path.write_text(json.dumps(record) + "\n")
            rows = evaluator.load_train_paths(path, layout, maximum_rows=1)
        self.assertEqual(1, len(rows))
        self.assertEqual(3, len(rows[0]))
        self.assertEqual(identifier("row"), rows[0][0])
        self.assertEqual("en", rows[0][1])
        self.assertEqual(128, len(rows[0][2]))
        self.assertTrue(all("secret-target" not in str(item) for item in rows[0]))

    def test_publish_reserved_slots_displaces_inside_the_31_ranking_bound(self):
        base = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), -0.1 * index)
            for index in range(32)
        ]
        reserved = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry("oovword", "en", (2, 3), 0), -9.0, frequency_free=True,
            ),
            evaluator.ScoredLexiconEntry(base[0].entry, 9.0, frequency_free=True),
        ]
        published = evaluator.publish_reserved_slots(base, reserved, reserved_budget=1)
        self.assertEqual(31, len(published))
        self.assertEqual([candidate.word for candidate in base[:30]], [candidate.word for candidate in published[:30]])
        self.assertEqual("oovword", published[30].word)
        self.assertNotIn("target", inspect.signature(evaluator.publish_reserved_slots).parameters)

    def test_publish_reserved_budget11_stays_inside_31_without_floor_clamp(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index)
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        extras = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(12)
        ]
        reserved = [greedy, *extras]
        occupants = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=11)
        self.assertEqual(11, len(occupants))
        self.assertEqual("cax", occupants[0].word)
        self.assertIn("wa9", [item.word for item in occupants])
        published = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=11)
        self.assertEqual(31, len(published))
        self.assertEqual(11, sum(1 for item in published if item.frequency_free))
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published, reserved, lexicon_reference=lexicon, park_min_rank=0,
            )
        ]
        self.assertEqual(31, len(ranked))
        self.assertEqual("cax", ranked[0])
        self.assertIn("wa9", ranked)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.publish_reserved_slots).parameters)

    def test_spatial_budget11_cannot_seat_oov_ctc_rank_past_ten_extras(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index,
            )
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        extra_oov = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(10)
        ]
        leftover = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (70,), 0), 6.9, frequency_free=True, source="greedy_alts",
        )
        reserved = [greedy, *extra_oov, leftover]
        ranks = evaluator.reserved_oov_ctc_ranks(reserved)
        self.assertEqual(11, ranks[("caz", "en")])
        self.assertTrue(evaluator.reserved_clears_lexicon_top3(leftover, lexicon))
        occupants = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=11)
        words = [item.word for item in occupants]
        self.assertEqual(11, len(occupants))
        self.assertEqual("cax", words[0])
        self.assertNotIn("caz", words)
        published = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=11)
        self.assertEqual(31, len(published))
        self.assertNotIn("caz", [item.word for item in published])
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published, reserved, lexicon_reference=lexicon, park_min_rank=0,
            )
        ]
        self.assertEqual("cax", ranked[0])
        self.assertNotIn("caz", ranked)
        self.assertNotIn("target", inspect.signature(evaluator.prioritize_reserved_candidates).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)
        occupants12 = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=12)
        words12 = [item.word for item in occupants12]
        self.assertEqual(12, len(occupants12))
        self.assertEqual("cax", words12[0])
        self.assertIn("caz", words12)
        published12 = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=12)
        self.assertEqual(31, len(published12))
        self.assertIn("caz", [item.word for item in published12])
        ranked12 = [
            entry.word
            for entry in evaluator.published_ranking(
                published12, reserved, lexicon_reference=lexicon, park_min_rank=0,
            )
        ]
        self.assertEqual(31, len(ranked12))
        self.assertIn("caz", ranked12)
        self.assertEqual("cax", ranked12[0])

    def test_unblend_greedy_alts_only_when_greedy_misses_top3(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ] + [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"zz{index}", "en", (10 + index,), 1), -1.0,
            )
            for index in range(28)
        ]
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0),
            1.2,
            frequency_free=True,
            source="greedy_alts",
            oov_map_blend=0.5,
        )
        quieter_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (7,), 0),
            0.8,
            frequency_free=True,
            source="greedy_alts",
            oov_map_blend=0.5,
        )
        converting_greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        reserved_hit = [converting_greedy, converting_alt, quieter_alt]
        published_hit = evaluator.publish_reserved_slots(lexicon, reserved_hit, reserved_budget=11)
        held = evaluator.unblend_greedy_alts_when_greedy_misses_top3(
            published_hit, reserved_hit, lexicon,
        )
        ranked_hit = [
            entry.word
            for entry in evaluator.published_ranking(held, reserved_hit, lexicon_reference=lexicon)
        ]
        self.assertEqual("cax", ranked_hit[0])
        self.assertNotIn("caz", ranked_hit[:3])
        self.assertEqual(0.5, next(item for item in held if item.word == "caz").oov_map_blend)
        missing_greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), -3.0, frequency_free=True, source="greedy",
        )
        reserved_miss = [missing_greedy, converting_alt, quieter_alt]
        published_miss = evaluator.publish_reserved_slots(lexicon, reserved_miss, reserved_budget=11)
        unblended = evaluator.unblend_greedy_alts_when_greedy_misses_top3(
            published_miss, reserved_miss, lexicon,
        )
        caz = next(item for item in unblended if item.word == "caz")
        caa = next(item for item in unblended if item.word == "caa")
        self.assertEqual(0.0, caz.oov_map_blend)
        self.assertGreater(caz.spatial, converting_alt.spatial)
        self.assertEqual(0.5, caa.oov_map_blend)
        ranked_miss = [
            entry.word
            for entry in evaluator.published_ranking(unblended, reserved_miss, lexicon_reference=lexicon)
        ]
        self.assertIn("caz", ranked_miss[:3])
        self.assertNotIn("caa", ranked_miss[:3])
        self.assertEqual(1, sum(1 for word in ranked_miss[:3] if word in {"cax", "caz", "caa"}))
        self.assertNotIn("target", inspect.signature(evaluator.unblend_greedy_alts_when_greedy_misses_top3).parameters)

    def test_prefer_converting_budget11_seats_window_losing_alt_and_truncated(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index,
            )
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(12)
        ]
        nbest = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (80 + index,), 0),
                9.0 - 0.02 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(4)
        ]
        truncated = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 0.2, frequency_free=False, source="truncated_ctc",
        )
        reserved = [greedy, *converting, *nbest, truncated]
        self.assertGreater(evaluator.reserved_oov_ctc_ranks(reserved)[("wa8", "en")], 7)
        self.assertTrue(evaluator.reserved_clears_lexicon_top3(converting[8], lexicon))
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=11)
        self.assertNotIn("wa8", [item.word for item in spatial[:8]])
        preferred = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=11, prefer_converting_greedy_alts=True,
        )
        self.assertEqual(11, len(preferred))
        self.assertEqual("cax", preferred[0].word)
        self.assertIn("wa8", [item.word for item in preferred])
        self.assertEqual(31, len(evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=11, prefer_converting_greedy_alts=True,
        )))
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                evaluator.publish_reserved_slots(
                    lexicon, reserved, reserved_budget=11, prefer_converting_greedy_alts=True,
                ),
                reserved,
                lexicon_reference=lexicon,
                park_min_rank=0,
            )
        ]
        self.assertEqual("cax", ranked[0])
        self.assertEqual(31, len(ranked))
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)

    def test_lift_near_top3_unblends_only_rank_4_to_6_frequency_free(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("the", "en", (1,), 200), 3.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("and", "en", (1,), 180), 2.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("for", "en", (1,), 160), 2.2),
        ] + [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), 0.5 - 0.05 * index)
            for index in range(27)
        ]
        oov = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0),
            1.2,
            frequency_free=True,
            source="greedy",
            oov_map_blend=0.5,
        )
        published = [*lexicon, oov]
        before = [
            entry.word
            for entry in evaluator.published_ranking(published, [oov], lexicon_reference=lexicon)
        ]
        self.assertEqual(4, before.index("cax") + 1)
        lifted = evaluator.lift_near_top3_frequency_free(published, [oov], lexicon, blend=0.5)
        self.assertEqual({item.word for item in published}, {item.word for item in lifted})
        lifted_cax = next(item for item in lifted if item.word == "cax")
        self.assertGreater(lifted_cax.spatial, oov.spatial)
        self.assertEqual(0.0, lifted_cax.oov_map_blend)
        unique = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0),
            1.2,
            frequency_free=True,
            source="greedy",
            oov_map_blend=0.0,
        )
        published_unique = [*lexicon, unique]
        lifted_unique = evaluator.lift_near_top3_frequency_free(
            published_unique, [unique], lexicon, blend=0.5,
        )
        self.assertEqual(1.2, lifted_unique[-1].spatial)
        weak = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (3,), 0),
            -4.0,
            frequency_free=True,
            source="greedy_alts",
            oov_map_blend=0.5,
        )
        published_weak = [*lexicon, weak]
        before_weak = [
            entry.word
            for entry in evaluator.published_ranking(published_weak, [weak], lexicon_reference=lexicon)
        ]
        lifted_weak = evaluator.lift_near_top3_frequency_free(published_weak, [weak], lexicon, blend=0.5)
        after_weak = [
            entry.word
            for entry in evaluator.published_ranking(lifted_weak, [weak], lexicon_reference=lexicon)
        ]
        self.assertEqual(before_weak.index("caz"), after_weak.index("caz"))
        self.assertNotIn("caz", after_weak[:3])
        self.assertNotIn("target", inspect.signature(evaluator.lift_near_top3_frequency_free).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.published_ranking_scored).parameters)
        offset = 0.12
        unique_offset = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0),
            1.2,
            frequency_free=True,
            source="greedy",
            oov_map_blend=0.0,
        )
        published_offset = [*lexicon, unique_offset]
        lifted_offset = evaluator.lift_near_top3_frequency_free(
            published_offset, [unique_offset], lexicon, blend=0.0, optimism_offset=offset,
        )
        lifted_offset_cax = next(item for item in lifted_offset if item.word == "cax")
        self.assertAlmostEqual(1.2 + offset, lifted_offset_cax.spatial)
        far = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (3,), 0),
            -4.0,
            frequency_free=True,
            source="greedy",
            oov_map_blend=0.0,
        )
        published_far = [*lexicon, far]
        lifted_far = evaluator.lift_near_top3_frequency_free(
            published_far, [far], lexicon, blend=0.0, optimism_offset=offset,
        )
        self.assertEqual(-4.0, next(item for item in lifted_far if item.word == "caz").spatial)
        alt_near = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cay", "en", (4,), 0),
            1.2,
            frequency_free=True,
            source="greedy_alts",
            oov_map_blend=0.0,
        )
        published_alt = [*lexicon, alt_near]
        lifted_alt = evaluator.lift_near_top3_frequency_free(
            published_alt, [alt_near], lexicon, blend=0.0, optimism_offset=offset,
        )
        self.assertEqual(1.2, next(item for item in lifted_alt if item.word == "cay").spatial)
        mid = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (5,), 0),
            0.2,
            frequency_free=True,
            source="greedy",
            oov_map_blend=0.0,
        )
        published_mid = [*lexicon, mid]
        before_mid = [
            entry.word
            for entry in evaluator.published_ranking(published_mid, [mid], lexicon_reference=lexicon)
        ]
        rank_mid = before_mid.index("caa") + 1
        self.assertGreaterEqual(rank_mid, 7)
        self.assertLessEqual(rank_mid, 10)
        no_expand = evaluator.lift_near_top3_frequency_free(
            published_mid, [mid], lexicon, blend=0.0, optimism_offset=offset, max_rank=6,
        )
        self.assertEqual(0.2, next(item for item in no_expand if item.word == "caa").spatial)
        expanded = evaluator.lift_near_top3_frequency_free(
            published_mid, [mid], lexicon, blend=0.0, optimism_offset=offset, max_rank=10,
        )
        self.assertAlmostEqual(0.2 + offset, next(item for item in expanded if item.word == "caa").spatial)
        self.assertNotIn("target", inspect.signature(evaluator.lift_near_top3_frequency_free).parameters)
        extra = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("ex0", "en", (10,), 0),
            8.0,
            frequency_free=True,
            source="greedy_alts",
        )
        parked_greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0),
            1.2,
            frequency_free=True,
            source="greedy",
            oov_map_blend=0.0,
        )
        published_park = [*lexicon, parked_greedy, extra]
        extra_keys = {("ex0", "en")}
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published_park,
                [parked_greedy, extra],
                lexicon_reference=lexicon,
                extra_park_keys=extra_keys,
                park_min_rank=4,
            )
        ]
        self.assertGreaterEqual(parked.index("cax") + 1, 4)
        self.assertLessEqual(parked.index("cax") + 1, 6)
        parked_lift = evaluator.lift_near_top3_frequency_free(
            published_park,
            [parked_greedy, extra],
            lexicon,
            blend=0.0,
            optimism_offset=offset,
            extra_park_keys=extra_keys,
            park_min_rank=4,
        )
        self.assertAlmostEqual(
            1.2 + offset, next(item for item in parked_lift if item.word == "cax").spatial,
        )
        self.assertEqual(8.0, next(item for item in parked_lift if item.word == "ex0").spatial)
        self.assertNotIn("target", inspect.signature(evaluator.lift_near_top3_frequency_free).parameters)

    def test_reserved_fill_ranks_order_greedy_then_spatial_oov(self):
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0), -0.5, frequency_free=True, source="greedy",
        )
        strong_nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (4,), 0), -0.1, frequency_free=True, source="nbest",
        )
        extra_alts = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"ca{index}", "en", (5,), 0),
                -0.2 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(6)
        ]
        weak_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (3,), 0), -2.0, frequency_free=True, source="greedy_alts",
        )
        neighbors = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (1,), 50 - index),
                0.1,
                frequency_free=False,
                source="neighbors",
            )
            for index in range(4)
        ]
        reserved = [weak_alt, *neighbors, strong_nbest, greedy, *extra_alts]
        fill = evaluator.reserved_fill_ranks(reserved)
        oov = evaluator.reserved_oov_ctc_ranks(reserved)
        self.assertEqual(1, fill[("cax", "en")])
        self.assertEqual(2, fill[("caa", "en")])
        self.assertGreater(fill[("caz", "en")], 11)
        self.assertEqual(1, oov[("caa", "en")])
        self.assertEqual(8, oov[("caz", "en")])
        self.assertNotIn(("cax", "en"), oov)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_fill_ranks).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_oov_ctc_ranks).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.prioritize_reserved_candidates).parameters)

    def test_reserved_occupants_skip_lexicon_duplicates_and_keep_greedy_first(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("the", "en", (1,), 200), 3.0),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0), -4.0, frequency_free=True, source="greedy",
        )
        alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (3,), 0), 8.0, frequency_free=True, source="greedy_alts",
        )
        duplicate = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("the", "en", (1,), 200), 9.0, frequency_free=False, source="greedy",
        )
        occupants = evaluator.reserved_occupants(
            [duplicate, greedy, alt], lexicon, reserved_budget=1, skip_keys={("the", "en")},
        )
        self.assertEqual(["cax"], [item.word for item in occupants])
        two = evaluator.reserved_occupants(
            [greedy, alt], lexicon, reserved_budget=2, skip_keys=(),
        )
        self.assertEqual(["cax", "caz"], [item.word for item in two])
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)

    def test_ablation_extra_fill_seats_neighbors_before_extra_oov_tail(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        louder_nbest = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (7 + index,), 0),
                9.5 - 0.1 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(7)
        ]
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 0.2, frequency_free=False, source="neighbors",
        )
        trunc = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cae", "en", (9,), 40), 0.1, frequency_free=False, source="truncated_geometry",
        )
        alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 7.5, frequency_free=True, source="greedy_alts",
        )
        reserved = [greedy, alt, neighbor, trunc, *louder_nbest]
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", spatial[0].word)
        self.assertNotIn("cad", [item.word for item in spatial])
        ablation = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, ablation_extra_fill_sources=True,
        )
        words = [item.word for item in ablation]
        self.assertEqual("cax", words[0])
        self.assertIn("cad", words)
        self.assertIn("cae", words)
        self.assertIn("caz", words)
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, ablation_extra_fill_sources=True,
        )
        self.assertIn(("cad", "en"), extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, ablation_extra_fill_sources=True,
        )
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", ranked[0])
        self.assertIn("cad", ranked)
        self.assertNotIn("caz", ranked[:3])
        self.assertNotIn("target", inspect.signature(evaluator.ablation_extra_fill).parameters)

    def test_prefer_converting_greedy_alts_displace_nonconverting_extras(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 4.0, frequency_free=True, source="greedy_alts",
        )
        louder_nbest = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (7 + index,), 0),
                9.5 - 0.05 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(8)
        ]
        reserved = [greedy, converting_alt, *louder_nbest]
        self.assertGreater(evaluator.reserved_oov_ctc_ranks(reserved)[("caz", "en")], 7)
        spatial = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=2,
        )
        self.assertEqual(["cax", "n0"], [item.word for item in spatial])
        preferred = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=2, prefer_converting_greedy_alts=True,
        )
        self.assertEqual(["cax", "caz"], [item.word for item in preferred])
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=2, prefer_converting_greedy_alts=True,
        )
        self.assertEqual({("caz", "en")}, extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=2, prefer_converting_greedy_alts=True,
        )
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertNotIn("caz", parked[:3])
        self.assertIn("caz", parked)
        self.assertNotIn("n0", parked[:3])
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.window_losing_converting_greedy_alts).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.prefer_window_losing_converting_alts).parameters)

    def test_prefer_converting_alts_on_budget8_seat_oov_rank11_and_park_extras(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 4.0, frequency_free=True, source="greedy_alts",
        )
        louder = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (10 + index,), 0),
                9.5 - 0.05 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(10)
        ]
        reserved = [greedy, converting_alt, *louder]
        oov_rank = evaluator.reserved_oov_ctc_ranks(reserved)
        self.assertGreater(oov_rank[("caz", "en")], 7)
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertNotIn("caz", [item.word for item in spatial])
        preferred = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, prefer_converting_greedy_alts=True,
        )
        self.assertEqual("cax", preferred[0].word)
        self.assertIn("caz", [item.word for item in preferred])
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, prefer_converting_greedy_alts=True,
        )
        self.assertIn(("caz", "en"), extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, prefer_converting_greedy_alts=True,
        )
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertIn("caz", parked)
        self.assertNotIn("caz", parked[:3])
        self.assertEqual(1, sum(1 for word in parked[:3] if word in {"cax", "caz"}))
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)

    def test_leftover_extra_seats_go_to_neighbors_after_converting_oov(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 7.0, frequency_free=True, source="greedy_alts",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 1.2, frequency_free=False, source="neighbors",
        )
        nbest = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (10 + index,), 0),
                -8.0 - 0.1 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(8)
        ]
        reserved = [greedy, converting_alt, neighbor, *nbest]
        spatial = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, leftover_inlex_fill=True,
        )
        self.assertEqual("cax", spatial[0].word)
        self.assertIn("cad", [item.word for item in spatial])
        self.assertIn("caz", [item.word for item in spatial])
        winners = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (20 + index,), 0),
                7.5 - 0.1 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(7)
        ]
        saturated = evaluator.reserved_occupants(
            [greedy, *winners, neighbor], lexicon, reserved_budget=8, leftover_inlex_fill=True,
        )
        self.assertEqual("cax", saturated[0].word)
        self.assertNotIn("cad", [item.word for item in saturated])
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, leftover_inlex_fill=True,
        )
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, leftover_inlex_fill=True,
        )
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertIn("cad", parked)
        self.assertEqual(1, sum(1 for word in parked[:3] if word in {"cax", "caz"}))
        self.assertNotIn("target", inspect.signature(evaluator.leftover_inlex_after_converting_oov).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)

    def test_converting_leftover_extras_skip_nonconverting_nbest_and_neighbors(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 7.0, frequency_free=True, source="greedy_alts",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 1.2, frequency_free=False, source="neighbors",
        )
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (7,), 0), -9.0, frequency_free=True, source="nbest",
        )
        reserved = [greedy, converting_alt, neighbor, nbest]
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertIn("caa", [item.word for item in spatial])
        converting = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, converting_leftover_extras_fill=True,
        )
        self.assertEqual(["cax", "caz"], [item.word for item in converting])
        self.assertNotIn("cad", [item.word for item in converting])
        self.assertNotIn("caa", [item.word for item in converting])
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, converting_leftover_extras_fill=True,
        )
        self.assertEqual({("caz", "en")}, extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, converting_leftover_extras_fill=True,
        )
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertIn("caz", parked)
        self.assertNotIn("caz", parked[:3])
        self.assertEqual(1, sum(1 for word in parked[:3] if word in {"cax", "caz"}))
        self.assertNotIn("target", inspect.signature(evaluator.converting_leftover_extras).parameters)

    def test_converting_alt_expand_cannot_evict_frozen_lexicon_ranks(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index)
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(10)
        ]
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (70,), 0), 9.0, frequency_free=True, source="nbest",
        )
        reserved = [greedy, *converting, nbest]
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual(8, len(spatial))
        self.assertIn("caa", [item.word for item in spatial])
        expanded = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, converting_alt_expand=True,
        )
        self.assertEqual(8, len(expanded))
        self.assertEqual("cax", expanded[0].word)
        self.assertNotIn("wa8", [item.word for item in expanded])
        self.assertNotIn("caa", [item.word for item in expanded])
        self.assertIn("wa0", [item.word for item in expanded])
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, converting_alt_expand=True,
        )
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, converting_alt_expand=True,
        )
        self.assertEqual(31, len(published))
        lexicon_kept = [item.word for item in published if not item.frequency_free]
        self.assertGreaterEqual(len(lexicon_kept), 23)
        self.assertIn("lex0", {item.word for item in published})
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertNotIn("wa8", parked)
        self.assertEqual(1, sum(1 for word in parked[:3] if word == "cax" or word.startswith("wa")))
        self.assertNotIn("target", inspect.signature(evaluator.converting_alt_expand_fill).parameters)

    def test_converting_fill_loss_append_keeps_nbest_and_parks_extras(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index,
            )
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(10)
        ]
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (70,), 0), 9.0, frequency_free=True, source="nbest",
        )
        reserved = [greedy, *converting, nbest]
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", spatial[0].word)
        self.assertIn("caa", [item.word for item in spatial])
        self.assertNotIn("wa8", [item.word for item in spatial])
        appended = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        words = [item.word for item in appended]
        self.assertEqual("cax", words[0])
        self.assertIn("caa", words)
        self.assertLess(words.index("caa"), 8)
        self.assertIn("wa8", words)
        self.assertGreater(len(appended), 8)
        self.assertLessEqual(len(appended), 11)
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        self.assertIn(("wa8", "en"), extras)
        self.assertIn(("caa", "en"), extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, converting_fill_loss_append=True,
        )
        self.assertEqual(31, len(published))
        lexicon_kept = [item.word for item in published if not item.frequency_free]
        self.assertGreaterEqual(len(lexicon_kept), 20)
        self.assertIn("caa", [item.word for item in published])
        self.assertIn("wa8", [item.word for item in published])
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertNotIn("wa8", parked[:3])
        self.assertNotIn("caa", parked[:3])
        self.assertEqual(1, sum(1 for word in parked[:3] if word == "cax" or word.startswith("wa") or word == "caa"))
        self.assertNotIn("target", inspect.signature(evaluator.converting_fill_loss_append_fill).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)

    def test_converting_fill_loss_append_unparked_keeps_bound_when_greedy_misses(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ] + [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"zz{index}", "en", (10 + index,), 1), -1.0,
            )
            for index in range(28)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), -3.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (20 + index,), 0),
                1.2 - 0.02 * index,
                frequency_free=True,
                source="greedy_alts",
                oov_map_blend=0.5,
            )
            for index in range(10)
        ]
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (70,), 0), 0.4, frequency_free=True, source="nbest",
            oov_map_blend=0.5,
        )
        reserved = [greedy, *converting, nbest]
        appended = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        self.assertGreater(len(appended), 8)
        self.assertLessEqual(len(appended), 11)
        self.assertIn("wa8", [item.word for item in appended])
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, converting_fill_loss_append=True,
        )
        self.assertEqual(31, len(published))
        unblended = evaluator.unblend_greedy_alts_when_greedy_misses_top3(
            published, reserved, lexicon,
        )
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                unblended, reserved, lexicon_reference=lexicon, park_min_rank=0,
            )
        ]
        self.assertEqual(31, len(ranked))
        self.assertEqual(1, sum(1 for word in ranked[:3] if word == "cax" or word.startswith("wa") or word == "caa"))
        self.assertNotIn("target", inspect.signature(evaluator.converting_fill_loss_append_fill).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.unblend_greedy_alts_when_greedy_misses_top3).parameters)

    def test_converting_fill_loss_append_displaces_nonconverting_extra_oov_nbest(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 5.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 4.9),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 4.8),
        ] + [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"zz{index}", "en", (100 + index,), 1), 0.0,
            )
            for index in range(28)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 13.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (20 + index,), 0),
                12.0 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(10)
        ]
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (70,), 0),
            14.0,
            frequency_free=True,
            source="nbest",
            oov_map_blend=1.0,
        )
        reserved = [greedy, *converting, nbest]
        self.assertFalse(evaluator.reserved_clears_lexicon_top3(nbest, lexicon))
        self.assertTrue(evaluator.reserved_clears_lexicon_top3(converting[0], lexicon))
        self.assertTrue(evaluator.reserved_clears_lexicon_top3(converting[8], lexicon))
        ranks = evaluator.reserved_oov_ctc_ranks(reserved)
        self.assertLessEqual(ranks[("caa", "en")], 7)
        self.assertGreater(ranks[("wa8", "en")], 7)
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", spatial[0].word)
        self.assertIn("caa", [item.word for item in spatial])
        self.assertIn("wa0", [item.word for item in spatial])
        self.assertNotIn("wa8", [item.word for item in spatial])
        appended = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        words = [item.word for item in appended]
        self.assertEqual("cax", words[0])
        self.assertNotIn("caa", words[:8])
        self.assertIn("wa0", words[:8])
        self.assertIn("wa8", words)
        self.assertGreater(len(appended), 8)
        self.assertLessEqual(len(appended), 11)
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        self.assertIn(("wa8", "en"), extras)
        self.assertIn(("wa0", "en"), extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, converting_fill_loss_append=True,
        )
        self.assertEqual(31, len(published))
        lexicon_kept = [item.word for item in published if not item.frequency_free]
        self.assertGreaterEqual(len(lexicon_kept), 20)
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertNotIn("wa8", parked[:3])
        self.assertNotIn("caa", parked[:3])
        self.assertEqual(1, sum(1 for word in parked[:3] if word == "cax" or word.startswith("wa") or word == "caa"))
        self.assertNotIn("target", inspect.signature(evaluator.prefer_window_losing_converting_alts).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.converting_fill_loss_append_fill).parameters)

    def test_converting_fill_loss_append_seats_converting_greedy_neighbors_before_leftover_alts(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index,
            )
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(10)
        ]
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (70,), 0), 9.0, frequency_free=True, source="nbest",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 7.0, frequency_free=False, source="neighbors",
        )
        reserved = [greedy, *converting, nbest, neighbor]
        self.assertTrue(evaluator.reserved_clears_lexicon_top3(neighbor, lexicon))
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertNotIn("cad", [item.word for item in spatial])
        self.assertIn("caa", [item.word for item in spatial])
        appended = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        words = [item.word for item in appended]
        self.assertEqual("cax", words[0])
        self.assertIn("caa", words[:8])
        self.assertIn("wa6", words)
        self.assertIn("wa8", words)
        self.assertLess(words.index("wa6"), words.index("cad") if "cad" in words else 11)
        self.assertLessEqual(len(appended), 11)
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, converting_fill_loss_append=True,
        )
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, converting_fill_loss_append=True,
        )
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", ranked[0])
        self.assertIn("wa8", ranked)
        self.assertNotIn("caa", ranked[:3])
        self.assertNotIn("wa8", ranked[:3])
        self.assertEqual(31, len(ranked))
        self.assertNotIn("target", inspect.signature(evaluator.leftover_converting_greedy_neighbors).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.converting_fill_loss_append_fill).parameters)

    def test_leftover_greedy_alts_append_seats_unpublished_alts_without_parking(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index,
            )
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(12)
        ]
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (70,), 0), 9.0, frequency_free=True, source="nbest",
        )
        reserved = [greedy, *converting, nbest]
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", spatial[0].word)
        self.assertIn("caa", [item.word for item in spatial])
        self.assertNotIn("wa8", [item.word for item in spatial])
        appended = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, leftover_greedy_alts_append=True,
        )
        words = [item.word for item in appended]
        self.assertEqual("cax", words[0])
        self.assertIn("caa", words[:8])
        self.assertIn("wa8", words)
        self.assertGreater(len(appended), 8)
        self.assertLessEqual(len(appended), 28)
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, leftover_greedy_alts_append=True,
        )
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, leftover_greedy_alts_append=True,
        )
        self.assertEqual(31, len(published))
        self.assertIn("wa8", [item.word for item in published])
        unparked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=0,
            )
        ]
        self.assertIn("wa0", unparked[:3] or unparked)
        self.assertNotIn("target", inspect.signature(evaluator.leftover_greedy_alts_append_fill).parameters)

    def test_ablation_first_source_fill_seats_length_changing_alts_then_neighbors(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        same_len = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 7.5, frequency_free=True, source="greedy_alts",
        )
        diff_len = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caxt", "en", (7,), 0), 4.0, frequency_free=True, source="greedy_alts",
        )
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (8,), 0), 9.0, frequency_free=True, source="nbest",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (9,), 80), 1.2, frequency_free=False, source="neighbors",
        )
        reserved = [greedy, same_len, diff_len, nbest, neighbor]
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", spatial[0].word)
        self.assertIn("caa", [item.word for item in spatial])
        filled = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, ablation_first_source_fill_sources=True,
        )
        words = [item.word for item in filled]
        self.assertEqual("cax", words[0])
        self.assertLess(words.index("caxt"), words.index("caz"))
        self.assertIn("cad", words)
        self.assertLess(words.index("cad"), words.index("caa"))
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8, ablation_first_source_fill_sources=True,
        )
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8, ablation_first_source_fill_sources=True,
        )
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertNotIn("caz", parked[:3])
        self.assertNotIn("caxt", parked[:3])
        self.assertEqual(1, sum(1 for word in parked[:3] if word in {"cax", "caz", "caxt", "caa"}))
        self.assertNotIn("target", inspect.signature(evaluator.ablation_first_source_fill).parameters)

    def test_ablation_first_source_budget11_displaces_nbest_with_leftover_alts(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"lex{index}", "en", (index,), 50 - index), 2.0 - 0.05 * index,
            )
            for index in range(31)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (40,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (50 + index,), 0),
                7.5 - 0.05 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(10)
        ]
        nbest = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (80 + index,), 0),
                9.5 - 0.02 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(5)
        ]
        truncated = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 0.2, frequency_free=False, source="truncated_ctc",
        )
        reserved = [greedy, *converting, *nbest, truncated]
        self.assertGreater(evaluator.reserved_oov_ctc_ranks(reserved)[("wa8", "en")], 7)
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=11)
        self.assertIn("n0", [item.word for item in spatial])
        self.assertNotIn("wa8", [item.word for item in spatial])
        filled = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=11, ablation_first_source_fill_sources=True,
        )
        words = [item.word for item in filled]
        self.assertEqual(11, len(filled))
        self.assertEqual("cax", words[0])
        self.assertIn("wa8", words)
        self.assertIn("cad", words)
        self.assertLess(words.index("wa8"), words.index("n0") if "n0" in words else 11)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=11, ablation_first_source_fill_sources=True,
        )
        self.assertEqual(31, len(published))
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published, reserved, lexicon_reference=lexicon, park_min_rank=0,
            )
        ]
        self.assertEqual("cax", ranked[0])
        self.assertEqual(31, len(ranked))
        self.assertNotIn("target", inspect.signature(evaluator.ablation_first_source_fill).parameters)

    def test_window_losing_converting_alts_do_not_displace_converting_extra_oov_winners(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        winners = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"wa{index}", "en", (20 + index,), 0),
                7.5 - 0.1 * index,
                frequency_free=True,
                source="greedy_alts",
            )
            for index in range(7)
        ]
        loser = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 3.0, frequency_free=True, source="greedy_alts",
        )
        reserved = [greedy, *winners, loser]
        ranks = evaluator.reserved_oov_ctc_ranks(reserved)
        self.assertLessEqual(ranks[("wa0", "en")], 7)
        self.assertGreater(ranks[("caz", "en")], 7)
        spatial = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", spatial[0].word)
        self.assertNotIn("caz", [item.word for item in spatial])
        preferred = evaluator.reserved_occupants(
            reserved, lexicon, reserved_budget=8, prefer_converting_greedy_alts=True,
        )
        self.assertEqual([item.word for item in spatial], [item.word for item in preferred])
        self.assertNotIn("caz", [item.word for item in preferred])
        self.assertNotIn("target", inspect.signature(evaluator.prefer_window_losing_converting_alts).parameters)

    def test_budget8_spatial_publishes_fill_rank5_converting_alt_outside_top3(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 7.5, frequency_free=True, source="greedy_alts",
        )
        louder = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"n{index}", "en", (7 + index,), 0),
                9.5 - 0.1 * index,
                frequency_free=True,
                source="nbest",
            )
            for index in range(3)
        ]
        reserved = [greedy, converting_alt, *louder]
        fill = evaluator.reserved_fill_ranks(reserved)
        self.assertEqual(1, fill[("cax", "en")])
        self.assertEqual(5, fill[("caz", "en")])
        four = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=4)
        self.assertNotIn("caz", [item.word for item in four])
        eight = evaluator.reserved_occupants(reserved, lexicon, reserved_budget=8)
        self.assertEqual("cax", eight[0].word)
        self.assertIn("caz", [item.word for item in eight])
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=8,
        )
        self.assertIn(("caz", "en"), extras)
        published = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=8,
        )
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertIn("caz", parked)
        self.assertNotIn("caz", parked[:3])
        self.assertEqual(1, sum(1 for word in parked[:3] if word == "cax"))
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_fill_ranks).parameters)

    def test_converting_best_occupant_is_highest_ols_clearing_oov(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (4,), 0), 8.0, frequency_free=True, source="greedy",
        )
        stronger_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (5,), 0), 9.0, frequency_free=True, source="greedy_alts",
        )
        weaker_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (6,), 0), 8.5, frequency_free=True, source="greedy_alts",
        )
        parked_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cab", "en", (7,), 0), -9.0, frequency_free=True, source="greedy_alts",
        )
        greedy_first = evaluator.reserved_occupants(
            [greedy, stronger_alt], lexicon, reserved_budget=1,
        )
        self.assertEqual(["cax"], [item.word for item in greedy_first])
        best = evaluator.reserved_occupants(
            [greedy, parked_alt, stronger_alt, weaker_alt],
            lexicon,
            reserved_budget=1,
            converting_best=True,
        )
        self.assertEqual(["caz"], [item.word for item in best])
        keep_greedy = evaluator.reserved_occupants(
            [greedy, parked_alt], lexicon, reserved_budget=1, converting_best=True,
        )
        self.assertEqual(["cax"], [item.word for item in keep_greedy])
        none_convert = evaluator.reserved_occupants(
            [parked_alt], lexicon, reserved_budget=1, converting_best=True,
        )
        self.assertEqual(["cab"], [item.word for item in none_convert])
        published = evaluator.publish_reserved_slots(
            lexicon, [greedy, stronger_alt], reserved_budget=1, converting_best=True,
        )
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(published, [greedy, stronger_alt], lexicon_reference=lexicon)
        ]
        self.assertEqual("caz", ranked[0])
        self.assertNotIn("cax", ranked[:3])
        self.assertNotIn("target", inspect.signature(evaluator.converting_best_reserved).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.ols_reserved_candidate).parameters)

    def test_park_extra_reserved_keeps_greedy_in_top3_and_demotes_alts(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 9.0, frequency_free=True, source="greedy_alts",
        )
        nbest = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caa", "en", (7,), 0), 8.5, frequency_free=True, source="nbest",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 7.0, frequency_free=False, source="neighbors",
        )
        reserved = [greedy, alt, nbest, neighbor]
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=2,
        )
        self.assertEqual({("caz", "en")}, extras)
        published = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=2)
        unparked = [
            entry.word
            for entry in evaluator.published_ranking(published, reserved, lexicon_reference=lexicon)
        ]
        self.assertEqual("caz", unparked[0])
        parked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked[0])
        self.assertNotIn("caz", parked[:3])
        self.assertGreaterEqual(parked.index("caz") + 1, 4)
        self.assertIn("caz", parked)
        already_low = evaluator.park_extra_reserved_below_rank(
            [(greedy, 0.1), (lexicon[0], 0.0), (lexicon[1], -0.1), (lexicon[2], -0.2), (alt, -1.0)],
            {("caz", "en")},
            min_rank=4,
        )
        self.assertEqual("caz", already_low[-1][0].word)
        self.assertNotIn("target", inspect.signature(evaluator.park_extra_reserved_below_rank).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.extra_reserved_occupant_keys).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_clears_lexicon_top3).parameters)
        extras4 = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=4,
        )
        self.assertEqual({("caz", "en"), ("caa", "en"), ("cad", "en")}, extras4)
        published4 = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=4)
        parked4 = [
            entry.word
            for entry in evaluator.published_ranking(
                published4,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras4,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", parked4[0])
        self.assertNotIn("caz", parked4[:3])
        self.assertNotIn("caa", parked4[:3])
        self.assertIn("caz", parked4)
        self.assertIn("caa", parked4)
        self.assertEqual(1, sum(1 for word in parked4[:3] if word in {"cax", "caz", "caa"}))
        extras_best = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=4, converting_best=True,
        )
        self.assertEqual({("cax", "en"), ("caa", "en"), ("cad", "en")}, extras_best)
        published_best = evaluator.publish_reserved_slots(
            lexicon, reserved, reserved_budget=4, converting_best=True,
        )
        parked_best = [
            entry.word
            for entry in evaluator.published_ranking(
                published_best,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras_best,
                park_min_rank=4,
            )
        ]
        self.assertEqual("caz", parked_best[0])
        self.assertNotIn("cax", parked_best[:3])
        self.assertNotIn("caa", parked_best[:3])
        self.assertIn("cax", parked_best)
        self.assertEqual(1, sum(1 for word in parked_best[:3] if word in {"cax", "caz", "caa"}))
        self.assertNotIn("target", inspect.signature(evaluator.reserved_occupants).parameters)

    def test_ols_extra_converts_when_greedy_does_not_and_inlex_is_not_parked(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cab", "en", (4,), 20), 0.5),
        ]
        weak_greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), -9.0, frequency_free=True, source="greedy",
        )
        converting_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 8.0, frequency_free=True, source="greedy_alts",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cad", "en", (8,), 80), 7.0, frequency_free=False, source="neighbors",
        )
        reserved = [weak_greedy, converting_alt, neighbor]
        extras = evaluator.extra_reserved_occupant_keys(
            reserved, lexicon, reserved_budget=2,
        )
        self.assertEqual({("caz", "en")}, extras)
        published = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=2)
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual("caz", ranked[0])
        self.assertNotIn("cax", ranked[:3])
        converting_greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        with_greedy = evaluator.publish_reserved_slots(
            lexicon, [converting_greedy, converting_alt, neighbor], reserved_budget=3,
        )
        extras3 = evaluator.extra_reserved_occupant_keys(
            [converting_greedy, converting_alt, neighbor], lexicon, reserved_budget=3,
        )
        ranked_greedy = [
            entry.word
            for entry in evaluator.published_ranking(
                with_greedy,
                [converting_greedy, converting_alt, neighbor],
                lexicon_reference=lexicon,
                extra_park_keys=extras3,
                park_min_rank=4,
            )
        ]
        self.assertEqual("cax", ranked_greedy[0])
        self.assertNotIn("caz", ranked_greedy[:3])
        self.assertIn("cad", ranked_greedy)
        self.assertNotIn("target", inspect.signature(evaluator.park_extra_frequency_free_to_protect_top3).parameters)

    def test_reserved_gap_to_lexicon_top3_does_not_take_targets(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ]
        strong = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (4,), 0), 8.0, frequency_free=True, source="greedy",
        )
        weak = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (5,), 0), -9.0, frequency_free=True, source="greedy_alts",
        )
        self.assertGreater(evaluator.reserved_gap_to_lexicon_top3(strong, lexicon), 0.0)
        self.assertLess(evaluator.reserved_gap_to_lexicon_top3(weak, lexicon), 0.0)
        dest = [2.0, 1.5, 1.0, -0.5]
        ols = 1.8
        conservative = evaluator.conservative_lexicon_spatial(dest)
        blended = 0.5 * ols + 0.5 * conservative
        self.assertAlmostEqual(ols, evaluator.unblend_oov_spatial(blended, dest, 0.5))
        self.assertNotIn("target", inspect.signature(evaluator.reserved_gap_to_lexicon_top3).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.unblend_oov_spatial).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.lexicon_top3_fusion_floor).parameters)

    def test_in_lexicon_reserved_spatial_is_on_the_merged_scale(self):
        calibration = evaluator.fit_oov_score_calibration([-0.40, -0.10], [-0.40, -0.10])
        neighbor = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("cat", "en", (1, 2, 3), 80), "neighbors",
        )
        merged_scale = [2.0, 0.5, -0.5, -1.0]
        scored = evaluator.score_reserved_candidate(
            neighbor,
            calibration=calibration,
            ctc_spatials=[-0.8, -0.2],
            lexicon_spatials=merged_scale,
        )
        self.assertFalse(scored.frequency_free)
        self.assertEqual(evaluator.median_lexicon_spatial(merged_scale), scored.spatial)
        self.assertEqual(80, scored.entry.frequency)

    def test_in_lexicon_neighbor_fusion_uses_real_frequency_and_is_not_parked(self):
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("rare", "en", (1,), 1), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("mid", "en", (2,), 10), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("low", "en", (3,), 5), 1.0),
        ]
        calibration = evaluator.fit_oov_score_calibration([-0.40, -0.10], [-0.40, -0.10])
        neighbor = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("the", "en", (4,), 10_000), "neighbors", decoder_spatial=1.5,
        )
        scored = evaluator.score_reserved_candidate(
            neighbor,
            calibration=calibration,
            ctc_spatials=[-0.8, -0.2],
            lexicon_spatials=[2.0, 1.5, 1.0],
        )
        self.assertFalse(scored.frequency_free)
        self.assertEqual(10_000, scored.entry.frequency)
        oov = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), scored.spatial, frequency_free=True, source="greedy",
        )
        in_lex_fusion, oov_fusion = evaluator.reserved_fusion_values(lexicon, [scored, oov])
        self.assertGreater(in_lex_fusion, oov_fusion)
        published = evaluator.publish_reserved_slots(lexicon, [scored, oov], reserved_budget=2)
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published, [scored, oov], lexicon_reference=lexicon, park_min_rank=0,
            )
        ]
        self.assertEqual("the", ranked[0])
        self.assertNotIn("target", inspect.signature(evaluator.score_reserved_candidate).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.reserved_fusion_values).parameters)

    def test_greedy_oov_keeps_full_ols_while_alts_keep_the_blend(self):
        greedy = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("cax", "en", (1,), 0), "greedy", forward=-0.05,
        )
        alt = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("caz", "en", (2,), 0), "greedy_alts", forward=-0.40,
        )
        blends = evaluator._oov_blend_by_margin([greedy, alt], 0.5)
        self.assertEqual(0.0, blends[id(greedy)])
        self.assertEqual(0.5, blends[id(alt)])
        close = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("cay", "en", (3,), 0), "greedy_alts", forward=-0.06,
        )
        tied = evaluator._oov_blend_by_margin([greedy, close], 0.5)
        self.assertEqual(0.0, tied[id(greedy)])
        self.assertEqual(0.5, tied[id(close)])
        nbest = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("caa", "en", (4,), 0), "nbest", forward=-0.01,
        )
        vs_nbest = evaluator._oov_blend_by_margin([greedy, alt, nbest], 0.5)
        self.assertEqual(0.0, vs_nbest[id(greedy)])
        self.assertEqual(0.5, vs_nbest[id(nbest)])
        calibration = evaluator.fit_oov_score_calibration([-0.40, -0.10], [-0.40, -0.10])
        spatials = [-0.40, -0.20, -0.10]
        dest = [2.0, 0.5, -0.5, -1.0]
        scored = evaluator.score_reserved_sources(
            {"greedy": [greedy], "greedy_alts": [alt]},
            calibration=calibration,
            ctc_spatials=spatials,
            oov_map_blend=0.5,
            lexicon_spatials=dest,
        )
        by_word = {item.word: item for item in scored}
        self.assertEqual(0.0, by_word["cax"].oov_map_blend)
        self.assertEqual(0.5, by_word["caz"].oov_map_blend)
        self.assertGreater(by_word["cax"].spatial, by_word["caz"].spatial)
        self.assertNotIn("target", inspect.signature(evaluator._oov_blend_by_margin).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.score_reserved_sources).parameters)

    def test_true_oov_use_the_train_fit_map_without_floor_clamp(self):
        calibration = evaluator.fit_oov_score_calibration([-0.40, -0.10], [-0.40, -0.10])
        candidate = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("cax", "en", (1, 2, 3), 0), "greedy", forward=-0.10,
        )
        spatials = [-0.40, -0.20, -0.10]
        mapped = evaluator.score_reserved_candidate(
            candidate, calibration=calibration, ctc_spatials=spatials, oov_conservative=False,
        )
        conservative = evaluator.score_reserved_candidate(
            candidate, calibration=calibration, ctc_spatials=spatials, oov_conservative=True,
        )
        self.assertTrue(mapped.frequency_free)
        self.assertEqual("greedy", mapped.source)
        self.assertGreater(mapped.spatial, conservative.spatial)
        blended = evaluator.score_reserved_candidate(
            candidate, calibration=calibration, ctc_spatials=spatials,
            oov_conservative=False, oov_map_blend=0.5,
        )
        self.assertGreater(mapped.spatial, blended.spatial)
        self.assertGreaterEqual(blended.spatial, conservative.spatial)
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ]
        strong = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (4,), 0), 8.0, frequency_free=True, source="greedy",
        )
        ranked = [entry.word for entry in evaluator.rank_static_fusion_with_reserved(lexicon, [strong])]
        self.assertEqual("cax", ranked[0])

    def test_greedy_oov_occupies_a_published_31_slot(self):
        layout = {"keyLabels": list("caxyz") + [None] * 59}
        output = logits([1, 2, 3], classes=8, frames=8)
        calibration = evaluator.fit_oov_score_calibration([-0.2, -0.1], [-0.2, -0.1])
        existing = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (4,), 10), 1.0 - 0.01 * index)
            for index in range(32)
        ]
        reserved, rejected = evaluator.reserved_oov_from_logits(
            output,
            layout,
            calibration=calibration,
            n_best=1,
            beam_width=4,
            existing=existing,
            known_offensive=set(),
            lexicon_by_word={},
            language="en",
            ctc_spatials=[1.0 - 0.01 * index for index in range(32)],
            oov_conservative=True,
        )
        self.assertEqual(0, rejected)
        self.assertIn("cax", {item.word for item in reserved})
        published = evaluator.publish_reserved_slots(existing, reserved, reserved_budget=1)
        ranked = evaluator.published_ranking(published, reserved)
        self.assertEqual(31, len(ranked))
        self.assertIn("cax", [entry.word for entry in ranked])

    def test_publish_mixes_one_extra_oov_with_in_lexicon_neighbors(self):
        base = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), 1.0 - 0.01 * index)
            for index in range(32)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0), -0.5, frequency_free=True, source="greedy",
        )
        weak_oov = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (3,), 0), -1.5, frequency_free=True, source="greedy_alts",
        )
        strong_oov = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cay", "en", (4,), 0), -0.2, frequency_free=True, source="greedy_alts",
        )
        rare_neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cab", "en", (5,), 3), 0.1, frequency_free=False, source="neighbors",
        )
        common_neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cat", "en", (1, 2, 3), 200), 0.1, frequency_free=False, source="neighbors",
        )
        ordered = evaluator.prioritize_reserved_candidates(
            [greedy, weak_oov, strong_oov, rare_neighbor, common_neighbor],
        )
        self.assertEqual("cax", ordered[0].word)
        self.assertEqual("cay", ordered[1].word)
        self.assertIn("cat", [candidate.word for candidate in ordered])
        published = evaluator.publish_reserved_slots(
            base, [greedy, weak_oov, strong_oov, rare_neighbor, common_neighbor], reserved_budget=4,
        )
        words = {item.word for item in published}
        self.assertTrue({"cax", "cay", "cat"}.issubset(words))
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                [greedy, weak_oov, strong_oov, rare_neighbor, common_neighbor],
                lexicon_reference=base,
            )
        ]
        self.assertIn("cax", ranked)
        self.assertIn("cat", ranked)

    def test_publish_fills_greedy_then_best_spatial_alts(self):
        base = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), 1.0 - 0.01 * index)
            for index in range(32)
        ]
        greedy = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (2,), 0), -0.5, frequency_free=True, source="greedy",
        )
        weak_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (3,), 0), -1.5, frequency_free=True, source="greedy_alts",
        )
        strong_alt = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cay", "en", (4,), 0), -0.2, frequency_free=True, source="greedy_alts",
        )
        neighbor = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cat", "en", (1, 2, 3), 77), 0.2, frequency_free=False, source="neighbors",
        )
        fill = evaluator.reserved_fill_ranks([greedy, weak_alt, strong_alt, neighbor])
        self.assertEqual(1, fill[("cax", "en")])
        self.assertEqual(2, fill[("cay", "en")])
        self.assertGreater(fill[("caz", "en")], fill[("cay", "en")])
        published = evaluator.publish_reserved_slots(
            base, [greedy, weak_alt, strong_alt, neighbor], reserved_budget=4,
        )
        words = {item.word for item in published}
        self.assertIn("cax", words)
        self.assertIn("cay", words)
        self.assertIn("caz", words)
        self.assertIn("cat", words)
        ranked = evaluator.published_ranking(
            published, [greedy, weak_alt, strong_alt, neighbor], lexicon_reference=base,
        )
        self.assertEqual(31, len(ranked))
        self.assertTrue({"cax", "cay"}.issubset({entry.word for entry in ranked}))

    def test_displacing_ranked_tail_keeps_lexicon_top3(self):
        base = [
            evaluator.ScoredLexiconEntry(
                evaluator.LexiconEntry(f"w{index}", "en", (1,), 10 + (32 - index)),
                1.0 - 0.01 * index,
            )
            for index in range(32)
        ]
        before = [entry.word for entry in evaluator.rank_static_fusion(base)[:3]]
        weak = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("oovword", "en", (2,), 0), -9.0, frequency_free=True,
        )
        published = evaluator.publish_reserved_slots(base, [weak], reserved_budget=1)
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(published, [weak], lexicon_reference=base)
        ]
        self.assertEqual(before, ranked[:3])
        self.assertIn("oovword", ranked)

    def test_weak_reserved_stays_in_the_published_31(self):
        """Greedy-41 proof: replacing the last of 31 keeps the extra in ranked[:31]."""
        base = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f"w{index}", "en", (1,), 10), 1.0 - 0.01 * index)
            for index in range(32)
        ]
        weak = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("oovword", "en", (2,), 0), -9.0, frequency_free=True,
        )
        published = evaluator.publish_reserved_slots(base, [weak], reserved_budget=1)
        ranked = evaluator.published_ranking(published, [weak])
        self.assertEqual(31, len(published))
        self.assertEqual(31, len(ranked))
        self.assertIn("oovword", {item.word for item in published})
        self.assertIn("oovword", [entry.word for entry in ranked])
        self.assertNotIn(base[30].word, {item.word for item in published})

    def test_in_lexicon_reserved_neighbors_use_real_frequency(self):
        layout = {"keyLabels": list("catd") + [None] * 60}
        output = logits([1, 2, 3], classes=8, frames=6)
        calibration = evaluator.fit_oov_score_calibration([-0.2, -0.1], [-0.2, -0.1])
        lexicon = {
            "cat": evaluator.LexiconEntry("cat", "en", (1, 2, 3), 77),
            "cad": evaluator.LexiconEntry("cad", "en", (1, 2, 4), 12),
        }
        existing = [evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("zz", "en", (4,), 8), -0.3)]
        reserved, rejected = evaluator.reserved_oov_from_logits(
            output,
            layout,
            calibration=calibration,
            n_best=1,
            beam_width=4,
            existing=existing,
            known_offensive=set(),
            lexicon_by_word=lexicon,
            language="en",
            ctc_spatials=[-0.3, -0.4],
            include_lexicon_neighbors=True,
        )
        self.assertEqual(0, rejected)
        neighbors = [item for item in reserved if item.word in {"cat", "cad"}]
        self.assertTrue(neighbors)
        for item in neighbors:
            self.assertFalse(item.frequency_free)
            self.assertEqual(lexicon[item.word].frequency, item.entry.frequency)
        oov = [item for item in reserved if item.frequency_free]
        for item in oov:
            self.assertEqual(0, item.entry.frequency)

    def test_in_lexicon_reserved_uses_decoder_spatial_already_on_the_slate(self):
        layout = {"keyLabels": list("catdxyz") + [None] * 57}
        output = logits([1, 2, 3], classes=8, frames=6)
        lexicon_by_word = {
            "cat": evaluator.LexiconEntry("cat", "en", (1, 2, 3), 80),
        }
        ctc = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (1, 2, 5), 40), 0.4),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cat", "en", (1, 2, 3), 80), 2.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (1, 2, 6), 30), -0.2),
        ]
        existing = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("zzz", "en", (7,), 5), 0.1),
        ]
        sources, rejected, _spellings = evaluator.collect_reserved_sources(
            output,
            layout,
            n_best=1,
            beam_width=4,
            existing=existing,
            known_offensive=set(),
            lexicon_by_word=lexicon_by_word,
            language="en",
            ctc=ctc,
        )
        self.assertEqual(0, rejected)
        cat = next(item for item in sources["greedy"] if item.entry.word == "cat")
        self.assertGreater(cat.entry.frequency, 0)
        self.assertIsNotNone(cat.decoder_spatial)
        expected = evaluator._z_normalize([item.spatial for item in ctc])[1]
        self.assertAlmostEqual(expected, cat.decoder_spatial)
        calibration = evaluator.fit_oov_score_calibration([-0.40, -0.10], [-0.40, -0.10])
        dest = [2.0, 0.5, -0.5, -1.0]
        scored = evaluator.score_reserved_candidate(
            cat,
            calibration=calibration,
            ctc_spatials=[-0.8, -0.2],
            lexicon_spatials=dest,
        )
        self.assertFalse(scored.frequency_free)
        self.assertEqual(80, scored.entry.frequency)
        self.assertAlmostEqual(evaluator.map_z_onto_pool(cat.decoder_spatial, dest), scored.spatial)
        calibrated = evaluator.score_reserved_candidate(
            evaluator.ReservedCandidate(cat.entry, "greedy", forward=cat.forward),
            calibration=calibration,
            ctc_spatials=[-0.8, -0.2],
            lexicon_spatials=dest,
        )
        self.assertNotAlmostEqual(calibrated.spatial, scored.spatial)
        oov_layout = {"keyLabels": list("caxd") + [None] * 60}
        oov_sources, _, _ = evaluator.collect_reserved_sources(
            logits([1, 2, 3], classes=8, frames=6),
            oov_layout,
            n_best=1,
            beam_width=4,
            existing=existing,
            known_offensive=set(),
            lexicon_by_word={},
            language="en",
            ctc=[
                evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cax", "en", (1, 2, 3), 0), 3.0),
                evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("cad", "en", (1, 2, 4), 12), 0.1),
            ],
        )
        oov = oov_sources["greedy"][0]
        self.assertEqual(0, oov.entry.frequency)
        self.assertIsNone(oov.decoder_spatial)
        missing, _, _ = evaluator.collect_reserved_sources(
            output,
            layout,
            n_best=1,
            beam_width=4,
            existing=existing,
            known_offensive=set(),
            lexicon_by_word=lexicon_by_word,
            language="en",
            ctc=[evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("zzz", "en", (7,), 5), 0.1)],
        )
        self.assertIsNone(next(item for item in missing["greedy"] if item.entry.word == "cat").decoder_spatial)
        lexicon = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("the", "en", (1,), 50), 2.0),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("car", "en", (2,), 40), 1.5),
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("can", "en", (3,), 30), 1.0),
        ]
        greedy_ff = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("cax", "en", (5,), 0), 8.0, frequency_free=True, source="greedy",
        )
        extra_ff = evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry("caz", "en", (6,), 0), 9.0, frequency_free=True, source="greedy_alts",
        )
        reserved = [greedy_ff, extra_ff, scored]
        extras = evaluator.extra_reserved_occupant_keys(reserved, lexicon, reserved_budget=3)
        published = evaluator.publish_reserved_slots(lexicon, reserved, reserved_budget=3)
        ranked = [
            entry.word
            for entry in evaluator.published_ranking(
                published,
                reserved,
                lexicon_reference=lexicon,
                extra_park_keys=extras,
                park_min_rank=4,
            )
        ]
        self.assertEqual(1, sum(1 for word in ranked[:3] if word in {"cax", "caz"}))
        self.assertNotIn("target", inspect.signature(evaluator.collect_reserved_sources).parameters)
        self.assertNotIn("target", inspect.signature(evaluator.score_reserved_candidate).parameters)

    def test_first_recovered_source_labels_without_construction_reading_targets(self):
        greedy = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("cax", "en", (1,), 0), "greedy", forward=-0.2,
        )
        neighbor = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("cat", "en", (1, 2, 3), 10), "neighbors",
        )
        leftover = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("catalog", "en", (1,), 4), "truncated_ctc", decoder_spatial=0.1,
        )
        sources = {
            "greedy": [greedy],
            "greedy_alts": [],
            "nbest": [],
            "neighbors": [neighbor],
            "truncated_ctc": [leftover],
            "truncated_geometry": [],
        }
        self.assertIsNone(evaluator.first_recovered_source("cax", ["cax"], sources))
        self.assertEqual("greedy", evaluator.first_recovered_source("cax", ["other"], sources))
        self.assertEqual("neighbors", evaluator.first_recovered_source("cat", ["other"], sources))
        self.assertEqual("truncated_ctc", evaluator.first_recovered_source("catalog", ["other"], sources))
        stage = evaluator.cumulative_recovery_stage("cat", sources)
        self.assertFalse(stage["greedyOnly"])
        self.assertTrue(stage["plusNeighbors"])
        self.assertNotIn("target", inspect.signature(evaluator.collect_reserved_sources).parameters)

    def test_calibration_observations_do_not_consult_targets(self):
        output = logits([1, 2], classes=8, frames=6)
        candidates = [evaluator.ScoredLexiconEntry(evaluator.LexiconEntry("ab", "en", (1, 2), 9), -0.05)]
        pairs, gap = evaluator.collect_oov_calibration_observations(
            output, candidates, (1, 3), "ac", {"ab"},
        )
        self.assertEqual(1, len(pairs))
        self.assertIsNotNone(gap)
        self.assertNotIn("target", inspect.signature(evaluator.collect_oov_calibration_observations).parameters)


if __name__ == "__main__":
    unittest.main()
