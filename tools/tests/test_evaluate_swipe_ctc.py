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

    def test_unique_best_oov_uses_more_of_the_ols_map(self):
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
        self.assertEqual(0.5, tied[id(greedy)])
        nbest_close = evaluator.ReservedCandidate(
            evaluator.LexiconEntry("caa", "en", (4,), 0), "nbest", forward=-0.06,
        )
        vs_nbest = evaluator._oov_blend_by_margin([greedy, alt, nbest_close], 0.5)
        self.assertEqual(0.0, vs_nbest[id(greedy)])

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
