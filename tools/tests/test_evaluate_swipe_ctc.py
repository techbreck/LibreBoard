# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
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


if __name__ == "__main__":
    unittest.main()
