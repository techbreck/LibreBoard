# SPDX-License-Identifier: GPL-3.0-only
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_ite_spacing as spacing


class SpacingSiteTest(unittest.TestCase):
    def sites(self, typed, prompt):
        return list(spacing.spacing_sites(typed.split(), prompt.split()))

    def test_joined_words_and_split_words_are_spacing_sites(self):
        self.assertEqual([(1, 2, 1, 3)], self.sites("see thehospital now", "see the hospital now"))
        self.assertEqual([(1, 3, 1, 2)], self.sites("go inter company wide", "go intercompany wide"))

    def test_case_insensitive_regroups_qualify(self):
        self.assertEqual([(0, 1, 0, 2)], self.sites("Thehospital", "The Hospital"))

    def test_substitutions_and_punctuation_regroups_do_not_qualify(self):
        self.assertEqual([], self.sites("see the clinic now", "see the hospital now"))
        self.assertEqual([], self.sites("hi !there friend", "hi! there friend"))
        self.assertEqual([], self.sites("one two three", "one four three"))

    def test_equal_length_replace_is_not_a_regroup(self):
        self.assertEqual([], self.sites("a b c", "a d c"))


class TokenSpanTest(unittest.TestCase):
    def test_spans_track_character_offsets(self):
        self.assertEqual([("see", 0), ("the", 4), ("hospital", 8)],
                         spacing.token_spans("see the hospital"))


if __name__ == "__main__":
    unittest.main()
