# SPDX-License-Identifier: GPL-3.0-only
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_ite_contractions as contractions


class IteContractionTest(unittest.TestCase):
    def row(self, raw="dont", target="don't", current="I really dont"):
        return {"TYPED_WORD": raw, "ORIGINAL_WORD": target, "CURRENT_INPUT": current,
                "AC_WORD": "this output is never a label"}

    def test_missing_apostrophe_and_correct_contraction_keep_the_observed_input(self):
        sentence = "I really don't know."
        self.assertEqual(("dont", "don't", "I really "), contractions.aligned_contraction(self.row(), sentence)[0])
        self.assertEqual(("don't", "don't", "I really "), contractions.aligned_contraction(
            self.row("don't", current="I really don't"), sentence)[0])

    def test_noun_possessives_are_not_contractions(self):
        self.assertEqual("not_contraction", contractions.aligned_contraction(
            self.row("weeks", "week's", "It was last weeks"), "It was last week's news.")[1])

    def test_wrong_reference_position_private_prefix_and_control_characters_fail(self):
        for row, sentence, reason in (
            (self.row(), "I really don'ts know.", "reference_alignment"),
            (self.row(current="Private note dont"), "I really don't know.", "prefix"),
            (self.row(raw="don\x92t"), "I really don't know.", "invalid_raw"),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(reason, contractions.aligned_contraction(row, sentence)[1])


if __name__ == "__main__":
    unittest.main()
