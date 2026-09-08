# SPDX-License-Identifier: GPL-3.0-only
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_swipe_reference_context as prepare


class ReferencePrefixTest(unittest.TestCase):
    def row(self, sentence, index):
        return {"sentence": sentence, "word_idx": index, "potentially_invalid_sentence": False}

    def test_only_preceding_words_are_kept_even_when_target_repeats(self):
        self.assertEqual(("go now ", None), prepare.reference_prefix(self.row("go now go later", 2), "go"))
        self.assertEqual(("", None), prepare.reference_prefix(self.row("go now go later", 0), "go"))

    def test_alignment_uses_canonical_punctuation_and_normalization(self):
        self.assertEqual(("We ", None), prepare.reference_prefix(self.row("We can’t, stop.", 1), "can't"))
        self.assertEqual((None, "unaligned_word"), prepare.reference_prefix(self.row("We can’t stop.", 2), "can't"))

    def test_invalid_or_ambiguous_indices_do_not_get_guessed(self):
        for index in (True, -1, 3, "1", None):
            with self.subTest(index=index):
                self.assertEqual((None, "invalid_word_idx"), prepare.reference_prefix(self.row("we go now", index), "go"))

    def test_publisher_flag_and_unsafe_text_are_rejected(self):
        row = self.row("we go now", 1)
        row["potentially_invalid_sentence"] = True
        self.assertEqual((None, "invalid_sentence_flag"), prepare.reference_prefix(row, "go"))
        self.assertEqual((None, "unsafe_sentence"), prepare.reference_prefix(self.row("we\ud800 go now", 1), "go"))

    def test_prefix_bound_counts_utf16_units_and_preserves_valid_unicode(self):
        prefix, reason = prepare.reference_prefix(self.row("😀" * 200 + "x target later", 1), "target")
        self.assertIsNone(reason)
        self.assertLessEqual(len(prefix.encode("utf-16-le")) // 2, 256)
        self.assertTrue(prefix.endswith("x "))
        self.assertNotIn("target", prefix)
        prefix.encode("utf-8", errors="strict")


if __name__ == "__main__":
    unittest.main()
