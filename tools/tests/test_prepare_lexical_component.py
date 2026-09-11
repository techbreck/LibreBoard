# SPDX-License-Identifier: GPL-3.0-only
import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_lexical_component as lexical


class LexicalComponentTest(unittest.TestCase):
    def test_typo_stays_one_edit_from_the_target(self):
        rng = random.Random(7)
        for _ in range(50):
            word = rng.choice(lexical.COMPOUNDS)
            raw = lexical.typo(word, rng)
            self.assertEqual(len(word), len(raw))
            changed = [i for i, (a, b) in enumerate(zip(word, raw)) if a != b]
            self.assertEqual(1, len(changed))
            self.assertIn(raw[changed[0]], lexical.ADJACENT[word[changed[0]]])

    def test_personal_rows_carry_the_target_in_a_synthetic_fixture(self):
        rng = random.Random(1)
        for row in lexical.personal_rows(0, rng):
            self.assertEqual("personal", row["lexicalKind"])
            self.assertIn(row["target"], row["personalWords"])
            self.assertNotEqual(row["raw"], row["target"])
            self.assertEqual("project_authored", row["collectionMethod"])
            self.assertTrue(row["target"].isalpha())

    def test_compound_rows_are_typo_or_split_form_variants(self):
        rng = random.Random(2)
        for row in lexical.compound_rows(0, rng):
            self.assertEqual("compound", row["lexicalKind"])
            self.assertIn(row["target"], lexical.COMPOUNDS)
            self.assertNotEqual(row["raw"], row["target"])
            if " " in row["raw"]:
                self.assertEqual(row["target"], row["raw"].replace(" ", ""))

    def test_generation_is_deterministic(self):
        first = [row["id"] for row in lexical.personal_rows(3, random.Random(lexical.SEED))]
        second = [row["id"] for row in lexical.personal_rows(3, random.Random(lexical.SEED))]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
