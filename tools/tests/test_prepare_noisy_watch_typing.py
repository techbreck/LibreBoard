# SPDX-License-Identifier: GPL-3.0-only
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_noisy_watch_typing as watch

KEYBOARD = """# test layout
q;10;10;20;20
w;30;10;20;20
e;50;10;20;20
r;70;10;20;20
t;90;10;20;20
y;110;10;20;20
u;130;10;20;20
i;150;10;20;20
o;170;10;20;20
p;190;10;20;20
a;20;40;20;20
s;40;40;20;20
d;60;40;20;20
f;80;40;20;20
g;100;40;20;20
h;120;40;20;20
j;140;40;20;20
k;160;40;20;20
l;180;40;20;20
z;30;70;20;20
x;50;70;20;20
c;70;70;20;20
v;90;70;20;20
b;110;70;20;20
n;130;70;20;20
m;150;70;20;20
';170;70;20;20
"""


class WatchKeyboardTest(unittest.TestCase):
    def test_apostrophe_key_is_accepted_but_excluded_from_letter_bounds(self):
        keys, (left, top, width, height) = watch.parse_keyboard(KEYBOARD)
        self.assertIn("'", keys)
        self.assertEqual((0, 0, 200, 80), (left, top, width, height))


class WatchWordTest(unittest.TestCase):
    def setUp(self):
        self.keyboard = watch.parse_keyboard(KEYBOARD)

    def test_letter_tap_replays_the_key(self):
        parsed, reason = watch.parse_word("q", "30,10,100;30,10,120", self.keyboard)
        self.assertIsNone(reason)
        self.assertEqual("w", parsed[0])
        self.assertEqual(0, parsed[1][0]["timeMillis"])

    def test_lock_confidence_suffix_is_stripped(self):
        parsed, reason = watch.parse_word("q", "30,10,100:0.4:2;30,10,120:0.4:2", self.keyboard)
        self.assertIsNone(reason)
        self.assertEqual("w", parsed[0])

    def test_apostrophe_tap_is_a_control_key_reject(self):
        self.assertEqual((None, "control_key"), watch.parse_word("q", "170,70,100", self.keyboard))

    def test_negative_timestamp_is_an_invalid_sample_reject(self):
        self.assertEqual((None, "invalid_sample"), watch.parse_word("q", "10,10,-5", self.keyboard))

    def test_malformed_sample_shape_is_an_invalid_sample_reject(self):
        self.assertEqual((None, "invalid_sample"), watch.parse_word("q", "10,10,5,900", self.keyboard))
        self.assertEqual((None, "invalid_sample"), watch.parse_word("q", "10,ten,5", self.keyboard))


class WatchInclusionTest(unittest.TestCase):
    def included(self, name):
        return watch.included(name)

    def test_force_aligned_impact_uses_test_word_and_native_word_condition(self):
        self.assertTrue(self.included("noisy_typing/watch/impact_exp1/test_word/p1.log"))
        self.assertTrue(self.included("noisy_typing/watch/impact_exp1/test/p1_word.log"))
        self.assertFalse(self.included("noisy_typing/watch/impact_exp1/test/p1_sentence.log"))
        self.assertFalse(self.included("noisy_typing/watch/impact_exp1/test/p1_compose.log"))

    def test_vw_and_comp_use_all_test_logs(self):
        self.assertTrue(self.included("noisy_typing/watch/vw_exp3/test/p2_lock.log"))
        self.assertTrue(self.included("noisy_typing/watch/comp_exp2/test/p3.log"))

    def test_practice_dev_phone_and_unknown_experiments_are_excluded(self):
        for name in (
            "noisy_typing/watch/vw_exp3/pract/p1.log",
            "noisy_typing/watch/impact_exp1/dev/p1.log",
            "noisy_typing/watch/impact_exp1/pract_word/p1.log",
            "noisy_typing/phone/vt_exp1/test_word/p1.log",
            "noisy_typing/watch/vw_exp2/test/p1.log",
            "noisy_typing/watch/vw_exp1/test/p1.txt",
        ):
            with self.subTest(name=name):
                self.assertFalse(self.included(name))


if __name__ == "__main__":
    unittest.main()
