# SPDX-License-Identifier: GPL-3.0-only
import collections
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_noisy_typing_dataset as adapter


def keyboard_text():
    return "\n".join(f"{letter};{i * 10};10;8;8" for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz"))


def events(word):
    return "|".join(f"{(ord(letter)-97)*10},10,{index*100};{(ord(letter)-97)*10+1},10,{index*100+20}"
                    for index, letter in enumerate(word))


class NoisyTypingTests(unittest.TestCase):
    def test_manifest_pins_version_license_and_excludes_known_overlap(self):
        source = adapter.load_source()
        self.assertTrue(source["downloadUrl"].endswith("?revision=1"))
        self.assertNotIn("vt_pilot1", source["includedExperiments"])
        self.assertNotIn("vt_potential", source["includedExperiments"])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manifest.json"
            for key, value in (("license", "CC-BY-NC-4.0"), ("downloadUrl", "https://osf.io/download/7ev5z/"),
                               ("archiveBytes", True), ("archiveSha256", "invalid")):
                path.write_text(json.dumps({**source, key: value}))
                with self.subTest(key=key), self.assertRaises(adapter.NoisyTypingError):
                    adapter.load_source(path)

    def test_one_touch_down_per_tap_and_relative_times(self):
        (raw, points), reason = adapter.parse_word("dog", events("cat"), adapter.parse_keyboard(keyboard_text()))
        self.assertIsNone(reason)
        self.assertEqual("cat", raw)
        self.assertEqual(3, len(points))
        self.assertEqual([0, 100, 200], [p["timeMillis"] for p in points])
        self.assertEqual((20 + 4) / 258, points[0]["x"])

    def test_correct_replay_is_not_fabricated_into_error(self):
        self.assertEqual((None, "correct_replay"), adapter.parse_word("cat", events("cat"), adapter.parse_keyboard(keyboard_text())))

    def test_nonfinite_geometry_and_duplicate_keys_rejected(self):
        for text in (keyboard_text() + "\na;0;10;8;8", keyboard_text().replace("a;0;10", "a;nan;10")):
            with self.assertRaises(adapter.NoisyTypingError):
                adapter.parse_keyboard(text)

    def test_nonfinite_samples_fail_and_out_of_bounds_are_not_clamped(self):
        keyboard = adapter.parse_keyboard(keyboard_text())
        with self.assertRaises(adapter.NoisyTypingError):
            adapter.parse_word("dog", "nan,10,0", keyboard)
        self.assertEqual((None, "out_of_bounds"), adapter.parse_word("dog", "9000,10,0", keyboard))
        self.assertEqual((None, "non_monotonic_time"), adapter.parse_word("dog", "20,10,100;20,10,90", keyboard))

    def test_control_keys_and_equidistant_keys_are_excluded(self):
        keyboard = adapter.parse_keyboard(keyboard_text() + "\n<b>;20;16;8;8")
        self.assertEqual((None, "control_key"), adapter.parse_word("dog", "20,16,0", keyboard))
        self.assertEqual((None, "ambiguous_key"), adapter.parse_word("dog", "5,10,0", keyboard))

    def test_context_discarded_and_participant_conditions_stay_together(self):
        text = f"ID: public-prompt\nREF: dog\nLEFT: private fixture\nRIGHT: discard this\nORIG: discard too\nIN: {events('cat')}\n"
        counts = collections.Counter()
        keyboard = adapter.parse_keyboard(keyboard_text())
        first = list(adapter.rows_from_log(text, "noisy_typing/phone/vt_exp1/test_word/p3_feed.log", keyboard, counts))[0]
        second = list(adapter.rows_from_log(text, "noisy_typing/phone/vt_exp1/test_word/p3_nofeed.log", keyboard, counts))[0]
        self.assertEqual(first["sessionId"], second["sessionId"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual("", first["precedingContext"])
        self.assertNotIn("private", json.dumps(first))
        self.assertNotIn("discard", json.dumps(first))

    def test_missing_reference_and_misalignment_are_counted_not_guessed(self):
        counts = collections.Counter()
        text = f"IN: {events('cat')}\n\nREF: two words\nIN: {events('cat')}\n"
        rows = list(adapter.rows_from_log(text, "noisy_typing/phone/vt_exp1/test_word/p3_feed.log",
                                        adapter.parse_keyboard(keyboard_text()), counts))
        self.assertEqual([], rows)
        self.assertEqual({"orphan_input": 1, "unaligned_sentence": 1}, dict(counts))

    def test_failed_download_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "source.zip"
            path.write_bytes(b"existing")
            source = {"archiveBytes": 3, "archiveSha256": hashlib.sha256(b"new").hexdigest(), "downloadUrl": "https://osf.io/fixture"}
            with patch.object(adapter.urllib.request, "urlopen", return_value=io.BytesIO(b"too large")):
                with self.assertRaises(adapter.NoisyTypingError):
                    adapter.fetch_archive(path, source)
            self.assertEqual(b"existing", path.read_bytes())
            self.assertEqual([path], list(path.parent.iterdir()))


if __name__ == "__main__":
    unittest.main()
