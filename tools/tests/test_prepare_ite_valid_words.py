# SPDX-License-Identifier: GPL-3.0-only
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_ite_valid_words as ite


class IteValidWordsTest(unittest.TestCase):
    def row(self, **changes):
        return dict({"TYPED_WORD": "form", "ORIGINAL_WORD": "from", "CURRENT_INPUT": "I heard form",
                     "AC_WORD": "proprietary-output-is-ignored"}, **changes)

    def test_real_word_label_uses_reference_position_not_autocorrect_output(self):
        vocabulary = {"form", "from", "heard"}
        case, reason = ite.aligned_case(self.row(), "I heard from a friend.", vocabulary)
        self.assertEqual(("form", "from", "I heard "), case)
        self.assertIsNone(reason)
        keep, _ = ite.aligned_case(self.row(TYPED_WORD="from", CURRENT_INPUT="I heard from"),
                                  "I heard from a friend.", vocabulary)
        self.assertEqual(keep[0], keep[1])

    def test_unaligned_prefix_wrong_target_and_nonwords_are_rejected(self):
        vocabulary = {"form", "from", "heard"}
        for changes, reason in (({"CURRENT_INPUT": "Private prose form"}, "prefix"),
                                ({"ORIGINAL_WORD": "heard"}, "reference_alignment"),
                                ({"TYPED_WORD": "frmo"}, "not_valid_word"),
                                ({"CURRENT_INPUT": "I heard from"}, "raw_suffix"),
                                ({"CURRENT_INPUT": "form"}, "prefix")):
            with self.subTest(changes=changes):
                self.assertEqual(reason, ite.aligned_case(self.row(**changes), "I heard from a friend.", vocabulary)[1])

    def test_manifest_rejects_source_substitution_and_unsafe_members(self):
        source = ite.load_source()
        for change in ({"archiveUrl": "https://example.com/other.zip"}, {"license": "CC-BY-NC-4.0"},
                       {"files": [dict(source["files"][0], member="../unsafe"), *source["files"][1:]]}):
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "manifest.json"
                path.write_text(json.dumps(dict(source, **change)))
                with self.assertRaises(ite.IteDataError):
                    ite.load_source(path)

    def response(self, body=b"abc", status=206, content_range="bytes 2-4/10"):
        response = io.BytesIO(body)
        response.status = status
        response.headers = {"Content-Range": content_range}
        return response

    def test_range_reader_checks_exact_range_and_advances_only_after_success(self):
        reader = ite.RangeReader("https://example.com/archive", 10)
        reader.seek(2)
        with patch.object(ite.urllib.request, "urlopen", return_value=self.response()) as request:
            self.assertEqual(b"abc", reader.read(3))
            self.assertEqual("bytes=2-4", request.call_args.args[0].headers["Range"])
        self.assertEqual(5, reader.tell())
        self.assertEqual(8, reader.seek(-2, 2))
        with self.assertRaises(ite.IteDataError):
            reader.seek(-11, 2)

    def test_range_reader_retries_truncation_but_rejects_full_archive_response(self):
        reader = ite.RangeReader("https://example.com/archive", 10)
        reader.seek(2)
        with patch.object(ite.urllib.request, "urlopen", side_effect=[self.response(b"a"), self.response()]):
            self.assertEqual(b"abc", reader.read(3))
        reader.seek(2)
        with patch.object(ite.urllib.request, "urlopen", side_effect=[self.response(status=200) for _ in range(3)]):
            with self.assertRaises(ite.IteDataError):
                reader.read(3)
        self.assertEqual(2, reader.tell())

    def test_source_file_verification_rejects_corruption_and_links(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "data.csv"
            path.write_bytes(b"source")
            item = {"bytes": 6, "sha256": hashlib.sha256(b"source").hexdigest()}
            self.assertTrue(ite.verify_file(path, item))
            link = path.with_name("linked.csv")
            link.symlink_to(path)
            self.assertFalse(ite.verify_file(link, item))
            path.write_bytes(b"broken")
            self.assertFalse(ite.verify_file(path, item))


if __name__ == "__main__":
    unittest.main()
