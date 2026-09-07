# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import merge_tap_evaluation_sources as merger


class MergeTapSourcesTest(unittest.TestCase):
    def component(self, root, name, row_id, license_name="MIT"):
        folder = root / name
        folder.mkdir()
        raw = {"schemaVersion": 1, "id": row_id, "sessionId": "shared-participant",
               "category": "valid_word", "target": "word", "raw": "word", "shouldCorrect": False,
               "languageTag": "en-US", "precedingContext": "a ", "fieldClass": "plain",
               "collectionMethod": "project_authored", "touchPoints": []}
        data = folder / "rows.jsonl"
        data.write_text(json.dumps(raw) + "\n")
        manifest = {"schemaVersion": 1, "datasetId": name, "dataFile": data.name,
                    "dataSha256": hashlib.sha256(data.read_bytes()).hexdigest(), "license": license_name,
                    "collectionProtocol": "Explicit test fixture", "containsHumanContributions": False,
                    "consentStatement": ""}
        path = folder / "source-manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_merge_preserves_shared_session_and_is_input_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            sources = [self.component(root, "first", "row-a"), self.component(root, "second", "row-b")]
            report = merger.merge(sources, root / "one")
            merger.merge(list(reversed(sources)), root / "two")
            self.assertEqual(1, report["sessions"])
            self.assertEqual(1, report["crossComponentSessions"])
            self.assertFalse(report["releaseEligible"])
            rows = [json.loads(line) for line in (root / "one/tap-cases.jsonl").read_text().splitlines()]
            self.assertEqual({"shared-participant"}, {row["sessionId"] for row in rows})
            for path in (root / "one").iterdir():
                self.assertEqual(path.read_bytes(), (root / "two" / path.name).read_bytes())

    def test_duplicate_ids_and_mixed_licenses_fail_before_output(self):
        for duplicate in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                sources = [self.component(root, "first", "same"),
                           self.component(root, "second", "same" if duplicate else "different", "MIT" if duplicate else "CC0-1.0")]
                with self.assertRaises(merger.MergeError):
                    merger.merge(sources, root / "out")
                self.assertFalse((root / "out").exists())

    def test_changed_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = self.component(root, "first", "one")
            (source.parent / "rows.jsonl").write_text("changed")
            with self.assertRaises(ValueError):
                merger.merge([source], root / "out")


if __name__ == "__main__":
    unittest.main()
