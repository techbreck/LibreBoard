# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import hashlib
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_tap_evaluation as tap_data  # noqa: E402


def find_session(split: str, policy: tap_data.Policy) -> str:
    for number in range(100_000):
        candidate = f"session-{split}-{number}"
        if tap_data.split_for_session(candidate, policy) == split:
            return candidate
    raise AssertionError("could not find split fixture")


def row(
    identifier: str,
    session: str,
    category: str = "tap_error",
    *,
    collection_method: str = "human_replay",
) -> dict:
    value = {
        "schemaVersion": 1,
        "id": identifier,
        "sessionId": session,
        "category": category,
        "target": "this",
        "raw": "thsi",
        "languageTag": "en-US",
        "precedingContext": "type",
        "fieldClass": "short_message",
        "collectionMethod": collection_method,
        "touchPoints": [
            {"x": 0.4, "y": 0.2, "timeMillis": 0},
            {"x": 0.6, "y": 0.5, "timeMillis": 70},
        ],
    }
    if category == "valid_word":
        value["target"] = "there"
        value["raw"] = "their"
        value["shouldCorrect"] = True
    if category == "lexical":
        value["target"] = "LibreBoard"
        value["raw"] = "libreboars"
        value["lexicalKind"] = "personal"
        value["personalWords"] = ["LibreBoard"]
    return value


def write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in rows))


def write_source_manifest(path: pathlib.Path, source: pathlib.Path, *, human: bool = True) -> None:
    path.write_text(json.dumps({
        "schemaVersion": 1,
        "datasetId": "libreboard-tap-fixture-v1",
        "dataFile": source.name,
        "dataSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "license": "CC0-1.0",
        "collectionProtocol": "Unit-test fixture with no real participant data.",
        "containsHumanContributions": human,
        "consentStatement": "Fixture contributors consented to CC0 publication." if human else "",
    }))


class PrepareTapEvaluationTest(unittest.TestCase):
    def test_preparation_hashes_sessions_splits_whole_sessions_and_is_reproducible(self):
        policy = tap_data.load_policy()
        sessions = {split: find_session(split, policy) for split in tap_data.SPLITS}
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            output = root / "output"
            write_rows(source, [
                row("train", sessions["train"]),
                row("validation", sessions["validation"], "spacing", collection_method="project_authored"),
                row("test", sessions["test"], "lexical", collection_method="project_authored"),
            ])
            source_manifest = root / "source-manifest.json"
            write_source_manifest(source_manifest, source)

            report = tap_data.prepare(source, source_manifest, output, allow_small=True)
            first_outputs = report["outputs"]
            test_record = json.loads((output / "test.jsonl").read_text())
            self.assertEqual("test", test_record["split"])
            self.assertRegex(test_record["sessionId"], r"^[0-9a-f]{64}$")
            self.assertNotEqual(sessions["test"], test_record["sessionId"])
            self.assertEqual("personal", test_record["lexicalKind"])
            self.assertFalse(report["releaseEligible"])

            report_again = tap_data.prepare(source, source_manifest, output, allow_small=True)
            self.assertEqual(first_outputs, report_again["outputs"])
            self.assertEqual(report["source"], report_again["source"])

    def test_spatial_errors_require_real_human_touch_data(self):
        policy = tap_data.load_policy()
        value = row("spatial", find_session("test", policy), collection_method="project_authored")
        with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "require human touch data"):
            tap_data.normalize_record(value, 1, policy)
        value["collectionMethod"] = "human_natural"
        value["touchPoints"] = []
        with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "require human touch data"):
            tap_data.normalize_record(value, 1, policy)

    def test_category_specific_fields_are_fail_closed(self):
        policy = tap_data.load_policy()
        session = find_session("test", policy)
        valid_word = row("valid", session, "valid_word")
        valid_word.pop("shouldCorrect")
        with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "requires shouldCorrect"):
            tap_data.normalize_record(valid_word, 1, policy)

        personal = row("personal", session, "lexical", collection_method="project_authored")
        personal.pop("personalWords")
        with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "require personalWords"):
            tap_data.normalize_record(personal, 2, policy)

    def test_small_corpus_cannot_be_mistaken_for_release_data(self):
        policy = tap_data.load_policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            write_rows(source, [row("one", find_session("test", policy))])
            source_manifest = root / "source-manifest.json"
            write_source_manifest(source_manifest, source)
            with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "release minimums not met"):
                tap_data.prepare(source, source_manifest, root / "output")

    def test_duplicate_identifiers_within_a_session_are_rejected(self):
        policy = tap_data.load_policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            duplicate = row("same", find_session("test", policy))
            write_rows(source, [duplicate, duplicate])
            source_manifest = root / "source-manifest.json"
            write_source_manifest(source_manifest, source)
            with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "duplicate IDs"):
                tap_data.prepare(source, source_manifest, root / "output", allow_small=True)

    def test_source_manifest_binds_license_consent_and_exact_data(self):
        policy = tap_data.load_policy()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            write_rows(source, [row("one", find_session("test", policy))])
            source_manifest = root / "source-manifest.json"
            write_source_manifest(source_manifest, source)
            manifest = json.loads(source_manifest.read_text())
            manifest["dataSha256"] = "0" * 64
            source_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(tap_data.TapEvaluationDataError, "data hash does not match"):
                tap_data.prepare(source, source_manifest, root / "output", allow_small=True)

    def test_exact_release_minima_produce_release_eligible_manifest(self):
        policy = tap_data.load_policy()
        sessions = {split: find_session(split, policy) for split in tap_data.SPLITS}
        rows = [
            row("train-seed", sessions["train"], "spacing", collection_method="project_authored"),
            row("validation-seed", sessions["validation"], "spacing", collection_method="project_authored"),
        ]
        rows.extend(row(f"tap-{index}", sessions["test"]) for index in range(3_000))
        for index in range(1_000):
            value = row(f"valid-{index}", sessions["test"], "valid_word")
            if index >= 500:
                value["target"] = value["raw"]
                value["shouldCorrect"] = False
            rows.append(value)
        rows.extend(
            row(f"spacing-{index}", sessions["test"], "spacing", collection_method="project_authored")
            for index in range(500)
        )
        for index in range(500):
            value = row(f"lexical-{index}", sessions["test"], "lexical", collection_method="project_authored")
            kind = ("personal", "contraction", "compound")[index % 3]
            value["lexicalKind"] = kind
            if kind != "personal":
                value.pop("personalWords")
            rows.append(value)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.jsonl"
            write_rows(source, rows)
            source_manifest = root / "source-manifest.json"
            write_source_manifest(source_manifest, source)

            report = tap_data.prepare(source, source_manifest, root / "output")

            self.assertTrue(report["releaseEligible"])
            self.assertEqual([], report["releaseErrors"])
            self.assertEqual(3_000, report["counts"]["test"]["tap_error"])
            self.assertEqual({"correct": 500, "keep": 500}, report["testValidWordCounts"])
            self.assertTrue(all(report["testLexicalCounts"][kind] > 0 for kind in tap_data.LEXICAL_KINDS))


if __name__ == "__main__":
    unittest.main()
