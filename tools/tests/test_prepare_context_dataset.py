# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import prepare_context_dataset as context_data  # noqa: E402


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def source_row(session: str, sentence: str, *, invalid: bool = False) -> dict:
    return {
        "id": f"{session}-{sentence}",
        "session": session,
        "timestamp": 123456789,
        "word": "typed",
        "sentence": sentence,
        "word_idx": 1,
        "potentially_invalid_sentence": invalid,
        "data": [{"x": 1, "y": 2, "t": 3}],
    }


class PreparedFixture:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="libreboard-context-data-test-")
        self.root = pathlib.Path(self.temporary.name)
        self.source_root = self.root / "sources"
        self.output_root = self.root / "output"
        self.manifest_path = self.root / "sources.json"
        self.policy_path = self.root / "policy.json"
        self.project_path = self.root / "project.json"
        self.source_dir = self.source_root / "fixture-context"
        self.source_dir.mkdir(parents=True)

        policy = json.loads(context_data.DEFAULT_POLICY.read_text())
        policy.update({
            "sourceId": "fixture-context",
            "projectAuthoredData": "models/context/fixture-project.json",
            "sentenceSampleBasisPoints": 10_000,
            "minimumAcceptedSentences": 1,
            "minimumGermanSentences": 1,
            "maximumRejectedFraction": 0.5,
        })
        self.policy_path.write_bytes(canonical(policy))
        loaded_policy = context_data.load_policy(self.policy_path)

        sessions = {}
        candidate = 0
        while set(sessions) != set(context_data.SPLITS):
            session = f"session-{candidate}"
            split = context_data.split_for_session("fixture-context", session, loaded_policy)
            sessions.setdefault(split, session)
            candidate += 1
        train_sentence = source_row(sessions["train"], "The keyboard works completely offline.")
        artifacts = []
        for filename, values in {
            "train.jsonl": [
                train_sentence,
                train_sentence,
                source_row(sessions["train"], "invalid", invalid=True),
            ],
            "dev.jsonl": [
                source_row(sessions["validation"], "This sentence belongs to validation."),
            ],
            "test.jsonl": [
                source_row(sessions["test"], "The held out sentence remains separate."),
            ],
        }.items():
            path = self.source_dir / filename
            payload = b"".join(canonical(value) for value in values)
            path.write_bytes(payload)
            artifacts.append({
                "path": filename,
                "purpose": "training-data",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        license_path = self.source_dir / "LICENSE"
        license_path.write_text("fixture license\n", encoding="utf-8")
        artifacts.insert(0, {
            "path": "LICENSE",
            "purpose": "license",
            "bytes": license_path.stat().st_size,
            "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest(),
        })
        self.manifest_path.write_bytes(canonical({
            "schemaVersion": 1,
            "sources": [{
                "id": "fixture-context",
                "kind": "dataset",
                "repositoryType": "dataset",
                "repository": "example/context",
                "revision": "a" * 40,
                "license": "MIT",
                "sourceUrl": "https://huggingface.co/datasets/example/context",
                "artifacts": artifacts,
            }],
        }))
        self.project_path.write_bytes(canonical({
            "schemaVersion": 1,
            "id": "fixture-project-de",
            "locale": "de",
            "license": "Apache-2.0",
            "slots": {
                "sentence": [
                    "Die Tastatur bleibt vollständig offline.",
                    "Das Wörterbuch funktioniert ohne Netzwerk.",
                ],
            },
            "templates": [{
                "id": "fixture-template",
                "fieldClass": "plain",
                "text": "{sentence}",
                "slots": ["sentence"],
                "sessionBuckets": 4,
            }],
            "confusionSets": [["das", "dass"]],
            "aliases": {"ists": "ist es"},
        }))

    def close(self):
        self.temporary.cleanup()

    def prepare(self, output_root: pathlib.Path | None = None):
        return context_data.prepare(
            source_root=self.source_root,
            output_root=output_root or self.output_root,
            manifest_path=self.manifest_path,
            policy_path=self.policy_path,
            project_path=self.project_path,
            enforce_minimums=False,
        )


class PrepareContextDatasetTest(unittest.TestCase):
    def test_shared_collection_sessions_match_swipe_identity_and_split(self):
        policy = context_data.load_policy()
        swipe = context_data.prepare_swipe_dataset.load_policy()
        for index in range(1000):
            session = f"shared-session-{index}"
            self.assertEqual(context_data.prepare_swipe_dataset.session_hash(session, swipe),
                             context_data.session_hash(swipe.source_id, session, policy))
            self.assertEqual(context_data.prepare_swipe_dataset.split_for_session(session, swipe),
                             context_data.split_for_session(swipe.source_id, session, policy))

    def test_project_authored_sessions_keep_their_namespace(self):
        policy = context_data.load_policy()
        self.assertNotEqual(context_data.session_hash(policy.shared_source_id, "same-session", policy),
                            context_data.session_hash("project-authored", "same-session", policy))

    def test_shared_policy_hash_and_boundaries_cannot_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "policy.json"
            for update in (
                {"sharedSwipePolicySha256": "0" * 64},
                {"splitBasisPoints": {"train": 8000, "validation": 1000, "test": 1000}},
            ):
                with self.subTest(update=update):
                    policy = json.loads(context_data.DEFAULT_POLICY.read_text())
                    policy.update(update)
                    path.write_bytes(canonical(policy))
                    with self.assertRaises(context_data.ContextDataError):
                        context_data.load_policy(path)

    def setUp(self):
        self.fixture = PreparedFixture()

    def tearDown(self):
        self.fixture.close()

    def test_prepares_deterministic_session_separated_privacy_minimized_corpus(self):
        report = self.fixture.prepare()
        loaded = context_data.load_prepared_manifest(self.fixture.output_root)

        self.assertEqual(5, report["counts"]["acceptedSentences"])
        self.assertEqual(report, loaded)
        self.assertEqual(3, report["counts"]["acceptedSourceSentences"])
        self.assertEqual(2, report["counts"]["acceptedProjectSentences"])
        self.assertEqual(1, report["counts"]["duplicateSentences"])
        sessions_by_split = {}
        languages = set()
        for split in context_data.SPLITS:
            path = self.fixture.output_root / f"{split}.sentences.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            sessions_by_split[split] = {record["sessionId"] for record in records}
            for record in records:
                self.assertEqual(split, record["split"])
                self.assertEqual(
                    {"schemaVersion", "id", "sessionId", "split", "language", "fieldClass", "text", "sourceId"},
                    set(record),
                )
                self.assertNotIn("timestamp", record)
                self.assertNotIn("word", record)
                languages.add(record["language"])
        for first_index, first in enumerate(context_data.SPLITS):
            for second in context_data.SPLITS[first_index + 1:]:
                self.assertTrue(sessions_by_split[first].isdisjoint(sessions_by_split[second]))
        self.assertEqual({"en-US", "de"}, languages)

        second_root = self.fixture.root / "second-output"
        second = self.fixture.prepare(second_root)
        self.assertEqual(report, second)
        for filename in (*report["outputs"], "split-manifest.json"):
            self.assertEqual(
                (self.fixture.output_root / filename).read_bytes(),
                (second_root / filename).read_bytes(),
            )

    def test_rejects_changed_source_bytes(self):
        with (self.fixture.source_dir / "train.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(context_data.ContextDataError, "wrong size"):
            self.fixture.prepare()

    def test_prepared_manifest_rejects_changed_output_and_wrong_pinned_bytes(self):
        self.fixture.prepare()
        train_path = self.fixture.output_root / "train.sentences.jsonl"
        with train_path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(context_data.ContextDataError, "wrong size"):
            context_data.load_prepared_manifest(self.fixture.output_root)

        self.fixture.prepare()
        pinned_path = self.fixture.root / "pinned.json"
        pinned_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(context_data.ContextDataError, "does not match"):
            context_data.load_prepared_manifest(
                self.fixture.output_root,
                require_pinned=True,
                pinned_path=pinned_path,
            )

    def test_rejects_project_schema_and_template_expansion_drift(self):
        value = json.loads(self.fixture.project_path.read_text())
        value["license"] = "CC-BY-4.0"
        self.fixture.project_path.write_bytes(canonical(value))
        with self.assertRaisesRegex(context_data.ContextDataError, "license is invalid"):
            self.fixture.prepare()

        value["license"] = "Apache-2.0"
        value["templates"][0]["slots"] = []
        self.fixture.project_path.write_bytes(canonical(value))
        with self.assertRaisesRegex(context_data.ContextDataError, "placeholders do not match"):
            self.fixture.prepare()


if __name__ == "__main__":
    unittest.main()
