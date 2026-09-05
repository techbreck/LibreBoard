# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import model_sources  # noqa: E402
import prepare_swipe_dataset as swipe_data  # noqa: E402


def canonical(value: dict) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def row(identifier: int, session: str, word: str, points=None) -> dict:
    return {
        "id": identifier,
        "session": session,
        "word": word,
        "orientation": "portrait-primary",
        "data": points or [
            {"x": 0.05, "y": 1 / 6, "t": 0},
            {"x": 0.45, "y": 1 / 6, "t": 10},
        ],
    }


def layout() -> dict:
    positions = {
        "q": (0.05, 1 / 6), "w": (0.15, 1 / 6), "e": (0.25, 1 / 6),
        "r": (0.35, 1 / 6), "t": (0.45, 1 / 6), "y": (0.55, 1 / 6),
        "u": (0.65, 1 / 6), "i": (0.75, 1 / 6), "o": (0.85, 1 / 6),
        "p": (0.95, 1 / 6), "a": (0.10, 0.5), "s": (0.20, 0.5),
        "d": (0.30, 0.5), "f": (0.40, 0.5), "g": (0.50, 0.5),
        "h": (0.60, 0.5), "j": (0.70, 0.5), "k": (0.80, 0.5),
        "l": (0.90, 0.5), "z": (0.20, 5 / 6), "x": (0.30, 5 / 6),
        "c": (0.40, 5 / 6), "v": (0.50, 5 / 6), "b": (0.60, 5 / 6),
        "n": (0.70, 5 / 6), "m": (0.80, 5 / 6),
    }
    return {
        "name": "qwerty",
        "letters": "abcdefghijklmnopqrstuvwxyz",
        "keys": [
            {"letter": letter, "cx": x, "cy": y, "rx": 0.05, "ry": 1 / 6}
            for letter, (x, y) in positions.items()
        ],
    }


def policy_document() -> dict:
    value = json.loads(swipe_data.DEFAULT_POLICY.read_text())
    value["sourceId"] = "fixture-swipe"
    value["dataArtifacts"] = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    value["layoutArtifact"] = "layout.json"
    value["maximumRejectedFraction"] = 0.5
    return value


def find_session(split: str, policy: swipe_data.Policy) -> str:
    for number in range(100_000):
        candidate = f"session-{split}-{number}"
        if swipe_data.split_for_session(candidate, policy) == split:
            return candidate
    raise AssertionError("could not find session fixture")


class PreparedFixture:
    def __init__(self, root: pathlib.Path, rows_by_artifact: dict[str, list[dict]]):
        self.root = root
        self.source_root = root / "sources"
        self.output_root = root / "output"
        self.policy_path = root / "policy.json"
        self.policy_path.write_text(json.dumps(policy_document()))
        source_dir = self.source_root / "fixture-swipe"
        source_dir.mkdir(parents=True)
        artifacts = []
        for name, rows in rows_by_artifact.items():
            payload = b"".join(canonical(value) for value in rows)
            (source_dir / name).write_bytes(payload)
            artifacts.append(self._artifact(name, "training-data", payload))
        layout_payload = json.dumps(layout(), separators=(",", ":"), sort_keys=True).encode()
        (source_dir / "layout.json").write_bytes(layout_payload)
        artifacts.append(self._artifact("layout.json", "layout", layout_payload))
        license_payload = b"MIT fixture\n"
        (source_dir / "LICENSE").write_bytes(license_payload)
        artifacts.append(self._artifact("LICENSE", "license", license_payload))
        manifest = {
            "schemaVersion": 1,
            "sources": [{
                "id": "fixture-swipe",
                "kind": "dataset",
                "repositoryType": "dataset",
                "repository": "example/swipe",
                "revision": "b" * 40,
                "license": "MIT",
                "sourceUrl": "https://huggingface.co/datasets/example/swipe",
                "artifacts": artifacts,
            }],
        }
        self.manifest_path = root / "sources.json"
        self.manifest_path.write_text(json.dumps(manifest))

    @staticmethod
    def _artifact(path: str, purpose: str, payload: bytes) -> dict:
        return {
            "path": path,
            "purpose": purpose,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def prepare(self):
        return swipe_data.prepare(
            source_root=self.source_root,
            output_root=self.output_root,
            manifest_path=self.manifest_path,
            policy_path=self.policy_path,
        )


class PrepareSwipeDatasetTest(unittest.TestCase):
    def test_policy_is_pinned_to_the_runtime_tensor_abi(self):
        policy = swipe_data.load_policy()
        self.assertEqual(64, policy.path_points)
        self.assertEqual(64, policy.key_slots)
        self.assertEqual(32, policy.output_frames)
        self.assertEqual(10_000, sum(policy.split_basis_points.values()))

    def test_preparation_is_session_separated_canonical_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            temporary_policy_path = root / "lookup-policy.json"
            temporary_policy_path.write_text(json.dumps(policy_document()))
            policy = swipe_data.load_policy(temporary_policy_path)
            sessions = {split: find_session(split, policy) for split in swipe_data.SPLITS}
            rows = {
                "train.jsonl": [row(1, sessions["train"], "qt")],
                "dev.jsonl": [row(2, sessions["validation"], "qt")],
                "test.jsonl": [row(3, sessions["test"], "pop", [
                    {"x": 0.95, "y": 1 / 6},
                    {"x": 0.85, "y": 1 / 6},
                    {"x": 0.95, "y": 1 / 6},
                ])],
            }
            fixture = PreparedFixture(root, rows)
            report = fixture.prepare()

            self.assertEqual(3, report["counts"]["acceptedRows"])
            self.assertEqual({"train": 1, "validation": 1, "test": 1}, {
                split: report["outputs"][f"{split}.jsonl"]["records"]
                for split in swipe_data.SPLITS
            })
            test_record = json.loads((fixture.output_root / "test.jsonl").read_text())
            self.assertEqual(128, len(test_record["pathCoordinates"]))
            self.assertEqual("test", test_record["split"])
            self.assertIn("short", test_record["strata"])
            self.assertIn("return_trip", test_record["strata"])
            self.assertRegex(test_record["sessionId"], r"^[0-9a-f]{64}$")
            layout_value = json.loads((fixture.output_root / "layout.json").read_text())
            self.assertEqual(list("qwertyuiop"), layout_value["keyLabels"][:10])
            self.assertEqual(64, len(layout_value["keyMask"]))

            hashes = {
                name: model_sources.file_sha256(fixture.output_root / name)
                for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "layout.json")
            }
            report_again = fixture.prepare()
            self.assertEqual(report["outputs"], report_again["outputs"])
            self.assertEqual(hashes, {
                name: model_sources.file_sha256(fixture.output_root / name)
                for name in hashes
            })

    def test_duplicate_and_invalid_rows_are_counted_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            temporary_policy_path = root / "lookup-policy.json"
            temporary_policy_path.write_text(json.dumps(policy_document()))
            policy = swipe_data.load_policy(temporary_policy_path)
            sessions = {split: find_session(split, policy) for split in swipe_data.SPLITS}
            duplicate = row(1, sessions["train"], "qt")
            fixture = PreparedFixture(root, {
                "train.jsonl": [duplicate, duplicate],
                "dev.jsonl": [row(2, sessions["validation"], "qt")],
                "test.jsonl": [
                    row(3, sessions["test"], "qt"),
                    row(4, sessions["test"], "qt", [{"x": 0.1, "y": 0.1}]),
                ],
            })

            report = fixture.prepare()

            self.assertEqual(1, report["counts"]["duplicateRows"])
            self.assertEqual(1, report["counts"]["rejectedRows"])
            self.assertEqual(1, report["counts"]["rejections"]["invalid_point_count"])

    def test_wrong_source_hash_cannot_publish_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            temporary_policy_path = root / "lookup-policy.json"
            temporary_policy_path.write_text(json.dumps(policy_document()))
            policy = swipe_data.load_policy(temporary_policy_path)
            sessions = {split: find_session(split, policy) for split in swipe_data.SPLITS}
            fixture = PreparedFixture(root, {
                "train.jsonl": [row(1, sessions["train"], "qt")],
                "dev.jsonl": [row(2, sessions["validation"], "qt")],
                "test.jsonl": [row(3, sessions["test"], "qt")],
            })
            source_file = fixture.source_root / "fixture-swipe" / "test.jsonl"
            payload = source_file.read_bytes()
            source_file.write_bytes(payload[:-1] + b" ")

            with self.assertRaisesRegex(swipe_data.SwipeDataError, "wrong SHA-256"):
                fixture.prepare()
            self.assertFalse(fixture.output_root.exists())


if __name__ == "__main__":
    unittest.main()
