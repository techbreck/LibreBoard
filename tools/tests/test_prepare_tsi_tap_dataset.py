# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import prepare_tsi_tap_dataset as tsi  # noqa: E402


def keyboard_document() -> dict:
    keys = {}
    names = [*"abcdefghijklmnopqrstuvwxyz", ".", "SPACE"]
    for index, name in enumerate(names):
        keys[name] = {
            "key_id": name,
            "text_literal": name,
            "key_center_x": float(index * 10 + 5),
            "key_center_y": 20.0,
            "key_width": 10.0,
            "key_height": 20.0,
        }
    return {
        "device_info": {},
        "keyboard_info": {"keyboard_width": 300.0, "keyboard_height": 100.0},
        "keys_info": keys,
    }


def write_csv(path: pathlib.Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def touch(index: int, intended: str, nearest: str, *, deleted: bool = False) -> dict:
    nearest_index = [*"abcdefghijklmnopqrstuvwxyz", ".", "SPACE"].index(nearest)
    value = {field: "ignored" for field in tsi.TOUCH_FIELDS}
    value.update({
        "participant_id": "user01",
        "task_id": "task1",
        "trial_id": "0",
        "timestamp_ms": str(1000 + index * 50),
        "ref_char": intended,
        "ref_char_index_in_prompt": str(index),
        "first_frame_touch_x": str(nearest_index * 10 + 5),
        "first_frame_touch_y": "20",
        "was_deleted": str(deleted),
        "lm_scores": "this is deliberately not parsed",
    })
    return value


class PrepareTsiTapDatasetTest(unittest.TestCase):
    def test_converts_only_complete_non_deleted_phrase_words(self):
        with tempfile.TemporaryDirectory() as temporary:
            keyboard = tsi.load_keyboard(self._write_keyboard(pathlib.Path(temporary)))
            prompts = {("user01", "task1", 0): {"type": "phrase", "text": "cat"}}
            trials = {("user01", "task1", 0): [
                tsi.Touch(0, 1000, "c", 25.0, 20.0, False),
                tsi.Touch(1, 1050, "a", 185.0, 20.0, False),  # nearest s
                tsi.Touch(2, 1100, "t", 195.0, 20.0, False),
            ]}

            records, counts = tsi.convert_records(prompts, trials, keyboard)

        self.assertEqual(1, len(records))
        self.assertEqual("cat", records[0]["target"])
        self.assertEqual("cst", records[0]["raw"])
        self.assertEqual("google-tsi:user01", records[0]["sessionId"])
        self.assertEqual([0, 50, 100], [point["timeMillis"] for point in records[0]["touchPoints"]])
        self.assertEqual(1, counts["tapErrorWords"])

    def test_loaders_reject_prompt_misalignment_and_ignore_lm_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            keyboard = tsi.load_keyboard(self._write_keyboard(root))
            prompt_path = root / "prompt_data.csv"
            write_csv(prompt_path, tsi.PROMPT_FIELDS, [{
                "participant_id": "user01", "task_id": "task1", "trial_id": "0",
                "prompt_type": "phrase", "prompt": "cat",
            }])
            prompts = tsi.load_prompts(prompt_path)
            touch_path = root / "touch_data.csv"
            rows = [touch(0, "c", "c"), touch(1, "a", "s"), touch(2, "t", "t")]
            write_csv(touch_path, tsi.TOUCH_FIELDS, rows)

            trials, counts = tsi.load_touches(touch_path, prompts, keyboard)

            self.assertEqual(3, counts["touches"])
            self.assertEqual(0, counts["outsideLayoutTouches"])
            self.assertEqual(3, len(trials[("user01", "task1", 0)]))
            rows[1]["ref_char"] = "b"
            write_csv(touch_path, tsi.TOUCH_FIELDS, rows)
            with self.assertRaisesRegex(tsi.TsiDataError, "disagrees with its prompt"):
                tsi.load_touches(touch_path, prompts, keyboard)

    def test_transposed_alignment_is_replayed_in_touch_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            keyboard = tsi.load_keyboard(self._write_keyboard(pathlib.Path(temporary)))
            prompts = {("user01", "task1", 0): {"type": "phrase", "text": "cat"}}
            trials = {("user01", "task1", 0): [
                tsi.Touch(0, 1000, "c", 25.0, 20.0, False),
                tsi.Touch(1, 1100, "a", 5.0, 20.0, False),
                tsi.Touch(2, 1050, "t", 195.0, 20.0, False),
            ]}

            records, _counts = tsi.convert_records(prompts, trials, keyboard)

        self.assertEqual("cta", records[0]["raw"])
        self.assertEqual([0, 50, 100], [point["timeMillis"] for point in records[0]["touchPoints"]])

    def test_prepare_verifies_every_pinned_source_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_root = root / "sources"
            source = source_root / tsi.SOURCE_ID
            source.mkdir(parents=True)
            (source / "LICENSE").write_text("license")
            (source / "README.md").write_text("readme")
            self._write_keyboard(source)
            write_csv(source / "prompt_data.csv", tsi.PROMPT_FIELDS, [{
                "participant_id": "user01", "task_id": "task1", "trial_id": "0",
                "prompt_type": "phrase", "prompt": "cat",
            }])
            write_csv(
                source / "touch_data.csv", tsi.TOUCH_FIELDS,
                [touch(0, "c", "c"), touch(1, "a", "s"), touch(2, "t", "t")],
            )
            artifacts = []
            for name, purpose in (
                ("LICENSE", "license"), ("README.md", "documentation"),
                ("keyboard_data.json", "layout"), ("prompt_data.csv", "evaluation-data"),
                ("touch_data.csv", "evaluation-data"),
            ):
                path = source / name
                artifacts.append({
                    "path": name, "purpose": purpose, "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
            manifest_path = root / "sources.json"
            manifest_path.write_text(json.dumps({
                "schemaVersion": 1,
                "sources": [{
                    "id": tsi.SOURCE_ID, "kind": "dataset", "repositoryType": "dataset",
                    "repository": "example/tap-data", "revision": "a" * 40,
                    "license": "CC-BY-4.0", "sourceUrl": "https://github.com/example/tap-data",
                    "artifacts": artifacts,
                }],
            }))
            output = root / "output"

            report = tsi.prepare(source_root, output, manifest_path)

            self.assertEqual(1, report["output"]["records"])
            derived = json.loads((output / "source-manifest.json").read_text())
            self.assertEqual("CC-BY-4.0", derived["license"])
            source.joinpath("touch_data.csv").write_text("tampered")
            with self.assertRaisesRegex(tsi.model_sources.ModelSourceError, "wrong size"):
                tsi.prepare(source_root, output, manifest_path)

    def _write_keyboard(self, root: pathlib.Path) -> pathlib.Path:
        path = root / "keyboard_data.json"
        path.write_text(json.dumps(keyboard_document()))
        return path


if __name__ == "__main__":
    unittest.main()
