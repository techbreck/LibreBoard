# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import train_swipe_model as trainer  # noqa: E402


SHUFFLE_TEST_STRATA = [
    ["short", "clean"],
    ["long", "double_letter"],
    ["medium"],
    ["long", "double_letter"],
    ["short", "sloppy"],
    ["return_trip", "medium"],
]
SHUFFLE_TEST_BASELINE_ORDER = [
    "0007", "0001", "0000", "0009", "000a", "000b", "000d", "0005",
    "0003", "000c", "0011", "0012", "000f", "0008", "0015", "0002",
    "0010", "0006", "0017", "0004", "0013", "0014", "0016", "000e",
]


def _write_shuffle_fixture(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "train.jsonl"
    with path.open("wb") as stream:
        for index in range(24):
            record = {
                "schemaVersion": 1,
                "id": f"{index:064x}",
                "split": "train",
                "pathCoordinates": [0.1] * 128,
                "ctcLabels": [1, 2, 3],
                "strata": SHUFFLE_TEST_STRATA[index % 6],
            }
            stream.write(json.dumps(record).encode() + b"\n")
    return path


def _shuffled_ids(path, *, seed=1234, buffer_size=8, strata_weights=None):
    return [
        record["id"]
        for record in trainer._shuffled_records(
            path,
            split="train",
            seed=seed,
            buffer_size=buffer_size,
            path_points=64,
            key_slots=64,
            output_frames=32,
            strata_weights=strata_weights,
        )
    ]


class TrainSwipeModelTest(unittest.TestCase):
    def test_unweighted_shuffle_matches_the_baseline_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_shuffle_fixture(pathlib.Path(temporary))
            expected = [f"{int(suffix, 16):064x}" for suffix in SHUFFLE_TEST_BASELINE_ORDER]
            self.assertEqual(expected, _shuffled_ids(path))

    def test_integer_strata_weight_duplicates_only_matching_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_shuffle_fixture(pathlib.Path(temporary))
            counts = collections.Counter(
                _shuffled_ids(path, strata_weights={"double_letter": 3.0})
            )
            for index in range(24):
                identifier = f"{index:064x}"
                expected = 3 if "double_letter" in SHUFFLE_TEST_STRATA[index % 6] else 1
                self.assertEqual(expected, counts[identifier])

    def test_strata_weights_combine_by_max_not_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_shuffle_fixture(pathlib.Path(temporary))
            counts = collections.Counter(
                _shuffled_ids(path, strata_weights={"long": 3.0, "double_letter": 2.0})
            )
            for index in range(24):
                identifier = f"{index:064x}"
                expected = 3 if "long" in SHUFFLE_TEST_STRATA[index % 6] else 1
                self.assertEqual(expected, counts[identifier])

    def test_fractional_strata_weight_is_deterministic_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_shuffle_fixture(pathlib.Path(temporary))
            counts = collections.Counter(
                _shuffled_ids(path, seed=99, strata_weights={"short": 1.5})
            )
            short_total = 0
            for index in range(24):
                identifier = f"{index:064x}"
                if "short" in SHUFFLE_TEST_STRATA[index % 6]:
                    self.assertIn(counts[identifier], (1, 2))
                    short_total += counts[identifier]
                else:
                    self.assertEqual(1, counts[identifier])
            self.assertEqual(14, short_total)

    def test_parse_record_rejects_invalid_strata(self):
        record = {
            "schemaVersion": 1,
            "id": "0" * 64,
            "split": "train",
            "pathCoordinates": [0.1] * 128,
            "ctcLabels": [1, 2, 3],
        }
        for bad_strata in (None, [], "short", [1], [None]):
            candidate = dict(record)
            if bad_strata is not None:
                candidate["strata"] = bad_strata
            with self.assertRaises(trainer.SwipeTrainingError):
                trainer._parse_record(json.dumps(candidate).encode(), "train", 64, 64, 32)

    def test_ctc_collapse_preserves_repeats_only_across_blank(self):
        self.assertEqual([4], trainer._collapse_ctc([0, 4, 4, 0]))
        self.assertEqual([4, 4], trainer._collapse_ctc([4, 0, 4]))
        self.assertEqual([3, 8, 3], trainer._collapse_ctc([3, 3, 8, 3]))

    def test_edit_distance_handles_insert_delete_and_replace(self):
        self.assertEqual(0, trainer._edit_distance([1, 2, 3], [1, 2, 3]))
        self.assertEqual(1, trainer._edit_distance([1, 3], [1, 2, 3]))
        self.assertEqual(1, trainer._edit_distance([1, 4, 3], [1, 2, 3]))

    def test_default_environment_fails_with_an_actionable_dependency_error(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("the optional training environment is active")
        with self.assertRaisesRegex(trainer.SwipeTrainingError, "requirements-linux-x86_64.lock"):
            trainer._dependencies()

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_implemented_model_matches_the_spec_and_fixed_output_shape(self):
        import torch

        root = pathlib.Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from models.training.swipe_model import SwipeCtcModel, trainable_parameter_count
        import swipe_model_contract

        spec = swipe_model_contract.load_spec()
        model = SwipeCtcModel(spec.architecture).eval()
        with torch.inference_mode():
            output = model(
                torch.zeros((1, 64, 2), dtype=torch.float32),
                torch.zeros((1, 64, 2), dtype=torch.float32),
                torch.cat((torch.ones((1, 26)), torch.zeros((1, 38))), dim=1),
            )
        self.assertEqual(spec.raw["parameterCount"], trainable_parameter_count(model))
        self.assertEqual((1, 32, 65), tuple(output.shape))
        self.assertTrue(torch.isfinite(output).all())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_atomic_checkpoint_restores_model_optimizer_and_rng(self):
        import torch
        from safetensors.torch import save_file

        torch.manual_seed(7)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones((1, 2))).sum().backward()
        optimizer.step()
        expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
        expected_optimizer = {
            name: {
                key: value.detach().clone()
                for key, value in optimizer.state[parameter].items()
            }
            for name, parameter in model.named_parameters()
        }
        expected_rng = torch.get_rng_state().clone()
        bindings = trainer._checkpoint_bindings(
            app_commit="a" * 40,
            data_manifest_sha256="b" * 64,
            model_spec_sha256="c" * 64,
            device_type="cpu",
            torch_version=str(torch.__version__),
            threads=1,
        )
        reports = [{"epoch": 1, "meanCtcLoss": 1.25}]
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "checkpoint.safetensors"
            trainer._save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch_reports=reports,
                bindings=bindings,
                device=torch.device("cpu"),
                torch=torch,
                save_file=save_file,
            )
            restored_model = torch.nn.Linear(2, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            completed, restored_reports, app_commit = trainer._restore_checkpoint(
                path,
                model=restored_model,
                optimizer=restored_optimizer,
                expected_bindings=bindings,
                device=torch.device("cpu"),
                torch=torch,
            )

            self.assertEqual(1, completed)
            self.assertEqual(reports, restored_reports)
            self.assertEqual("a" * 40, app_commit)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for name, value in restored_model.state_dict().items():
                self.assertTrue(torch.equal(expected_model[name], value))
            for name, parameter in restored_model.named_parameters():
                for key, value in restored_optimizer.state[parameter].items():
                    self.assertTrue(torch.equal(expected_optimizer[name][key], value))

            mismatched = dict(bindings)
            mismatched["threads"] = "2"
            with self.assertRaisesRegex(trainer.SwipeTrainingError, "threads"):
                trainer._restore_checkpoint(
                    path,
                    model=restored_model,
                    optimizer=restored_optimizer,
                    expected_bindings=mismatched,
                    device=torch.device("cpu"),
                    torch=torch,
                )


if __name__ == "__main__":
    unittest.main()
