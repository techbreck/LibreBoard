# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import train_swipe_model as trainer  # noqa: E402


class TrainSwipeModelTest(unittest.TestCase):
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
