# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib.util
import pathlib
import sys
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


if __name__ == "__main__":
    unittest.main()
