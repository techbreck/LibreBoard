# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path[:0] = [str(TOOLS), str(ROOT)]
import context_model_contract as contract  # noqa: E402


class ContextModelContractTest(unittest.TestCase):
    def test_committed_spec_matches_android_abi_and_exact_parameter_budget(self):
        spec = contract.load_spec()
        self.assertEqual(35_662_848, spec.raw["parameterCount"])
        self.assertEqual(spec.raw["parameterCount"], contract.expected_parameter_count(spec.architecture))
        self.assertEqual(25_165_824, spec.export["maximumModelBytes"])
        self.assertEqual(set(contract.EXPECTED_INPUTS), {
            tensor["name"] for tensor in spec.export["inputs"]
        })
        self.assertEqual(
            {"teacherLossWeight": 1.0, "observedLossWeight": 0.5, "rankingLossWeight": 0.25},
            {key: spec.training[key] for key in (
                "teacherLossWeight", "observedLossWeight", "rankingLossWeight",
            )},
        )
        self.assertEqual(4096, spec.training["checkpointEveryExamples"])

    def test_small_candidate_preserves_tensor_abi_and_exact_parameter_count(self):
        large = contract.load_spec()
        small = contract.load_spec(contract.DEFAULT_SPEC.with_name("model-spec-small-candidate.json"))
        self.assertEqual(7_605_760, small.raw["parameterCount"])
        self.assertEqual(small.raw["parameterCount"], contract.expected_parameter_count(small.architecture))
        self.assertEqual(large.export, small.export)
        self.assertEqual(large.training, small.training)

    def test_spec_rejects_architecture_and_tensor_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "model-spec.json"
            raw = json.loads(contract.DEFAULT_SPEC.read_text())
            raw["architecture"]["width"] = 256
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(contract.ContextModelContractError, "approved candidate profile"):
                contract.load_spec(path)

            raw = json.loads(contract.DEFAULT_SPEC.read_text())
            raw["export"]["inputs"][0]["shape"] = [-1, 31]
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(contract.ContextModelContractError, "violates the fixed ABI"):
                contract.load_spec(path)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_model_matches_spec_and_produces_one_finite_score_per_candidate(self):
        import torch
        from models.training.context_model import ContextCandidateModel, trainable_parameter_count

        spec = contract.load_spec()
        model = ContextCandidateModel(spec.architecture).eval()
        input_ids = torch.zeros((2, 32), dtype=torch.long)
        input_ids[:, :3] = torch.tensor([1, 3, 7])
        input_ids[:, 24:26] = torch.tensor([8, 9])
        attention_mask = torch.zeros((2, 32), dtype=torch.long)
        attention_mask[:, :3] = 1
        attention_mask[:, 24:26] = 1
        candidate_mask = torch.zeros((2, 32), dtype=torch.float32)
        candidate_mask[:, 24:26] = 1
        field_class = torch.tensor([0, 2], dtype=torch.long)
        with torch.inference_mode():
            output = model(input_ids, attention_mask, candidate_mask, field_class)

        self.assertEqual(spec.raw["parameterCount"], trainable_parameter_count(model))
        self.assertEqual((2,), tuple(output.shape))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
