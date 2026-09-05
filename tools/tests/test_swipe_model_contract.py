# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import swipe_model_contract as contract  # noqa: E402


class SwipeModelContractTest(unittest.TestCase):
    def test_committed_spec_matches_the_fixed_runtime_abi_and_parameter_budget(self):
        spec = contract.load_spec()
        self.assertEqual(821_121, spec.raw["parameterCount"])
        self.assertEqual(spec.raw["parameterCount"], contract.expected_parameter_count(spec.architecture))
        self.assertEqual(2_621_440, spec.export["maximumModelBytes"])
        self.assertEqual(set(contract.EXPECTED_INPUTS), {
            tensor["name"] for tensor in spec.export["inputs"]
        })

    def test_spec_rejects_tensor_and_parameter_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            raw = json.loads(contract.DEFAULT_SPEC.read_text())
            raw["parameterCount"] += 1
            path = root / "model-spec.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(contract.SwipeModelContractError, "does not match architecture"):
                contract.load_spec(path)

            raw = json.loads(contract.DEFAULT_SPEC.read_text())
            raw["export"]["inputs"][0]["shape"] = [1, 63, 2]
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(contract.SwipeModelContractError, "violates the fixed ABI"):
                contract.load_spec(path)


if __name__ == "__main__":
    unittest.main()
