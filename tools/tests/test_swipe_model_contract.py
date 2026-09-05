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

    def test_release_corpus_must_match_the_committed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            data_root = pathlib.Path(temp)
            manifest = json.loads(contract.DEFAULT_CORPUS_MANIFEST.read_text())
            for filename, details in manifest["outputs"].items():
                path = data_root / filename
                path.write_bytes(b"fixture\n")
                details["bytes"] = path.stat().st_size
                details["sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                if filename.endswith(".jsonl"):
                    details["records"] = 1
                    details["sessions"] = 1
            (data_root / "split-manifest.json").write_text(json.dumps(manifest))

            contract.load_prepared_manifest(data_root)
            with self.assertRaisesRegex(contract.SwipeModelContractError, "does not match"):
                contract.load_prepared_manifest(data_root, require_pinned=True)

            raw = json.loads(contract.DEFAULT_SPEC.read_text())
            raw["export"]["inputs"][0]["shape"] = [1, 63, 2]
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(contract.SwipeModelContractError, "violates the fixed ABI"):
                contract.load_spec(path)


if __name__ == "__main__":
    unittest.main()
