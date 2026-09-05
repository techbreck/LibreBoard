# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import canonical_safetensors  # noqa: E402


class CanonicalSafetensorsTest(unittest.TestCase):
    def test_rejects_truncated_and_linked_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            truncated = root / "truncated.safetensors"
            truncated.write_bytes(b"bad")
            with self.assertRaisesRegex(canonical_safetensors.CanonicalSafetensorsError, "empty, or too large"):
                canonical_safetensors.canonicalize(truncated, maximum_bytes=1024)
            linked = root / "linked.safetensors"
            linked.symlink_to(truncated)
            with self.assertRaisesRegex(canonical_safetensors.CanonicalSafetensorsError, "linked"):
                canonical_safetensors.canonicalize(linked, maximum_bytes=1024)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_canonical_header_is_byte_identical_for_different_metadata_order(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            tensors = {"weight": torch.arange(4, dtype=torch.float32)}
            save_file(tensors, first, metadata={"z": "last", "a": "first"})
            save_file(tensors, second, metadata={"a": "first", "z": "last"})
            canonical_safetensors.canonicalize(first, maximum_bytes=1024 * 1024)
            canonical_safetensors.canonicalize(second, maximum_bytes=1024 * 1024)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with safe_open(first, framework="pt", device="cpu") as data:
                self.assertEqual({"a": "first", "z": "last"}, data.metadata())
                self.assertTrue(torch.equal(tensors["weight"], data.get_tensor("weight")))


if __name__ == "__main__":
    unittest.main()
