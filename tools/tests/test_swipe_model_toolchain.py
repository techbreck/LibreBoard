# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path[:0] = [str(TOOLS), str(pathlib.Path(__file__).resolve().parent), str(ROOT)]
import export_swipe_model  # noqa: E402
import evaluate_swipe_ctc  # noqa: E402
import prepare_swipe_dataset as swipe_data  # noqa: E402
import train_swipe_model  # noqa: E402
from test_prepare_swipe_dataset import PreparedFixture, find_session, policy_document, row  # noqa: E402


HAS_TOOLCHAIN = all(importlib.util.find_spec(name) is not None for name in ("torch", "onnx", "safetensors"))


@unittest.skipUnless(HAS_TOOLCHAIN, "optional model toolchain is not installed")
class SwipeModelToolchainTest(unittest.TestCase):
    def test_prepared_data_trains_and_exports_a_checked_development_onnx(self):
        with tempfile.TemporaryDirectory(prefix="libreboard-model-e2e-") as temporary:
            work = pathlib.Path(temporary)
            lookup_policy = work / "lookup-policy.json"
            lookup_policy.write_text(json.dumps(policy_document()))
            policy = swipe_data.load_policy(lookup_policy)
            sessions = {split: find_session(split, policy) for split in swipe_data.SPLITS}
            fixture = PreparedFixture(work, {
                "train.jsonl": [
                    row(1, sessions["train"], "qt"),
                    row(2, sessions["train"], "pop", [
                        {"x": 0.95, "y": 1 / 6},
                        {"x": 0.85, "y": 1 / 6},
                        {"x": 0.95, "y": 1 / 6},
                    ]),
                ],
                "dev.jsonl": [row(3, sessions["validation"], "qt")],
                "test.jsonl": [row(4, sessions["test"], "qt")],
            })
            fixture.prepare()
            training_root = work / "training"
            training = train_swipe_model.train(argparse.Namespace(
                spec=ROOT / "models" / "swipe" / "model-spec.json",
                data_root=fixture.output_root,
                output_root=training_root,
                device="cpu",
                threads=1,
                max_train_steps=1,
                max_validation_steps=1,
                allow_dirty=True,
            ))
            export_root = work / "export"
            exported = export_swipe_model.export(argparse.Namespace(
                spec=ROOT / "models" / "swipe" / "model-spec.json",
                training_report=training_root / "training-report-development.json",
                output_root=export_root,
                development=True,
            ))

            self.assertFalse(training["releaseEligible"])
            self.assertFalse(exported["releaseEligible"])
            self.assertLessEqual(exported["model"]["bytes"], 2_621_440)
            self.assertGreater(exported["fp16StoredInitializerCount"], 0)
            self.assertTrue((export_root / "swipe-latin-v1-development.onnx").is_file())
            self.assertTrue((export_root / "required_operators-development.config").is_file())
            self.assertTrue((export_root / "model.json").is_file())
            self.assertTrue((export_root / "split-manifest.json").is_file())
            self.assertTrue((export_root / "training-report-development.json").is_file())
            manifest = json.loads((export_root / "manifest-development.json").read_text())
            self.assertEqual(exported["model"]["sha256"], manifest["modelSha256"])
            self.assertIn("MatMul", manifest["requiredOnnxOperators"])

            loaded, model_path, _report_hash = evaluate_swipe_ctc.load_export(
                export_root / "export-report-development.json",
                development=True,
            )
            numpy, onnxruntime, _versions = evaluate_swipe_ctc._dependencies()
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            prepared_layout = json.loads((fixture.output_root / "layout.json").read_text())
            output = session.run(["logits"], {
                "path_coordinates": numpy.zeros((1, 64, 2), dtype=numpy.float32),
                "key_centers": numpy.asarray(prepared_layout["keyCenters"], dtype=numpy.float32).reshape(1, 64, 2),
                "key_mask": numpy.asarray(prepared_layout["keyMask"], dtype=numpy.float32).reshape(1, 64),
            })[0]
            self.assertEqual(exported["model"]["sha256"], loaded["model"]["sha256"])
            self.assertEqual((1, 32, 65), output.shape)
            self.assertTrue(numpy.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
