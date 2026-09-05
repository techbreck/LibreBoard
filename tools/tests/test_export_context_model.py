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
sys.path.insert(0, str(TOOLS))
import context_model_contract  # noqa: E402
import export_context_model as exporter  # noqa: E402
import model_sources  # noqa: E402


def report(spec: context_model_contract.ContextModelSpec) -> dict:
    teacher = model_sources.load_manifest().source("hanse2-100m-base-teacher-v1")
    return {
        "schemaVersion": 1,
        "modelId": "context-en-de-v1",
        "releaseEligible": False,
        "appCommit": "a" * 40,
        "modelSpecSha256": spec.sha256,
        "dataManifestSha256": "b" * 64,
        "parameterCount": spec.raw["parameterCount"],
        "teacher": {"modelSha256": teacher.artifact("model.safetensors").sha256},
        "provenance": [{
            "name": "fixture",
            "revision": teacher.revision,
            "license": teacher.license,
            "source_url": teacher.source_url,
        }],
        "weights": {"file": "weights.safetensors", "bytes": 1, "sha256": "d" * 64},
        "tokenizer": {"file": "tokenizer.json", "bytes": 1, "sha256": "e" * 64},
    }


class ExportContextModelTest(unittest.TestCase):
    def test_development_report_retains_all_export_bindings(self):
        spec = context_model_contract.load_spec()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "training-report.json"
            path.write_text(json.dumps(report(spec)), encoding="utf-8")
            loaded = exporter._load_training_report(path, True, spec)
            self.assertEqual(spec.sha256, loaded["modelSpecSha256"])

    def test_release_rejects_partial_training_and_unsafe_artifact_path(self):
        spec = context_model_contract.load_spec()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "training-report.json"
            value = report(spec)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(exporter.ContextExportError, "cannot produce a release"):
                exporter._load_training_report(path, False, spec)

            value["weights"]["file"] = "../weights.safetensors"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(exporter.ContextExportError, "invalid weights filename"):
                exporter._load_training_report(path, True, spec)

    def test_rejects_model_spec_and_teacher_hash_drift(self):
        spec = context_model_contract.load_spec()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "training-report.json"
            value = report(spec)
            value["modelSpecSha256"] = "0" * 64
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(exporter.ContextExportError, "different model spec"):
                exporter._load_training_report(path, True, spec)

            value = report(spec)
            value["teacher"]["modelSha256"] = "not-a-hash"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(exporter.ContextExportError, "teacher model hash"):
                exporter._load_training_report(path, True, spec)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("torch", "onnx", "onnx_ir", "onnxruntime")),
        "optional context model export toolchain is not installed",
    )
    def test_synthetic_weights_export_to_bounded_int4_runtime_graph(self):
        import torch
        from safetensors.torch import save_file

        from models.training.context_model import ContextCandidateModel

        spec = context_model_contract.load_spec()
        teacher = model_sources.load_manifest().source("hanse2-100m-base-teacher-v1")
        with tempfile.TemporaryDirectory(prefix="libreboard-context-export-test-") as temporary:
            root = pathlib.Path(temporary)
            tokenizer_path = root / "tokenizer.json"
            vocabulary = {
                "<pad>": 0,
                "<bos>": 1,
                "<unk>": 2,
                "<lang:en>": 3,
                "<lang:de>": 4,
                "▁": 5,
            }
            vocabulary.update({f"token-{index:05d}": index for index in range(6, 16_384)})
            tokenizer_path.write_text(json.dumps({
                "schemaVersion": 1,
                "normalization": "NFKC_LOWER",
                "vocabulary": vocabulary,
                "merges": [],
                "specialTokens": {
                    "padding": "<pad>",
                    "beginningOfSequence": "<bos>",
                    "unknown": "<unk>",
                    "languages": {"en": "<lang:en>", "de": "<lang:de>"},
                },
            }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            tokenizer_sha = model_sources.file_sha256(tokenizer_path)
            data_manifest_sha = "b" * 64
            app_commit = "a" * 40
            weights_path = root / "weights.safetensors"
            torch.manual_seed(7)
            model = ContextCandidateModel(spec.architecture)
            save_file(model.state_dict(), weights_path, metadata={
                "appCommit": app_commit,
                "dataManifestSha256": data_manifest_sha,
                "modelSpecSha256": spec.sha256,
                "teacherModelSha256": teacher.artifact("model.safetensors").sha256,
                "tokenizerSha256": tokenizer_sha,
            })
            report_path = root / "training-report-development.json"
            value = report(spec)
            value.update({
                "appCommit": app_commit,
                "dataManifestSha256": data_manifest_sha,
                "weights": {
                    "file": weights_path.name,
                    "bytes": weights_path.stat().st_size,
                    "sha256": model_sources.file_sha256(weights_path),
                },
                "tokenizer": {
                    "file": tokenizer_path.name,
                    "bytes": tokenizer_path.stat().st_size,
                    "sha256": tokenizer_sha,
                },
            })
            report_path.write_text(json.dumps(value), encoding="utf-8")

            result = exporter.export(argparse.Namespace(
                spec=spec.path,
                training_report=report_path,
                output_root=root / "export",
                development=True,
            ))

            self.assertFalse(result["releaseEligible"])
            self.assertLessEqual(result["model"]["bytes"], spec.export["maximumModelBytes"])
            self.assertEqual(
                {
                    "com.microsoft::GatherBlockQuantized": 4,
                    "com.microsoft::MatMulNBits": 56,
                },
                result["customOperatorCounts"],
            )


if __name__ == "__main__":
    unittest.main()
