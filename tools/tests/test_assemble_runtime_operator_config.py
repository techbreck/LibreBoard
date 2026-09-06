# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import assemble_runtime_operator_config as assembler  # noqa: E402


def _write_export(root: pathlib.Path, model_id: str, release_eligible: bool = False) -> pathlib.Path:
    kind, tensor_abi = assembler.MODEL_IDENTITIES[model_id]
    if model_id == "swipe-latin-v1":
        config_lines = ["ai.onnx;18;Add,MatMul"]
        operators = ["Add", "MatMul"]
    else:
        config_lines = ["ai.onnx;21;Add,Cast", "com.microsoft;1;MatMulNBits"]
        operators = ["Add", "Cast", "com.microsoft::MatMulNBits"]
    model = root / f"{model_id}.onnx"
    model.write_bytes(model_id.encode("ascii"))
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    manifest = root / f"{model_id}-manifest.json"
    manifest.write_text(json.dumps({
        "schemaVersion": 1,
        "engineAbi": 1,
        "modelKind": kind,
        "tensorAbi": tensor_abi,
        "modelSha256": model_hash,
        "requiredOnnxOperators": operators,
    }, sort_keys=True), encoding="utf-8")
    config = root / f"{model_id}-operators.config"
    config.write_text(
        "# Generated from the exact LibreBoard model graph; do not edit by hand.\n"
        + "\n".join(config_lines) + "\n",
        encoding="utf-8",
    )
    report = root / f"{model_id}-export.json"
    report.write_text(json.dumps({
        "schemaVersion": 1,
        "modelId": model_id,
        "releaseEligible": release_eligible,
        "appCommit": "a" * 40,
        "model": {
            "file": model.name,
            "bytes": model.stat().st_size,
            "sha256": model_hash,
        },
        "manifest": manifest.name,
        "requiredOperators": config.name,
    }, sort_keys=True), encoding="utf-8")
    return report


class AssembleRuntimeOperatorConfigTest(unittest.TestCase):
    def test_type_reduction_is_required_explicitly_and_bound_to_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            arguments = argparse.Namespace(
                swipe_export_report=_write_export(root, "swipe-latin-v1"),
                context_export_report=_write_export(root, "context-en-de-v1"),
                output_root=root / "output",
                development=True,
                type_reduction=True,
            )
            typed_config = (
                b"# Generated from the exact LibreBoard ONNX graphs plus deterministic raw/optimized "
                b"type analysis; do not edit by hand.\n"
                b'ai.onnx;18;Add{"inputs":{"0":["float"]}},MatMul\n'
            )
            details = {
                "enabled": True,
                "onnxRuntimeVersion": "1.26.0",
                "optimizationLevels": ["disable", "all"],
                "targetPlatform": "arm",
                "convertedModelCount": 4,
                "annotatedOperatorEntries": 1,
            }
            with mock.patch.object(
                assembler,
                "_generate_type_reduced_config",
                return_value=(typed_config, details),
            ) as generator:
                report = assembler.assemble(arguments)

            self.assertEqual(details, report["typeReduction"])
            self.assertEqual(hashlib.sha256(typed_config).hexdigest(), report["operators"]["sha256"])
            self.assertEqual(
                typed_config,
                (arguments.output_root / report["operators"]["file"]).read_bytes(),
            )
            self.assertEqual(1, generator.call_count)
            self.assertEqual(
                {"context-en-de-v1", "swipe-latin-v1"},
                set(generator.call_args.args[0]),
            )

    def test_typed_operator_inventory_is_parsed_and_canonicalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "typed.config"
            path.write_text(
                "# upstream generator comment\n"
                'ai.onnx;18;Add{"inputs":{"0":["float"]}},MatMul\n'
                'ai.onnx;21;Add{"inputs":{"0":["float"]}},FusedMatMul\n',
                encoding="utf-8",
            )
            typed = assembler._parse_typed_config(path)
            raw = {
                ("ai.onnx", 18): {"Add", "Constant", "MatMul"},
                ("ai.onnx", 21): {"Add"},
            }
            config, annotations = assembler._canonical_typed_config(raw, typed)

            self.assertEqual(2, annotations)
            self.assertEqual(
                "# Generated from the exact LibreBoard ONNX graphs plus deterministic raw/optimized type analysis; do not edit by hand.\n"
                'ai.onnx;18;Add{"inputs":{"0":["float"]}},Constant,MatMul\n'
                'ai.onnx;21;Add{"inputs":{"0":["float"]}},FusedMatMul\n',
                config.decode("utf-8"),
            )

    def test_typed_config_rejects_ambiguous_or_unsafe_metadata(self):
        rejected = (
            'ai.onnx;18;Add{"inputs":{"0":["float","float"]}}\n',
            'ai.onnx;18;Add{"inputs":{"00":["float"]}}\n',
            'ai.onnx;18;Add{"inputs":{"0":["string"]}}\n',
            'ai.onnx;18;Add{"custom":{"0":["float"]}}\n',
            'ai.onnx;18;Add{"inputs":{"0":["float"],"0":["float"]}}\n',
            "ai.onnx;18;Add,Add\n",
            'example.invalid;1;Add{"inputs":{"0":["float"]}}\n',
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "typed.config"
            for content in rejected:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(assembler.RuntimeOperatorConfigError):
                        assembler._parse_typed_config(path)

    def test_type_annotations_must_not_conflict_across_opsets(self):
        raw = {
            ("ai.onnx", 18): {"Add"},
            ("ai.onnx", 21): {"Add"},
        }
        typed = {
            ("ai.onnx", 18): {"Add": {"inputs": {"0": ["float"]}}},
            ("ai.onnx", 21): {"Add": {"inputs": {"0": ["int64_t"]}}},
        }
        with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "conflicting type metadata"):
            assembler._canonical_typed_config(raw, typed)

    def test_development_exports_compose_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            arguments = argparse.Namespace(
                swipe_export_report=_write_export(root, "swipe-latin-v1"),
                context_export_report=_write_export(root, "context-en-de-v1"),
                output_root=root / "output",
                development=True,
            )
            first = assembler.assemble(arguments)
            first_config = (arguments.output_root / first["operators"]["file"]).read_bytes()
            second = assembler.assemble(arguments)
            self.assertEqual(first, second)
            self.assertEqual(first_config, (arguments.output_root / second["operators"]["file"]).read_bytes())
            self.assertEqual(4, first["operators"]["uniqueOperators"])
            self.assertEqual(3, first["operators"]["domainOpsets"])
            self.assertEqual(
                "# Generated from the exact LibreBoard swipe-latin-v1 and context-en-de-v1 model graphs; do not edit by hand.\n"
                "ai.onnx;18;Add,MatMul\n"
                "ai.onnx;21;Add,Cast\n"
                "com.microsoft;1;MatMulNBits\n",
                first_config.decode("utf-8"),
            )

    def test_release_rejects_development_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "not release eligible"):
                assembler.assemble(argparse.Namespace(
                    swipe_export_report=_write_export(root, "swipe-latin-v1"),
                    context_export_report=_write_export(root, "context-en-de-v1", release_eligible=True),
                    output_root=root / "output",
                    development=False,
                    type_reduction=True,
                ))

    def test_release_rejects_untyped_operator_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "requires type reduction"):
                assembler.assemble(argparse.Namespace(
                    swipe_export_report=_write_export(
                        root,
                        "swipe-latin-v1",
                        release_eligible=True,
                    ),
                    context_export_report=_write_export(
                        root,
                        "context-en-de-v1",
                        release_eligible=True,
                    ),
                    output_root=root / "output",
                    development=False,
                    type_reduction=False,
                ))

    def test_rejects_model_tamper_and_manifest_operator_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            swipe = _write_export(root, "swipe-latin-v1")
            (root / "swipe-latin-v1.onnx").write_bytes(b"tampered")
            with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "does not match"):
                assembler._load_export(
                    swipe,
                    expected_model_id="swipe-latin-v1",
                    development=True,
                )

            context = _write_export(root, "context-en-de-v1")
            manifest = root / "context-en-de-v1-manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["requiredOnnxOperators"].append("Sub")
            value["requiredOnnxOperators"].sort()
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "does not match its signed manifest"):
                assembler._load_export(
                    context,
                    expected_model_id="context-en-de-v1",
                    development=True,
                )

    def test_rejects_unexpected_opset_and_non_string_manifest_operator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            swipe = _write_export(root, "swipe-latin-v1")
            config = root / "swipe-latin-v1-operators.config"
            config.write_text(
                "# Generated from the exact LibreBoard model graph; do not edit by hand.\n"
                "ai.onnx;19;Add,MatMul\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "unexpected domains or opsets"):
                assembler._load_export(
                    swipe,
                    expected_model_id="swipe-latin-v1",
                    development=True,
                )

            context = _write_export(root, "context-en-de-v1")
            manifest = root / "context-en-de-v1-manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["requiredOnnxOperators"] = [{"operator": "Add"}]
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(assembler.RuntimeOperatorConfigError, "invalid runtime contract"):
                assembler._load_export(
                    context,
                    expected_model_id="context-en-de-v1",
                    development=True,
                )


if __name__ == "__main__":
    unittest.main()
