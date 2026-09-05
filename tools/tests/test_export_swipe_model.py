# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import export_swipe_model as exporter  # noqa: E402


class ExportSwipeModelTest(unittest.TestCase):
    def test_operator_config_is_sorted_and_uses_the_runtime_format(self):
        required = {
            ("ai.onnx", 18): {"Softmax", "Add", "MatMul"},
            ("example.domain", 1): {"Fixture"},
        }
        self.assertEqual(
            "# Generated from the exact LibreBoard model graph; do not edit by hand.\n"
            "ai.onnx;18;Add,MatMul,Softmax\n"
            "example.domain;1;Fixture\n",
            exporter._operator_config(required),
        )
        self.assertEqual(
            ["Add", "MatMul", "Softmax", "example.domain::Fixture"],
            exporter._canonical_operator_list(required),
        )

    def test_default_environment_fails_with_an_actionable_dependency_error(self):
        if importlib.util.find_spec("onnx") is not None:
            self.skipTest("the optional export environment is active")
        with self.assertRaisesRegex(exporter.SwipeExportError, "requirements-linux-x86_64.lock"):
            exporter._dependencies()

    @unittest.skipUnless(importlib.util.find_spec("onnx") is not None, "optional model toolchain is not installed")
    def test_fp16_storage_transform_remains_fully_type_checkable(self):
        import numpy
        import onnx

        initializer = onnx.numpy_helper.from_array(
            numpy.arange(4, dtype=numpy.float32).reshape(2, 2),
            name="weight",
        )
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"])],
            "fixture",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
            [initializer],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=10,
        )

        converted = exporter._convert_float_initializers_to_fp16_storage(model, numpy, onnx)

        self.assertEqual(1, converted)
        self.assertEqual(onnx.TensorProto.FLOAT16, model.graph.initializer[0].data_type)
        self.assertEqual("Cast", model.graph.node[0].op_type)
        onnx.checker.check_model(model, full_check=True)


if __name__ == "__main__":
    unittest.main()
