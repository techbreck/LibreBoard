import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_onnxruntime_android as builder


class OnnxRuntimeBuildTest(unittest.TestCase):
    def test_committed_settings_are_the_exact_cpu_only_configuration(self):
        settings = builder.load_settings()
        self.assertEqual(builder.EXPECTED_ABIS, settings["abis"])
        self.assertEqual(builder.REQUIRED_PARAMETERS, set(settings["buildParameters"]))
        self.assertFalse(any(
            marker in " ".join(settings["buildParameters"]).lower()
            for marker in builder.FORBIDDEN_PARAMETER_FRAGMENTS
        ))

    def test_settings_reject_provider_enablement_and_schema_drift(self):
        settings = builder.load_settings()
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "settings.json"
            settings["buildParameters"].append("--use_nnapi")
            path.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(builder.BuildConfigurationError):
                builder.load_settings(path)

            settings.pop("androidTargetSdk")
            path.write_text(json.dumps(settings), encoding="utf-8")
            with self.assertRaises(builder.BuildConfigurationError):
                builder.load_settings(path)

    def test_operator_configuration_must_be_nonempty_and_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "ops.config"
            path.write_text("# generated\nai.onnx;18;Add,MatMul,Softmax\n", encoding="utf-8")
            builder.validate_ops_config(path)

            path.write_text("ai.onnx;18;Add\nmalformed\n", encoding="utf-8")
            with self.assertRaises(builder.BuildConfigurationError):
                builder.validate_ops_config(path)

            path.write_text("# no operators\n", encoding="utf-8")
            with self.assertRaises(builder.BuildConfigurationError):
                builder.validate_ops_config(path)


if __name__ == "__main__":
    unittest.main()
