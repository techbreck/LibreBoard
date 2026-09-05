import io
import json
import pathlib
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_onnxruntime_android as builder


class OnnxRuntimeBuildTest(unittest.TestCase):
    def test_process_exhaustion_is_reported_as_a_build_error(self):
        with mock.patch.object(
            builder.subprocess,
            "run",
            side_effect=BlockingIOError(35, "Resource temporarily unavailable"),
        ):
            with self.assertRaisesRegex(builder.BuildConfigurationError, "cannot start command"):
                builder.run(["fixture"], cwd=pathlib.Path.cwd())

    def test_python_build_toolchain_is_exact_and_rejects_drift(self):
        builder.validate_python_toolchain(
            python_version=(3, 11),
            package_versions=dict(builder.EXPECTED_PYTHON_PACKAGES),
        )
        drifted = dict(builder.EXPECTED_PYTHON_PACKAGES)
        drifted["flatbuffers"] = "0.0.0"
        with self.assertRaises(builder.BuildConfigurationError):
            builder.validate_python_toolchain(
                python_version=(3, 11),
                package_versions=drifted,
            )
        with self.assertRaises(builder.BuildConfigurationError):
            builder.validate_python_toolchain(
                python_version=(3, 12),
                package_versions=dict(builder.EXPECTED_PYTHON_PACKAGES),
            )

    def test_committed_settings_are_the_exact_cpu_only_configuration(self):
        settings = builder.load_settings()
        self.assertEqual(builder.EXPECTED_ABIS, settings["abis"])
        self.assertEqual(builder.EXPECTED_NDK_REVISION, settings["ndkRevision"])
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

    def test_parallel_flag_is_audited_and_accepts_an_explicit_bound(self):
        settings = builder.load_settings()
        parameters = builder.resolved_build_parameters(settings, 2)
        self.assertIn("--parallel=2", parameters)
        self.assertNotIn("--parallel", parameters)
        with self.assertRaises(builder.BuildConfigurationError):
            builder.resolved_build_parameters(settings, 0)

    def test_aar_packaging_is_canonical_across_order_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            first_source = root / "first-input.aar"
            second_source = root / "second-input.aar"
            first_output = root / "first-output.aar"
            second_output = root / "second-output.aar"

            def nested_jar(timestamp, reverse=False):
                output = io.BytesIO()
                nested_entries = [("A.class", b"a"), ("B.class", b"b")]
                with zipfile.ZipFile(output, "w") as archive:
                    for name, payload in reversed(nested_entries) if reverse else nested_entries:
                        info = zipfile.ZipInfo(name, timestamp)
                        info.compress_type = zipfile.ZIP_DEFLATED
                        archive.writestr(info, payload)
                return output.getvalue()

            first_entries = {
                "classes.jar": nested_jar((2020, 1, 2, 3, 4, 6)),
                "jni/arm64-v8a/libfixture.so": b"elf",
            }
            second_entries = {
                "classes.jar": nested_jar((2025, 6, 7, 8, 9, 10), reverse=True),
                "jni/arm64-v8a/libfixture.so": b"elf",
            }
            with zipfile.ZipFile(first_source, "w") as archive:
                for name, payload in first_entries.items():
                    info = zipfile.ZipInfo(name, (2020, 1, 2, 3, 4, 6))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.extra = b"\x01\x00\x00\x00"
                    archive.writestr(info, payload)
            with zipfile.ZipFile(second_source, "w") as archive:
                for name, payload in reversed(second_entries.items()):
                    info = zipfile.ZipInfo(name, (2025, 6, 7, 8, 9, 10))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, payload)

            builder.canonicalize_aar(first_source, first_output)
            builder.canonicalize_aar(second_source, second_output)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            with zipfile.ZipFile(first_output) as archive:
                self.assertEqual(sorted(first_entries), archive.namelist())
                self.assertTrue(all(
                    info.date_time == builder.CANONICAL_ZIP_TIMESTAMP
                    and not info.extra
                    and not info.comment
                    for info in archive.infolist()
                ))
                with zipfile.ZipFile(io.BytesIO(archive.read("classes.jar"))) as classes:
                    self.assertEqual(["A.class", "B.class"], classes.namelist())
                    self.assertTrue(all(
                        info.date_time == builder.CANONICAL_ZIP_TIMESTAMP
                        for info in classes.infolist()
                    ))


if __name__ == "__main__":
    unittest.main()
