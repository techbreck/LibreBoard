import errno
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import package_model as packager


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    for attempt in range(5):
        try:
            return subprocess.run(command, capture_output=True, check=False)
        except BlockingIOError as failure:
            if failure.errno != errno.EAGAIN or attempt == 4:
                raise
            time.sleep(0.1)
    raise AssertionError("bounded subprocess retry loop did not return")


class ModelPackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.key = self.root / "model-signing.pem"
        result = run_subprocess([
            "openssl", "genpkey", "-algorithm", "RSA",
            "-pkeyopt", "rsa_keygen_bits:3072", "-out", str(self.key),
        ])
        self.assertEqual(0, result.returncode, result.stderr.decode(errors="replace"))
        self.key.chmod(0o600)

    def tearDown(self):
        self.temporary.cleanup()

    def test_openssl_retries_only_transient_process_exhaustion(self):
        success = subprocess.CompletedProcess(["openssl"], 0, b"ok", b"")
        with (
            mock.patch.object(
                packager.subprocess,
                "run",
                side_effect=[BlockingIOError(errno.EAGAIN, "busy"), success],
            ) as process,
            mock.patch.object(packager.time, "sleep") as sleep,
        ):
            self.assertEqual(b"ok", packager._run_openssl(["openssl"]))
        self.assertEqual(2, process.call_count)
        sleep.assert_called_once_with(0.1)

        with mock.patch.object(
            packager.subprocess,
            "run",
            side_effect=OSError(errno.ENOENT, "missing"),
        ) as process:
            with self.assertRaisesRegex(packager.ModelPackagingError, "could not start OpenSSL"):
                packager._run_openssl(["openssl"])
        self.assertEqual(1, process.call_count)

    def test_development_package_is_canonical_signed_and_reproducible(self):
        export = self.root / "export"
        self._write_export(export)
        first = self.root / "first"
        second = self.root / "second"

        first_report = packager.package_model(export, self.key, first, development=True)
        second_report = packager.package_model(export, self.key, second, development=True)
        first_archive = first / first_report["archive"]["file"]
        second_archive = second / second_report["archive"]["file"]
        self.assertEqual(first_archive.read_bytes(), second_archive.read_bytes())
        self.assertEqual(packager.SIGNATURE_ALGORITHM, first_report["signatureAlgorithm"])
        self.assertEqual(first_report["archive"]["sha256"], second_report["archive"]["sha256"])
        self.assertEqual(
            (first / "libreboard-model-signing-public.der").read_bytes(),
            (second / "libreboard-model-signing-public.der").read_bytes(),
        )

        with zipfile.ZipFile(first_archive) as archive:
            self.assertEqual(
                ["manifest.json", "model.onnx", "signature.der"],
                archive.namelist(),
            )
            self.assertTrue(all(
                entry.date_time == packager.CANONICAL_TIMESTAMP
                and entry.compress_type == zipfile.ZIP_STORED
                and not entry.extra
                and not entry.comment
                for entry in archive.infolist()
            ))
            manifest = archive.read("manifest.json")
            signature = archive.read("signature.der")

        public_pem = self.root / "public.pem"
        conversion = run_subprocess([
            "openssl", "pkey", "-pubin", "-inform", "DER",
            "-in", str(first / "libreboard-model-signing-public.der"),
            "-pubout", "-out", str(public_pem),
        ])
        self.assertEqual(0, conversion.returncode, conversion.stderr.decode(errors="replace"))
        manifest_path = self.root / "manifest.json"
        signature_path = self.root / "signature.der"
        manifest_path.write_bytes(manifest)
        signature_path.write_bytes(signature)
        verification = run_subprocess([
            "openssl", "dgst", "-sha256", "-verify", str(public_pem),
            "-signature", str(signature_path), str(manifest_path),
        ])
        self.assertEqual(0, verification.returncode, verification.stderr.decode(errors="replace"))
        self.assertEqual(b"Verified OK\n", verification.stdout)

    def test_tampered_export_and_exposed_private_key_fail_closed(self):
        export = self.root / "export"
        self._write_export(export)
        (export / "swipe-latin-v1-development.onnx").write_bytes(b"tampered")
        with self.assertRaisesRegex(packager.ModelPackagingError, "does not match"):
            packager.package_model(export, self.key, self.root / "tampered", development=True)

        self._write_export(export)
        self.key.chmod(0o644)
        with self.assertRaisesRegex(packager.ModelPackagingError, "group or other"):
            packager.package_model(export, self.key, self.root / "exposed", development=True)

    def test_context_export_requires_hash_matched_tokenizer(self):
        export = self.root / "context"
        self._write_export(export, model_id="context-en-de-v1")
        report_path = export / "export-report-development.json"
        report = json.loads(report_path.read_text())
        report.pop("tokenizer")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(packager.ModelPackagingError, "no tokenizer descriptor"):
            packager.package_model(export, self.key, self.root / "context-output", development=True)

    def _write_export(self, root: pathlib.Path, model_id: str = "swipe-latin-v1"):
        root.mkdir(parents=True, exist_ok=True)
        is_context = model_id == "context-en-de-v1"
        model_name = f"{model_id}-development.onnx"
        model = ("model:" + model_id).encode()
        model_hash = hashlib.sha256(model).hexdigest()
        (root / model_name).write_bytes(model)
        manifest = {
            "schemaVersion": 1,
            "engineAbi": 1,
            "modelKind": "context-rescorer" if is_context else "swipe-ctc",
            "tensorAbi": model_id,
            "locales": ["en-US", "de"],
            "architecture": "fixture",
            "parameterCount": 1,
            "quantization": "fixture",
            "modelSha256": model_hash,
            "requiredOnnxOperators": ["MatMul"],
            "license": "Apache-2.0",
            "provenance": [{
                "name": "fixture",
                "revision": "1",
                "license": "Apache-2.0",
                "source_url": "https://example.invalid/fixture",
            }],
            "minimumAppVersionCode": 1,
        }
        report = {
            "schemaVersion": 1,
            "modelId": model_id,
            "releaseEligible": False,
            "model": {
                "file": model_name,
                "bytes": len(model),
                "sha256": model_hash,
            },
            "manifest": "manifest-development.json",
        }
        if is_context:
            tokenizer = b"{}"
            tokenizer_hash = hashlib.sha256(tokenizer).hexdigest()
            (root / "tokenizer.json").write_bytes(tokenizer)
            manifest["tokenizerSha256"] = tokenizer_hash
            report["tokenizer"] = {
                "file": "tokenizer.json",
                "bytes": len(tokenizer),
                "sha256": tokenizer_hash,
            }
        (root / "manifest-development.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "export-report-development.json").write_text(
            json.dumps(report, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
