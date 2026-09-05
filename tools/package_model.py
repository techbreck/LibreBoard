#!/usr/bin/env python3
"""Package an accepted LibreBoard ONNX export as a deterministic signed .lbmodel archive."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from typing import Any


CANONICAL_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MODEL_IDS = {
    "swipe-latin-v1": {
        "kind": "swipe-ctc",
        "model": "swipe-latin-v1.onnx",
        "developmentModel": "swipe-latin-v1-development.onnx",
        "tokenizer": False,
    },
    "context-en-de-v1": {
        "kind": "context-rescorer",
        "model": "context-en-de-v1.onnx",
        "developmentModel": "context-en-de-v1-development.onnx",
        "tokenizer": True,
    },
}
MANIFEST_KEYS = {
    "schemaVersion",
    "engineAbi",
    "modelKind",
    "tensorAbi",
    "locales",
    "architecture",
    "parameterCount",
    "quantization",
    "modelSha256",
    "tokenizerSha256",
    "requiredOnnxOperators",
    "license",
    "provenance",
    "minimumAppVersionCode",
}
SIGNATURE_ALGORITHM = "SHA256withRSA"
MINIMUM_RSA_BITS = 3072


class ModelPackagingError(RuntimeError):
    pass


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelPackagingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: pathlib.Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ModelPackagingError(f"{label} is missing or is not a regular file")
    payload = path.read_bytes()
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_key)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackagingError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelPackagingError(f"{label} must be a JSON object")
    return value, payload


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(root: pathlib.Path, descriptor: dict[str, Any], label: str) -> pathlib.Path:
    if set(descriptor) != {"file", "bytes", "sha256"}:
        raise ModelPackagingError(f"{label} descriptor has an unexpected schema")
    filename = descriptor["file"]
    if not isinstance(filename, str) or pathlib.PurePosixPath(filename).name != filename:
        raise ModelPackagingError(f"{label} filename is unsafe")
    path = root / filename
    if not path.is_file() or path.is_symlink():
        raise ModelPackagingError(f"{label} file is missing or is not regular")
    if descriptor["bytes"] != path.stat().st_size or descriptor["sha256"] != sha256_file(path):
        raise ModelPackagingError(f"{label} does not match its export report")
    return path


def _run_openssl(command: list[str], *, input_bytes: bytes | None = None) -> bytes:
    for attempt in range(5):
        try:
            result = subprocess.run(command, input=input_bytes, capture_output=True, check=False)
            break
        except OSError as exc:
            if exc.errno == errno.EAGAIN and attempt < 4:
                time.sleep(0.1)
                continue
            raise ModelPackagingError(f"could not start OpenSSL: {exc}") from exc
    else:
        raise AssertionError("bounded OpenSSL retry loop did not return")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ModelPackagingError(f"OpenSSL command failed: {detail or result.returncode}")
    return result.stdout


def validate_private_key(path: pathlib.Path) -> tuple[bytes, str]:
    if not path.is_file() or path.is_symlink():
        raise ModelPackagingError("model signing key is missing or is not a regular file")
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ModelPackagingError("model signing key must not be accessible by group or other users")
    description = _run_openssl([
        "openssl", "pkey", "-in", str(path), "-text", "-noout",
    ]).decode("utf-8", errors="replace")
    match = re.search(r"Private-Key:\s*\((\d+) bit", description)
    if match is None or "modulus:" not in description.lower():
        raise ModelPackagingError("model signing key must be an RSA private key")
    if int(match.group(1)) < MINIMUM_RSA_BITS:
        raise ModelPackagingError(f"model signing RSA key must be at least {MINIMUM_RSA_BITS} bits")
    public_der = _run_openssl([
        "openssl", "pkey", "-in", str(path), "-pubout", "-outform", "DER",
    ])
    version = _run_openssl(["openssl", "version"]).decode("utf-8").strip()
    return public_der, version


def sign_manifest(private_key: pathlib.Path, manifest: bytes) -> bytes:
    command = ["openssl", "dgst", "-sha256", "-sign", str(private_key)]
    first = _run_openssl(command, input_bytes=manifest)
    second = _run_openssl(command, input_bytes=manifest)
    if not first or len(first) > 16 * 1024 or first != second:
        raise ModelPackagingError("model signature is empty, oversized, or nondeterministic")
    return first


def _write_canonical_archive(path: pathlib.Path, entries: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    temporary.close()
    try:
        with zipfile.ZipFile(temporary_path, "w", allowZip64=False) as archive:
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, CANONICAL_TIMESTAMP)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, entries[name])
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def package_model(
    export_root: pathlib.Path,
    private_key: pathlib.Path,
    output_root: pathlib.Path,
    *,
    development: bool = False,
) -> dict[str, Any]:
    export_root = export_root.resolve()
    report_name = "export-report-development.json" if development else "export-report.json"
    report_path = export_root / report_name
    report, report_bytes = load_json(report_path, "model export report")
    expected_eligibility = not development
    if report.get("schemaVersion") != 1 or report.get("releaseEligible") is not expected_eligibility:
        raise ModelPackagingError("model export report eligibility does not match packaging mode")
    model_id = report.get("modelId")
    contract = MODEL_IDS.get(model_id)
    if contract is None:
        raise ModelPackagingError("unsupported model identity")

    model_descriptor = report.get("model")
    if not isinstance(model_descriptor, dict):
        raise ModelPackagingError("model export report has no model descriptor")
    model_path = _required_file(export_root, model_descriptor, "model")
    expected_model = contract["developmentModel" if development else "model"]
    if model_path.name != expected_model:
        raise ModelPackagingError("model filename does not match its identity and packaging mode")

    manifest_name = report.get("manifest")
    expected_manifest = "manifest-development.json" if development else "manifest.json"
    if manifest_name != expected_manifest:
        raise ModelPackagingError("manifest filename does not match packaging mode")
    manifest, manifest_bytes = load_json(export_root / manifest_name, "model manifest")
    if set(manifest) - {"tokenizerSha256"} != MANIFEST_KEYS - {"tokenizerSha256"}:
        raise ModelPackagingError("model manifest has an unexpected schema")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("engineAbi") != 1
        or manifest.get("modelKind") != contract["kind"]
        or manifest.get("tensorAbi") != model_id
        or manifest.get("modelSha256") != model_descriptor["sha256"]
    ):
        raise ModelPackagingError("model manifest does not match its export")

    entries = {
        "manifest.json": manifest_bytes,
        "model.onnx": model_path.read_bytes(),
    }
    tokenizer_descriptor = report.get("tokenizer")
    if contract["tokenizer"]:
        if not isinstance(tokenizer_descriptor, dict):
            raise ModelPackagingError("context model export has no tokenizer descriptor")
        tokenizer_path = _required_file(export_root, tokenizer_descriptor, "tokenizer")
        if tokenizer_path.name != "tokenizer.json" or manifest.get("tokenizerSha256") != tokenizer_descriptor["sha256"]:
            raise ModelPackagingError("context tokenizer does not match the model manifest")
        entries["tokenizer.json"] = tokenizer_path.read_bytes()
    elif tokenizer_descriptor is not None or manifest.get("tokenizerSha256") is not None:
        raise ModelPackagingError("swipe model must not contain a tokenizer")

    private_key = private_key.resolve()
    public_der, openssl_version = validate_private_key(private_key)
    entries["signature.der"] = sign_manifest(private_key, manifest_bytes)
    output_root = output_root.resolve()
    archive_path = output_root / f"{model_id}{'-development' if development else ''}.lbmodel"
    _write_canonical_archive(archive_path, entries)

    public_key_path = output_root / "libreboard-model-signing-public.der"
    if public_key_path.exists() and public_key_path.read_bytes() != public_der:
        raise ModelPackagingError("output directory contains a different model signing public key")
    public_key_path.write_bytes(public_der)
    package_report = {
        "schemaVersion": 1,
        "modelId": model_id,
        "releaseEligible": not development,
        "signatureAlgorithm": SIGNATURE_ALGORITHM,
        "publicKeySha256": sha256_bytes(public_der),
        "opensslVersion": openssl_version,
        "exportReportSha256": sha256_bytes(report_bytes),
        "manifestSha256": sha256_bytes(manifest_bytes),
        "modelSha256": model_descriptor["sha256"],
        "archive": {
            "file": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
    }
    package_report_path = output_root / f"{model_id}{'-development' if development else ''}.package.json"
    package_report_path.write_text(
        json.dumps(package_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=pathlib.Path, required=True)
    parser.add_argument("--private-key", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = package_model(
            args.export_root,
            args.private_key,
            args.output_root,
            development=args.development,
        )
    except (ModelPackagingError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"model packaging error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
