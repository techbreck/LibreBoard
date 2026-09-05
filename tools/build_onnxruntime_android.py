#!/usr/bin/env python3
"""Build and verify LibreBoard's pinned, reduced, CPU-only ONNX Runtime Android AAR."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "onnxruntime"
SETTINGS = ROOT / "runtime" / "onnxruntime" / "build-settings.json"
DEFAULT_BUILD_ROOT = ROOT / "build" / "onnxruntime"
EXPECTED_KEYS = {
    "schemaVersion",
    "sourceCommit",
    "ndkRevision",
    "configuration",
    "abis",
    "androidMinSdk",
    "androidTargetSdk",
    "maximumNativeBytesPerAbi",
    "buildParameters",
}
EXPECTED_ABIS = ["armeabi-v7a", "arm64-v8a", "x86", "x86_64"]
EXPECTED_NDK_REVISION = "28.0.13004108"
REQUIRED_PARAMETERS = {
    "--android",
    "--parallel",
    "--cmake_generator=Ninja",
    "--build_java",
    "--build_shared_lib",
    "--disable_ml_ops",
    "--enable_lto",
    "--skip_tests",
}
FORBIDDEN_PARAMETER_FRAGMENTS = (
    "nnapi",
    "xnnpack",
    "webgpu",
    "extensions",
    "training",
    "vcpkg_ms_internal",
)
EXPECTED_LIBRARIES = {"libonnxruntime.so", "libonnxruntime4j_jni.so"}
EXPECTED_PYTHON_PACKAGES = {
    "flatbuffers": "25.12.19",
    "numpy": "2.2.6",
    "packaging": "26.3",
    "protobuf": "7.36.1",
}
OPS_LINE = re.compile(r"^[A-Za-z0-9_.-]+;[1-9][0-9]*;[A-Za-z0-9_.,-]+$")
CANONICAL_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class BuildConfigurationError(RuntimeError):
    pass


def validate_python_toolchain(
    *,
    python_version: tuple[int, int] | None = None,
    package_versions: dict[str, str] | None = None,
) -> None:
    version = python_version or sys.version_info[:2]
    if version != (3, 11):
        raise BuildConfigurationError(f"Python 3.11 is required, found {version[0]}.{version[1]}")
    if package_versions is None:
        try:
            package_versions = {
                package: importlib.metadata.version(package)
                for package in EXPECTED_PYTHON_PACKAGES
            }
        except importlib.metadata.PackageNotFoundError as failure:
            raise BuildConfigurationError(
                "install the hash-locked ONNX Runtime Python build requirements"
            ) from failure
    if package_versions != EXPECTED_PYTHON_PACKAGES:
        raise BuildConfigurationError(
            f"ONNX Runtime Python build requirements do not match the lock: {package_versions}"
        )


def run(command: list[str], *, cwd: pathlib.Path, env: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise BuildConfigurationError(
            f"cannot start command {' '.join(command)}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise BuildConfigurationError(f"command failed: {' '.join(command)}\n{detail}")
    return result.stdout.strip()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_settings(path: pathlib.Path = SETTINGS) -> dict:
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildConfigurationError(f"cannot read runtime build settings: {exc}") from exc
    if not isinstance(settings, dict) or set(settings) != EXPECTED_KEYS:
        raise BuildConfigurationError("runtime build settings have an unexpected schema")
    if settings["schemaVersion"] != 1:
        raise BuildConfigurationError("unsupported runtime build-settings schema")
    if not re.fullmatch(r"[0-9a-f]{40}", settings["sourceCommit"]):
        raise BuildConfigurationError("invalid ONNX Runtime source commit")
    if settings["ndkRevision"] != EXPECTED_NDK_REVISION:
        raise BuildConfigurationError("runtime must use the exact pinned NDK revision")
    if settings["configuration"] != "Release":
        raise BuildConfigurationError("runtime must use the Release configuration")
    if settings["abis"] != EXPECTED_ABIS:
        raise BuildConfigurationError("runtime ABI list must exactly match the application")
    if settings["androidMinSdk"] != 26 or settings["androidTargetSdk"] != 36:
        raise BuildConfigurationError("runtime Android SDK levels must match LibreBoard")
    if not isinstance(settings["maximumNativeBytesPerAbi"], int) or not (
        1 <= settings["maximumNativeBytesPerAbi"] <= 8 * 1024 * 1024
    ):
        raise BuildConfigurationError("invalid native runtime size ceiling")
    parameters = settings["buildParameters"]
    if not isinstance(parameters, list) or any(not isinstance(value, str) for value in parameters):
        raise BuildConfigurationError("runtime build parameters must be strings")
    if set(parameters) != REQUIRED_PARAMETERS or len(parameters) != len(REQUIRED_PARAMETERS):
        raise BuildConfigurationError("runtime build parameters do not match the audited CPU-only set")
    lowered = " ".join(parameters).lower()
    if any(fragment in lowered for fragment in FORBIDDEN_PARAMETER_FRAGMENTS):
        raise BuildConfigurationError("runtime build enables a forbidden provider or component")
    return settings


def validate_ops_config(path: pathlib.Path) -> None:
    if not path.is_file():
        raise BuildConfigurationError(f"reduced-operator configuration is missing: {path}")
    meaningful = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not OPS_LINE.fullmatch(line):
            raise BuildConfigurationError(f"invalid reduced-operator configuration line: {line}")
        meaningful.append(line)
    if not meaningful:
        raise BuildConfigurationError("reduced-operator configuration contains no operators")


def validate_source(settings: dict) -> None:
    if not (SOURCE / ".git").exists() or not (SOURCE / "tools" / "ci_build" / "build.py").is_file():
        raise BuildConfigurationError("initialize the pinned ONNX Runtime submodule")
    commit = run(["git", "rev-parse", "HEAD"], cwd=SOURCE)
    if commit != settings["sourceCommit"]:
        raise BuildConfigurationError(f"unexpected ONNX Runtime commit: {commit}")
    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SOURCE)
    if dirty:
        raise BuildConfigurationError("ONNX Runtime source tree has tracked modifications")
    submodules = run(["git", "submodule", "status", "--recursive"], cwd=SOURCE)
    unready = [line for line in submodules.splitlines() if line[:1] in {"-", "+", "U"}]
    if unready:
        raise BuildConfigurationError("initialize every pinned ONNX Runtime nested submodule")


def validate_ndk(ndk: pathlib.Path) -> pathlib.Path:
    source_properties = ndk / "source.properties"
    if not source_properties.is_file():
        raise BuildConfigurationError(f"Android NDK is missing: {ndk}")
    revision = next(
        (line.split("=", 1)[1].strip() for line in source_properties.read_text().splitlines()
         if line.startswith("Pkg.Revision")),
        "",
    )
    if revision != EXPECTED_NDK_REVISION:
        raise BuildConfigurationError(
            f"NDK {EXPECTED_NDK_REVISION} is required, found {revision or 'unknown'}"
        )
    candidates = list((ndk / "toolchains" / "llvm" / "prebuilt").glob("*/bin/llvm-readobj"))
    if len(candidates) != 1:
        raise BuildConfigurationError("could not resolve the NDK llvm-readobj")
    return candidates[0]


def build_environment(build_root: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    epoch = run(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=SOURCE)
    prefix_flags = f"-ffile-prefix-map={SOURCE}=. -ffile-prefix-map={build_root}=./build"
    env.update({
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "SOURCE_DATE_EPOCH": epoch,
        "CFLAGS": f"{env.get('CFLAGS', '')} {prefix_flags}".strip(),
        "CXXFLAGS": f"{env.get('CXXFLAGS', '')} {prefix_flags}".strip(),
        "LDFLAGS": f"{env.get('LDFLAGS', '')} -Wl,--build-id=sha1".strip(),
    })
    return env


def resolved_build_parameters(settings: dict, jobs: int) -> list[str]:
    if not 1 <= jobs <= 64:
        raise BuildConfigurationError("runtime build jobs must be between 1 and 64")
    return [
        f"--parallel={jobs}" if value == "--parallel" else value
        for value in settings["buildParameters"]
    ]


def _canonical_zip_payload(payload: bytes, *, depth: int = 0) -> bytes:
    source = io.BytesIO(payload)
    output = io.BytesIO()
    with zipfile.ZipFile(source) as input_archive:
        entries = input_archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise BuildConfigurationError("runtime archive contains duplicate entries")
        if any(
            not name
            or name.startswith(("/", "\\"))
            or ".." in pathlib.PurePosixPath(name).parts
            for name in names
        ):
            raise BuildConfigurationError("runtime archive contains an unsafe entry name")
        with zipfile.ZipFile(output, "w", allowZip64=True) as output_archive:
            for original in sorted(entries, key=lambda entry: entry.filename):
                entry_payload = input_archive.read(original)
                if (
                    depth == 0
                    and pathlib.PurePosixPath(original.filename).suffix.lower() in {".jar", ".zip"}
                    and zipfile.is_zipfile(io.BytesIO(entry_payload))
                ):
                    entry_payload = _canonical_zip_payload(entry_payload, depth=1)
                canonical = zipfile.ZipInfo(original.filename, CANONICAL_ZIP_TIMESTAMP)
                canonical.compress_type = original.compress_type
                canonical.create_system = 3
                canonical.external_attr = (
                    (0o40755 if original.is_dir() else 0o100644) << 16
                ) | (0x10 if original.is_dir() else 0)
                output_archive.writestr(
                    canonical,
                    entry_payload,
                    compress_type=original.compress_type,
                    compresslevel=9 if original.compress_type == zipfile.ZIP_DEFLATED else None,
                )
    return output.getvalue()


def canonicalize_aar(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Rewrite an AAR and nested JARs with deterministic ZIP metadata and compression."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    )
    temporary_path = pathlib.Path(temporary.name)
    try:
        temporary.write(_canonical_zip_payload(source.read_bytes()))
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary_path, destination)
    except BuildConfigurationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise BuildConfigurationError(f"cannot canonicalize runtime AAR: {exc}") from exc
    finally:
        if not temporary.closed:
            temporary.close()
        temporary_path.unlink(missing_ok=True)


def build_aar(
    settings: dict,
    sdk: pathlib.Path,
    ndk: pathlib.Path,
    ops_config: pathlib.Path,
    build_root: pathlib.Path,
    jobs: int,
) -> pathlib.Path:
    intermediates = build_root / "intermediates"
    jni_root = intermediates / "jnilibs" / settings["configuration"]
    env = build_environment(build_root)
    env["ANDROID_HOME"] = str(sdk)
    env["ANDROID_SDK_ROOT"] = str(sdk)
    env["ANDROID_NDK_HOME"] = str(ndk)
    headers = None

    for abi in settings["abis"]:
        abi_build = intermediates / abi
        parameters = resolved_build_parameters(settings, jobs)
        command = [
            sys.executable,
            str(SOURCE / "tools" / "ci_build" / "build.py"),
            *parameters,
            f"--config={settings['configuration']}",
            f"--android_abi={abi}",
            f"--android_api={settings['androidMinSdk']}",
            f"--android_sdk_path={sdk}",
            f"--android_ndk_path={ndk}",
            f"--include_ops_by_config={ops_config}",
            f"--build_dir={abi_build}",
        ]
        run(command, cwd=SOURCE, env=env)
        native_dir = abi_build / settings["configuration"]
        destination = jni_root / abi
        destination.mkdir(parents=True, exist_ok=True)
        for library in EXPECTED_LIBRARIES:
            source_library = native_dir / library
            if not source_library.is_file():
                raise BuildConfigurationError(f"runtime build did not produce {abi}/{library}")
            shutil.copyfile(source_library, destination / library)
        headers = headers or native_dir / "android" / "headers"

    if headers is None or not headers.is_dir():
        raise BuildConfigurationError("runtime build did not produce Android headers")
    aar_build = intermediates / "aar" / settings["configuration"]
    gradle = SOURCE / "java" / ("gradlew.bat" if os.name == "nt" else "gradlew")
    gradle_command = [
        str(gradle),
        "--no-daemon",
        "-b=build-android.gradle",
        "-c=settings-android.gradle",
        f"-DjniLibsDir={jni_root}",
        f"-DbuildDir={aar_build}",
        f"-DheadersDir={headers}",
        f"-DpublishDir={intermediates / 'unused-publish'}",
        f"-DminSdkVer={settings['androidMinSdk']}",
        f"-DtargetSdkVer={settings['androidTargetSdk']}",
        "-DENABLE_TRAINING_APIS=0",
        "clean",
        "assembleRelease",
    ]
    run(gradle_command, cwd=SOURCE / "java", env=env)
    generated = aar_build / "outputs" / "aar" / "onnxruntime-release.aar"
    if not generated.is_file():
        raise BuildConfigurationError("Android packaging did not produce the runtime AAR")
    output = build_root / "output"
    output.mkdir(parents=True, exist_ok=True)
    final_aar = output / "onnxruntime-mobile-1.26.0.aar"
    canonicalize_aar(generated, final_aar)
    return final_aar


def verify_aar(
    aar: pathlib.Path,
    settings: dict,
    readobj: pathlib.Path,
    ops_config: pathlib.Path,
    build_root: pathlib.Path,
) -> pathlib.Path:
    native_hashes: dict[str, dict[str, str | int]] = {}
    aar_payload = aar.read_bytes()
    if _canonical_zip_payload(aar_payload) != aar_payload:
        raise BuildConfigurationError("runtime AAR or a nested archive is not canonically packaged")
    with zipfile.ZipFile(aar) as archive, tempfile.TemporaryDirectory(prefix="libreboard-ort-") as temp:
        if (
            archive.comment
            or archive.namelist() != sorted(archive.namelist())
            or any(
                entry.date_time != CANONICAL_ZIP_TIMESTAMP
                or entry.extra
                or entry.comment
                for entry in archive.infolist()
            )
        ):
            raise BuildConfigurationError("runtime AAR is not canonically packaged")
        native_entries = [name for name in archive.namelist() if name.endswith(".so")]
        expected_entries = {
            f"jni/{abi}/{library}" for abi in settings["abis"] for library in EXPECTED_LIBRARIES
        }
        if set(native_entries) != expected_entries:
            raise BuildConfigurationError("runtime AAR contains an unexpected native-library set")
        for abi in settings["abis"]:
            total = sum(archive.getinfo(f"jni/{abi}/{library}").file_size for library in EXPECTED_LIBRARIES)
            if total > settings["maximumNativeBytesPerAbi"]:
                raise BuildConfigurationError(f"{abi} runtime exceeds the native size ceiling")
            for library in sorted(EXPECTED_LIBRARIES):
                entry = f"jni/{abi}/{library}"
                payload = archive.read(entry)
                extracted = pathlib.Path(temp) / abi / library
                extracted.parent.mkdir(parents=True, exist_ok=True)
                extracted.write_bytes(payload)
                headers = run([str(readobj), "--program-headers", str(extracted)], cwd=ROOT)
                alignments = [int(value) for value in re.findall(
                    r"Type:\s+PT_LOAD(?:(?!Type:).)*?Alignment:\s+(\d+)", headers, re.DOTALL
                )]
                if not alignments or any(value < 16_384 for value in alignments):
                    raise BuildConfigurationError(f"{entry} lacks 16 KiB ELF alignment")
                native_hashes[entry] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }

    manifest = {
        "schemaVersion": 2,
        "sourceCommit": settings["sourceCommit"],
        "ndkRevision": settings["ndkRevision"],
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}",
        "pythonPackages": dict(sorted(EXPECTED_PYTHON_PACKAGES.items())),
        "sourceDateEpoch": int(run(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=SOURCE)),
        "buildToolSha256": sha256(pathlib.Path(__file__)),
        "settingsSha256": sha256(SETTINGS),
        "operatorsSha256": sha256(ops_config),
        "aarSha256": sha256(aar),
        "nativeLibraries": native_hashes,
    }
    manifest_path = build_root / "output" / "onnxruntime-mobile-1.26.0.build.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops-config", required=True, type=pathlib.Path)
    parser.add_argument("--sdk", type=pathlib.Path, default=os.environ.get("ANDROID_HOME"))
    parser.add_argument("--ndk", type=pathlib.Path, default=os.environ.get("ANDROID_NDK_HOME"))
    parser.add_argument("--build-root", type=pathlib.Path, default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        parser.error("jobs must be between 1 and 64")
    return args


def main() -> int:
    args = parse_args()
    try:
        validate_python_toolchain()
        settings = load_settings()
        ops_config = args.ops_config.resolve()
        validate_ops_config(ops_config)
        validate_source(settings)
        if args.sdk is None or not pathlib.Path(args.sdk).is_dir():
            raise BuildConfigurationError("provide the Android SDK with --sdk or ANDROID_HOME")
        if args.ndk is None:
            raise BuildConfigurationError("provide Android NDK r28 with --ndk or ANDROID_NDK_HOME")
        sdk = pathlib.Path(args.sdk).resolve()
        ndk = pathlib.Path(args.ndk).resolve()
        readobj = validate_ndk(ndk)
        if args.check_only:
            print("LibreBoard ONNX Runtime source and build inputs are valid")
            return 0
        build_root = args.build_root.resolve()
        build_root.mkdir(parents=True, exist_ok=True)
        aar = build_aar(settings, sdk, ndk, ops_config, build_root, args.jobs)
        manifest = verify_aar(aar, settings, readobj, ops_config, build_root)
        print(f"Built {aar}")
        print(f"Verified {manifest}")
        return 0
    except (BuildConfigurationError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ONNX Runtime build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
