#!/usr/bin/env python3
"""Drive the Phase 0 measurement instrumentation on a bound device or emulator.

Pushes the prepared tap/swipe corpora and (optionally) the context and swipe models into the
debug app's private files directory, runs Phase0MeasurementInstrumentedTest, and pulls the
schema-4 measurement JSONL plus an environment sidecar merged into the --metadata file. The
output is raw measurement input for evaluate_engine.py, not release evidence alone.

Nothing is written until the run is proven complete: the driver verifies that the installed APK
and the injected model bytes hash to the values bound into metadata, that every gated test
actually ran and passed, and that every pulled row carries this run's id and environment. A run
that skipped its tests, crashed, or left a stale output file on the device is rejected instead of
being saved as evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evaluate_engine  # noqa: E402

PACKAGE = "org.libreboard.keyboard.debug"
RUNNER = "org.libreboard.keyboard.debug.test/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "helium314.keyboard.latin.engine.Phase0MeasurementInstrumentedTest"
REMOTE_DIR = "phase0-measurement"
STAGING_DIR = "/data/local/tmp/phase0-staging"
PUBLISH_DIR = f"/sdcard/Android/data/{PACKAGE}/files/{REMOTE_DIR}"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
ENVIRONMENTS = {"stock_android_hardware", "grapheneos_hardware", "low_ram_emulator"}
SHA256_LINE = re.compile(r"^([0-9a-f]{64})\s")
STATUS_CODE = re.compile(r"^INSTRUMENTATION_STATUS_CODE: (-?\d+)\s*$", re.MULTILINE)
RESULT_CODE = re.compile(r"^INSTRUMENTATION_CODE: (-?\d+)\s*$", re.MULTILINE)
# AndroidJUnitRunner per-test status codes.
STATUS_OK = 0
STATUS_START = 1
STATUS_ERROR = -1
STATUS_FAILURE = -2
STATUS_IGNORED = -3
STATUS_ASSUMPTION_FAILURE = -4
# Activity.RESULT_OK, the only run-level code that means the runner finished normally.
RESULT_OK = -1


class MeasurementRunError(RuntimeError):
    """The run cannot be trusted as measurement input."""


def adb_bytes(adb_path: str, serial: str | None, *args: str, stdin: bytes | None = None,
              timeout: int = 600) -> bytes:
    command = [adb_path]
    if serial:
        command += ["-s", serial]
    command += list(args)
    result = subprocess.run(command, input=stdin, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise MeasurementRunError(
            f"adb {' '.join(args)} failed: {result.stderr.decode(errors='replace')[:512]}")
    return result.stdout


def adb(adb_path: str, serial: str | None, *args: str, stdin: bytes | None = None,
        timeout: int = 600) -> str:
    return adb_bytes(adb_path, serial, *args, stdin=stdin, timeout=timeout).decode(errors="replace")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_instrumentation_result(output: str, expected_tests: int) -> dict[str, int]:
    """Reject anything but a clean, complete run.

    `am instrument -w` exits zero whether the tests passed, failed, were skipped by an
    assumption, or never ran at all, so the exit status alone cannot tell a measurement run from a
    no-op that leaves a previous run's output file in place.
    """
    codes = [int(value) for value in STATUS_CODE.findall(output)]
    results = [int(value) for value in RESULT_CODE.findall(output)]
    counts = {
        "passed": codes.count(STATUS_OK),
        "failed": codes.count(STATUS_ERROR) + codes.count(STATUS_FAILURE),
        "skipped": codes.count(STATUS_IGNORED) + codes.count(STATUS_ASSUMPTION_FAILURE),
        "started": codes.count(STATUS_START),
    }
    if counts["failed"]:
        raise MeasurementRunError(
            f"{counts['failed']} instrumented test(s) failed: {output[-1024:]}")
    # The class always reports both replay tests; the one whose corpus was not requested skips
    # itself by assumption, which is correct. What must not happen is a *requested* test skipping,
    # so the passing count is the authority rather than the skip count.
    if counts["passed"] != expected_tests:
        detail = (f"; {counts['skipped']} test(s) were skipped, so the measurement gate arguments "
                  f"most likely did not reach the runner") if counts["skipped"] else ""
        if not results:
            detail += "; the run reported no result code, so it did not finish"
        elif results[-1] != RESULT_OK:
            detail += f"; the runner exited with code {results[-1]}: {output[-512:]}"
        raise MeasurementRunError(
            f"expected {expected_tests} instrumented test(s) to pass, observed "
            f"{counts['passed']}{detail}")
    # Every requested test reported OK. A non-OK run-level code after that point is the system
    # reaping the app process ("due to finished inst"), which `am` reports as a crash even though
    # the measurement completed and its rows were written. The pulled rows are verified separately,
    # so trust the per-test stream rather than discarding a finished run.
    counts["runnerResultCode"] = results[-1] if results else None
    return counts


def verify_measurement_rows(payload: bytes, run_id: str, environment: str) -> dict[str, int]:
    """Prove the pulled bytes belong to this run before they are saved.

    `run-as cat` happily returns a file an earlier run left behind, so binding has to be checked
    against the rows themselves rather than against the fact that a file existed.
    """
    counts = {"tap": 0, "swipe": 0}
    for number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as failure:
            raise MeasurementRunError(f"pulled line {number}: invalid JSON: {failure.msg}") from failure
        if not isinstance(row, dict):
            raise MeasurementRunError(f"pulled line {number}: row must be an object")
        if row.get("schemaVersion") != evaluate_engine.SCHEMA_VERSION:
            raise MeasurementRunError(
                f"pulled line {number}: expected schemaVersion "
                f"{evaluate_engine.SCHEMA_VERSION}, found {row.get('schemaVersion')!r}")
        if row.get("testRunId") != run_id:
            raise MeasurementRunError(
                f"pulled line {number}: testRunId {row.get('testRunId')!r} is not {run_id!r}; "
                f"the device returned another run's output")
        if row.get("environmentKind") != environment:
            raise MeasurementRunError(
                f"pulled line {number}: environmentKind {row.get('environmentKind')!r} "
                f"is not {environment!r}")
        counts["swipe" if row.get("category") == "swipe" else "tap"] += 1
    return counts


def remote_sha256(adb_path: str, serial: str | None, run_as: list[str], remote: str) -> str:
    output = adb(adb_path, serial, "exec-out", *run_as, "sha256sum", remote)
    match = SHA256_LINE.match(output.strip())
    if not match:
        raise MeasurementRunError(f"could not hash {remote} on the device: {output[:256]}")
    return match.group(1)


def verify_installed_apk(adb_path: str, serial: str | None, expected: str,
                         device_user: int | None) -> str:
    """Hash the APK the device will actually run, not the one the caller believes it installed."""
    command = ["shell", "pm", "path"]
    if device_user is not None:
        command += ["--user", str(device_user)]
    paths = [line.split("package:", 1)[1].strip()
             for line in adb(adb_path, serial, *command, PACKAGE).splitlines()
             if line.startswith("package:")]
    if len(paths) != 1:
        raise MeasurementRunError(
            f"expected exactly one installed APK path for {PACKAGE}, found {paths}")
    output = adb(adb_path, serial, "exec-out", "sha256sum", paths[0])
    match = SHA256_LINE.match(output.strip())
    if not match:
        raise MeasurementRunError(f"could not hash the installed APK: {output[:256]}")
    installed = match.group(1)
    if installed != expected:
        raise MeasurementRunError(
            f"installed APK is {installed}, but metadata binds {expected}; "
            f"the device is not running the build being measured")
    return installed


def push_private(adb_path: str, serial: str | None, local: pathlib.Path, remote_name: str,
                 device_user: int | None = None, expected_sha256: str | None = None) -> str:
    if not SAFE_NAME.fullmatch(remote_name):
        raise MeasurementRunError(f"unsafe remote name {remote_name}")
    local_sha256 = sha256_file(local)
    if expected_sha256 is not None and local_sha256 != expected_sha256:
        raise MeasurementRunError(
            f"{local} hashes to {local_sha256}, but metadata binds {expected_sha256}")
    staging = f"/data/local/tmp/{remote_name}.phase0push"
    adb(adb_path, serial, "push", str(local), staging)
    run_as = ["run-as", PACKAGE]
    if device_user is not None:
        run_as += ["--user", str(device_user)]
    adb(adb_path, serial, "shell", *run_as, "sh", "-c",
        f"'mkdir -p files/{REMOTE_DIR} && cp {staging} files/{REMOTE_DIR}/{remote_name}'")
    adb(adb_path, serial, "shell", "rm", "-f", staging)
    injected = remote_sha256(adb_path, serial, run_as, f"files/{REMOTE_DIR}/{remote_name}")
    if injected != local_sha256:
        raise MeasurementRunError(
            f"{remote_name} landed on the device as {injected}, not {local_sha256}")
    return local_sha256


def package_allows_run_as(adb_path: str, serial: str | None,
                          device_user: int | None = None) -> bool:
    """Debuggable packages expose `run-as`; GrapheneOS testOnly measurement builds do not."""
    run_as = ["run-as", PACKAGE]
    if device_user is not None:
        run_as += ["--user", str(device_user)]
    try:
        output = adb(adb_path, serial, "shell", *run_as, "id")
    except MeasurementRunError:
        return False
    return "uid=" in output


def ensure_staging_dir(adb_path: str, serial: str | None) -> None:
    """Readable corpora live here; GrapheneOS apps cannot write to /data/local/tmp."""
    adb(adb_path, serial, "shell", "mkdir", "-p", STAGING_DIR)
    adb(adb_path, serial, "shell", "chmod", "777", STAGING_DIR)


def ensure_publish_dir(adb_path: str, serial: str | None) -> None:
    """App-owned external files dir: the test writes here, adb pulls without run-as."""
    adb(adb_path, serial, "shell", "mkdir", "-p", PUBLISH_DIR)
    adb(adb_path, serial, "shell", "chmod", "775", PUBLISH_DIR)


def push_staging(adb_path: str, serial: str | None, local: pathlib.Path, remote_name: str,
                 expected_sha256: str | None = None) -> str:
    """Stage a file under /data/local/tmp for a non-debuggable testOnly APK."""
    if not SAFE_NAME.fullmatch(remote_name):
        raise MeasurementRunError(f"unsafe remote name {remote_name}")
    local_sha256 = sha256_file(local)
    if expected_sha256 is not None and local_sha256 != expected_sha256:
        raise MeasurementRunError(
            f"{local} hashes to {local_sha256}, but metadata binds {expected_sha256}")
    ensure_staging_dir(adb_path, serial)
    dest = f"{STAGING_DIR}/{remote_name}"
    adb(adb_path, serial, "push", str(local), dest)
    adb(adb_path, serial, "shell", "chmod", "644", dest)
    output = adb(adb_path, serial, "exec-out", "sha256sum", dest)
    match = SHA256_LINE.match(output.strip())
    if not match:
        raise MeasurementRunError(f"could not hash {dest} on the device: {output[:256]}")
    if match.group(1) != local_sha256:
        raise MeasurementRunError(
            f"{remote_name} landed on the device as {match.group(1)}, not {local_sha256}")
    return local_sha256


def compile_package(adb_path: str, serial: str | None, compile_filter: str,
                    device_user: int | None = None) -> None:
    """Force AOT so GrapheneOS representative-latency runs are not interpreted."""
    if not SAFE_NAME.fullmatch(compile_filter):
        raise MeasurementRunError(f"unsafe compile filter {compile_filter}")
    command = ["shell", "cmd", "package", "compile", "-m", compile_filter, "-f"]
    if device_user is not None:
        command += ["--user", str(device_user)]
    command.append(PACKAGE)
    output = adb(adb_path, serial, *command, timeout=300)
    if "Failure" in output or "Error" in output:
        raise MeasurementRunError(
            f"cmd package compile -m {compile_filter} failed: {output[:512]}")


PIN_KEYS = {
    "schemaVersion", "candidateId", "scope", "contextModelSha256", "contextTokenizerSha256",
    "swipeModelSha256", "onnxRuntimeAarSha256", "rejectedContextModelSha256", "notes",
}
SHA256_VALUE = re.compile(r"^[0-9a-f]{64}$")


def load_artifact_pin(path: pathlib.Path) -> dict[str, object]:
    """Read the artifact pin that says which model candidate a qualifying run may measure.

    The 2026-09-11 GrapheneOS run measured the superseded independent-split context export and
    nothing rejected it, because the driver took every artifact hash on trust from its own
    arguments. The pin turns that into a named, up-front failure.
    """
    try:
        pin = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as failure:
        raise MeasurementRunError(f"cannot read artifact pin {path}: {failure}") from failure
    if not isinstance(pin, dict) or pin.get("schemaVersion") != 1 or not set(pin) <= PIN_KEYS:
        raise MeasurementRunError(f"{path} is not a schema-1 artifact pin")
    for field in ("contextModelSha256", "contextTokenizerSha256", "swipeModelSha256"):
        value = pin.get(field)
        if not isinstance(value, str) or not SHA256_VALUE.fullmatch(value):
            raise MeasurementRunError(f"{path} requires a lowercase {field}")
    rejected = pin.get("rejectedContextModelSha256", {})
    if not isinstance(rejected, dict) or any(
        not isinstance(key, str) or not SHA256_VALUE.fullmatch(key) or not isinstance(value, str)
        for key, value in rejected.items()
    ):
        raise MeasurementRunError(f"{path} has a malformed rejectedContextModelSha256 map")
    if pin["contextModelSha256"] in rejected:
        raise MeasurementRunError(f"{path} pins a context model it also rejects")
    return pin


def pinned_hash(pin: dict[str, object] | None, field: str, override: str | None,
                label: str) -> str | None:
    """The pin wins; an explicit argument may only agree with it."""
    if pin is None:
        return override
    expected = pin[field]
    if override is not None and override != expected:
        raise MeasurementRunError(
            f"--{label} is {override}, but the artifact pin requires {expected}")
    return expected


REMOTE_SCRIPT = "/data/local/tmp/phase0-instrument.sh"
REMOTE_LOG = "/data/local/tmp/phase0-instrument.log"
REMOTE_DONE = "/data/local/tmp/phase0-instrument.done"


def run_detached_instrumentation(adb_path: str, serial: str | None, command: list[str],
                                 timeout_seconds: int, poll_seconds: int = 15) -> str:
    """Run the instrumentation so that losing the cable does not lose the run.

    A blocking `adb shell am instrument -w` ties a multi-hour hardware run to one USB connection:
    when the link drops the shell dies and takes the instrumentation with it, discarding hours of
    measurement. Launching it detached on the device and polling for a completion marker lets the
    run continue across a disconnect, and lets the driver pick the results back up afterwards.
    """
    script = "#!/system/bin/sh\n%s > %s 2>&1\necho $? > %s\n" % (
        " ".join(shlex.quote(part) for part in command), REMOTE_LOG, REMOTE_DONE)
    # A detached run outlives the host process that started it, so a previous run may still be
    # executing on the device; its wrapper shell has to go or two instrumentations collide. Killing
    # the app package itself is left to `am instrument`, which force-stops it as part of starting —
    # doing it here as well races with that startup and the run dies before any test reports.
    # The bracket keeps the pattern from matching this very command line, which would make pkill
    # kill its own shell; the trailing exit keeps "nothing matched" from failing the run.
    adb(adb_path, serial, "shell", "sh", "-c",
        "'pkill -f \"phase0-instrumen[t].sh\" 2>/dev/null; exit 0'")
    adb(adb_path, serial, "shell", "rm", "-f", REMOTE_SCRIPT, REMOTE_LOG, REMOTE_DONE)
    adb(adb_path, serial, "shell", "sh", "-c", f"'cat > {REMOTE_SCRIPT}'",
        stdin=script.encode())
    adb(adb_path, serial, "shell", "sh", "-c",
        f"'nohup sh {REMOTE_SCRIPT} >/dev/null 2>&1 &'")

    deadline = time.monotonic() + timeout_seconds
    disconnected = False
    while True:
        if time.monotonic() > deadline:
            raise MeasurementRunError(
                f"instrumentation did not finish within {timeout_seconds}s; it may still be "
                f"running on the device, and {REMOTE_LOG} holds its output so far")
        try:
            finished = adb(adb_path, serial, "shell", "sh", "-c",
                           f"'test -f {REMOTE_DONE} && echo yes || echo no'", timeout=120).strip()
        except MeasurementRunError:
            # The device went away mid-run. The instrumentation is detached, so it survives; wait
            # for the cable to come back rather than failing a run that is still progressing.
            if not disconnected:
                print("device unreachable; the detached run continues, waiting for it to return",
                      file=sys.stderr)
                disconnected = True
            try:
                adb(adb_path, serial, "wait-for-device", timeout=min(600, poll_seconds * 20))
            except MeasurementRunError:
                pass
            continue
        if disconnected:
            print("device is back; resuming polling", file=sys.stderr)
            disconnected = False
        if finished.endswith("yes"):
            break
        time.sleep(poll_seconds)

    status = adb(adb_path, serial, "shell", "cat", REMOTE_DONE).strip()
    output = adb(adb_path, serial, "exec-out", "cat", REMOTE_LOG)
    adb(adb_path, serial, "shell", "rm", "-f", REMOTE_SCRIPT, REMOTE_DONE)
    if status not in {"0", ""}:
        raise MeasurementRunError(
            f"instrumentation exited with status {status}: {output[-1024:]}")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=pathlib.Path, default=pathlib.Path(shutil.which("adb") or "adb"))
    parser.add_argument("--serial")
    parser.add_argument("--corpus", type=pathlib.Path,
                        help="prepared tap corpus JSONL (e.g. build/evaluation-data/combined-tap-v2/test.jsonl)")
    parser.add_argument("--swipe-corpus", type=pathlib.Path,
                        help="prepared swipe corpus JSONL (e.g. build/model-data/swipe-latin-v1/test.jsonl)")
    parser.add_argument("--model-dir", type=pathlib.Path,
                        help="directory containing the context model *.onnx and tokenizer.json")
    parser.add_argument("--swipe-model", type=pathlib.Path,
                        help="the CTC swipe model *.onnx file")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment", required=True, choices=sorted(ENVIRONMENTS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="split the corpus into this many disjoint row partitions")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="measure only rows whose ordinal %% shard-count equals this index")
    parser.add_argument("--device-user", type=int,
                        help="install/run under this Android user id (e.g. a Play-free GrapheneOS "
                             "profile); adds --user to instrument and run-as calls")
    parser.add_argument("--instrument-timeout", type=int, default=4 * 60 * 60,
                        help="seconds to wait for the instrumented run (default 4h)")
    parser.add_argument("--external-staging", action="store_true",
                        help="stage corpora under /data/local/tmp and skip run-as (required for "
                             "non-debuggable testOnly APKs; auto-selected when run-as is refused)")
    parser.add_argument("--compile-filter",
                        help="run `cmd package compile -m FILTER -f` before measuring (use "
                             "`speed` for GrapheneOS representative-latency runs)")
    parser.add_argument("--prediction", action="store_true",
                        help="also replay the tap corpus's context->target pairs through the "
                             "next-word prediction path (empty composer, InputStyle.PREDICTION)")
    parser.add_argument("--rescorer-budget-ms", type=int,
                        help="diagnostic-only: override the production rescorer budget so neural "
                             "scores can be observed on runtimes too slow to meet it")
    parser.add_argument("--swipe-warm-arms", action="store_true",
                        help="diagnostic-only: interleave swipe rows across three arms by row "
                             "index (control, standalone CTC without concurrent geometric, fused "
                             "under a busy spinner) to test cluster-warmth effects on latency")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path,
                        help="schema-%d metadata JSON to create or update with this run's environment"
                             % evaluate_engine.SCHEMA_VERSION)
    parser.add_argument("--artifact-pin", type=pathlib.Path,
                        help="schema-1 artifact pin (docs/phase-0-artifact-pin.json) naming the "
                             "model candidate a qualifying run must measure; required for release "
                             "evidence, omit only for explicitly diagnostic runs")
    parser.add_argument("--app-commit", help="full lowercase git commit of the app build")
    parser.add_argument("--core-apk-sha256", help="SHA-256 of the installed app APK")
    parser.add_argument("--swipe-model-sha256")
    parser.add_argument("--context-model-sha256")
    args = parser.parse_args(argv)

    if not 0 < len(args.run_id) <= 512:
        return 2
    if args.corpus is None and args.swipe_corpus is None:
        raise MeasurementRunError("at least one of --corpus or --swipe-corpus is required")
    pin = load_artifact_pin(args.artifact_pin) if args.artifact_pin is not None else None
    context_model_sha256 = pinned_hash(pin, "contextModelSha256", args.context_model_sha256,
                                       "context-model-sha256")
    swipe_model_sha256 = pinned_hash(pin, "swipeModelSha256", args.swipe_model_sha256,
                                     "swipe-model-sha256")
    adb_path = str(args.adb)
    remote_outputs: list[str] = []
    # A dropped cable between install and instrument is the usual way a long hardware run turns
    # into a partially-measured file; block until the device is actually back.
    adb(adb_path, args.serial, "wait-for-device", timeout=600)
    if args.core_apk_sha256 is not None:
        verify_installed_apk(adb_path, args.serial, args.core_apk_sha256, args.device_user)
    use_run_as = (not args.external_staging) and package_allows_run_as(
        adb_path, args.serial, args.device_user)
    if args.compile_filter:
        compile_package(adb_path, args.serial, args.compile_filter, args.device_user)

    def inject(local: pathlib.Path, remote_name: str, expected: str | None = None) -> str:
        if use_run_as:
            return push_private(adb_path, args.serial, local, remote_name, args.device_user,
                                expected_sha256=expected)
        return push_staging(adb_path, args.serial, local, remote_name, expected_sha256=expected)

    def extra_path(remote_name: str) -> str:
        return f"{REMOTE_DIR}/{remote_name}" if use_run_as else f"{STAGING_DIR}/{remote_name}"

    def extra_output(remote_name: str) -> str:
        return remote_name

    model_arg: list[str] = []
    if args.model_dir is not None:
        onnx_files = sorted(args.model_dir.glob("*.onnx"))
        tokenizer = args.model_dir / "tokenizer.json"
        if len(onnx_files) != 1 or not tokenizer.is_file():
            raise MeasurementRunError(
                f"model dir must contain exactly one *.onnx and tokenizer.json: {args.model_dir}")
        injected = sha256_file(onnx_files[0])
        rejected = (pin or {}).get("rejectedContextModelSha256", {})
        if injected in rejected:
            raise MeasurementRunError(
                f"{onnx_files[0]} is the rejected context model {injected}: {rejected[injected]}")
        inject(onnx_files[0], "context.onnx", context_model_sha256)
        inject(tokenizer, "tokenizer.json", (pin or {}).get("contextTokenizerSha256"))
        model_arg = ["-e", "phase0ModelDir", REMOTE_DIR if use_run_as else STAGING_DIR]
    elif context_model_sha256 is not None:
        raise MeasurementRunError(
            "--context-model-sha256 binds metadata to a model this run never injected; "
            "pass --model-dir or drop the hash")

    instrument_args = [
        "-e", "class", TEST_CLASS,
        "-e", "phase0RunId", args.run_id,
        "-e", "phase0Environment", args.environment,
        *model_arg,
    ]
    if not use_run_as:
        instrument_args += ["-e", "phase0UseExternalFiles", "true"]
    if args.corpus is not None:
        inject(args.corpus, "corpus.jsonl")
        remote_outputs.append("measurement.jsonl")
        instrument_args += [
            "-e", "libreboardRequirePhase0Measurement", "true",
            "-e", "phase0CorpusFile", extra_path("corpus.jsonl"),
            "-e", "phase0OutputFile", extra_output("measurement.jsonl"),
        ]
    if args.prediction:
        if args.corpus is None:
            raise MeasurementRunError("--prediction needs --corpus to supply context->target rows")
        remote_outputs.append("prediction-measurement.jsonl")
        instrument_args += [
            "-e", "libreboardRequirePhase0PredictionMeasurement", "true",
            "-e", "phase0PredictionOutputFile", extra_output("prediction-measurement.jsonl"),
        ]
        if args.rescorer_budget_ms is not None:
            instrument_args += ["-e", "phase0RescorerBudgetMs", str(args.rescorer_budget_ms)]
    if args.swipe_corpus is not None:
        inject(args.swipe_corpus, "swipe-corpus.jsonl")
        remote_outputs.append("swipe-measurement.jsonl")
        instrument_args += [
            "-e", "libreboardRequirePhase0SwipeMeasurement", "true",
            "-e", "phase0SwipeCorpusFile", extra_path("swipe-corpus.jsonl"),
            "-e", "phase0SwipeOutputFile", extra_output("swipe-measurement.jsonl"),
        ]
        if args.swipe_warm_arms:
            instrument_args += ["-e", "phase0SwipeWarmArms", "true"]
        if args.swipe_model is not None:
            inject(args.swipe_model, "swipe.onnx", swipe_model_sha256)
            instrument_args += ["-e", "phase0SwipeModelFile", extra_path("swipe.onnx")]
        elif swipe_model_sha256 is not None:
            raise MeasurementRunError(
                "--swipe-model-sha256 binds metadata to a model this run never injected; "
                "pass --swipe-model or drop the hash")
    if args.limit:
        instrument_args += ["-e", "phase0Limit", str(args.limit)]
    if not 0 <= args.shard_index < args.shard_count:
        raise MeasurementRunError("require 0 <= --shard-index < --shard-count")
    if args.shard_count != 1:
        instrument_args += [
            "-e", "phase0ShardCount", str(args.shard_count),
            "-e", "phase0ShardIndex", str(args.shard_index),
        ]

    run_as = ["run-as", PACKAGE]
    if args.device_user is not None:
        run_as += ["--user", str(args.device_user)]
    # Clear the previous run's artifacts. Without this a run that dies before writing leaves an
    # older file in place, and every downstream check would be inspecting stale measurements.
    if use_run_as:
        stale = " ".join(f"files/{REMOTE_DIR}/{name}" for name in (*remote_outputs, "environment.json"))
        adb(adb_path, args.serial, "shell", *run_as, "sh", "-c", f"'rm -f {stale}'")
    else:
        ensure_staging_dir(adb_path, args.serial)
        stale = " ".join(f"{PUBLISH_DIR}/{name}" for name in (*remote_outputs, "environment.json"))
        adb(adb_path, args.serial, "shell", "sh", "-c", f"'rm -f {stale}'")
        print("using external staging (non-debuggable / no run-as)", file=sys.stderr)

    instrument_cmd = ["am", "instrument", "-w", "-r"]
    if args.device_user is not None:
        instrument_cmd += ["--user", str(args.device_user)]
    instrument_output = run_detached_instrumentation(
        adb_path, args.serial, [*instrument_cmd, *instrument_args, RUNNER],
        args.instrument_timeout)
    print(instrument_output)
    test_counts = parse_instrumentation_result(instrument_output, len(remote_outputs))

    adb(adb_path, args.serial, "wait-for-device", timeout=600)
    if use_run_as:
        pulled = b"".join(
            adb_bytes(adb_path, args.serial, "exec-out", *run_as,
                      "cat", f"files/{REMOTE_DIR}/{name}")
            for name in remote_outputs
        )
        sidecar_raw = adb(adb_path, args.serial, "exec-out", *run_as,
                          "cat", f"files/{REMOTE_DIR}/environment.json")
    else:
        pulled = b"".join(
            adb_bytes(adb_path, args.serial, "exec-out", "cat", f"{PUBLISH_DIR}/{name}")
            for name in remote_outputs
        )
        sidecar_raw = adb(adb_path, args.serial, "exec-out", "cat",
                          f"{PUBLISH_DIR}/environment.json")
    row_counts = verify_measurement_rows(pulled, args.run_id, args.environment)
    if args.corpus is not None and not row_counts["tap"]:
        raise MeasurementRunError("the tap corpus produced no measured rows")
    if args.swipe_corpus is not None and not row_counts["swipe"]:
        raise MeasurementRunError("the swipe corpus produced no measured rows")
    sidecar = json.loads(sidecar_raw)
    if sidecar.get("testRunId") != args.run_id or sidecar.get("kind") != args.environment:
        raise MeasurementRunError(
            f"environment sidecar describes run {sidecar.get('testRunId')!r} on "
            f"{sidecar.get('kind')!r}, not {args.run_id!r} on {args.environment!r}")

    # Only now is the run known to be complete and self-consistent.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pulled)

    if args.metadata is not None:
        if args.metadata.is_file():
            metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        else:
            metadata = {"schemaVersion": evaluate_engine.SCHEMA_VERSION, "environments": []}
        for key, value in (("appCommit", args.app_commit),
                           ("coreApkSha256", args.core_apk_sha256),
                           ("swipeModelSha256", swipe_model_sha256),
                           ("contextModelSha256", context_model_sha256)):
            if value is not None:
                metadata[key] = value
        if pin is not None:
            # Carry the candidate into the evidence so the report states what it qualifies.
            metadata["artifactPin"] = {
                "candidateId": pin["candidateId"],
                "contextModelSha256": pin["contextModelSha256"],
                "swipeModelSha256": pin["swipeModelSha256"],
            }
        environments = [e for e in metadata.get("environments", []) if e.get("kind") != sidecar["kind"]]
        environments.append(sidecar)
        metadata["environments"] = environments
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(args.output), "bytes": len(pulled),
        "sha256": hashlib.sha256(pulled).hexdigest(),
        "runId": args.run_id, "environment": args.environment,
        "deviceEnvironment": sidecar,
        "measuredRows": row_counts,
        "instrumentedTests": test_counts,
        "artifactPin": (pin or {}).get("candidateId"),
        "externalStaging": not use_run_as,
        "compileFilter": args.compile_filter,
        "releaseEligible": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasurementRunError as failure:
        print(f"ERROR: {failure}", file=sys.stderr)
        raise SystemExit(2) from failure
