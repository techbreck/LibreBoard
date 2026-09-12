#!/usr/bin/env python3
"""Drive the Phase 0 measurement instrumentation on a bound device or emulator.

Pushes the prepared tap/swipe corpora and (optionally) the context and swipe models into the
debug app's private files directory, runs Phase0MeasurementInstrumentedTest, and pulls the
schema-3 measurement JSONL plus an environment sidecar merged into the --metadata file. The
output is raw measurement input for evaluate_engine.py, not release evidence alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

PACKAGE = "org.libreboard.keyboard.debug"
RUNNER = "org.libreboard.keyboard.debug.test/androidx.test.runner.AndroidJUnitRunner"
TEST_CLASS = "helium314.keyboard.latin.engine.Phase0MeasurementInstrumentedTest"
REMOTE_DIR = "phase0-measurement"
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
ENVIRONMENTS = {"stock_android_hardware", "grapheneos_hardware", "low_ram_emulator"}


def adb(adb_path: str, serial: str | None, *args: str, stdin: bytes | None = None,
        timeout: int = 600) -> str:
    command = [adb_path]
    if serial:
        command += ["-s", serial]
    command += list(args)
    result = subprocess.run(command, input=stdin, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {result.stderr.decode(errors='replace')[:512]}")
    return result.stdout.decode(errors="replace")


def push_private(adb_path: str, serial: str | None, local: pathlib.Path, remote_name: str) -> None:
    if not SAFE_NAME.fullmatch(remote_name):
        raise RuntimeError(f"unsafe remote name {remote_name}")
    staging = f"/data/local/tmp/{remote_name}.phase0push"
    adb(adb_path, serial, "push", str(local), staging)
    adb(adb_path, serial, "shell", "run-as", PACKAGE, "sh", "-c",
        f"'mkdir -p files/{REMOTE_DIR} && cp {staging} files/{REMOTE_DIR}/{remote_name}'")
    adb(adb_path, serial, "shell", "rm", "-f", staging)


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
    parser.add_argument("--instrument-timeout", type=int, default=4 * 60 * 60,
                        help="seconds to wait for the instrumented run (default 4h)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--metadata", type=pathlib.Path,
                        help="schema-3 metadata JSON to create or update with this run's environment")
    parser.add_argument("--app-commit", help="full lowercase git commit of the app build")
    parser.add_argument("--core-apk-sha256", help="SHA-256 of the installed app APK")
    parser.add_argument("--swipe-model-sha256")
    parser.add_argument("--context-model-sha256")
    args = parser.parse_args(argv)

    if not 0 < len(args.run_id) <= 512:
        return 2
    if args.corpus is None and args.swipe_corpus is None:
        raise RuntimeError("at least one of --corpus or --swipe-corpus is required")
    adb_path = str(args.adb)
    remote_outputs: list[str] = []

    model_arg: list[str] = []
    if args.model_dir is not None:
        onnx_files = sorted(args.model_dir.glob("*.onnx"))
        tokenizer = args.model_dir / "tokenizer.json"
        if len(onnx_files) != 1 or not tokenizer.is_file():
            raise RuntimeError(
                f"model dir must contain exactly one *.onnx and tokenizer.json: {args.model_dir}")
        push_private(adb_path, args.serial, onnx_files[0], "context.onnx")
        push_private(adb_path, args.serial, tokenizer, "tokenizer.json")
        model_arg = ["-e", "phase0ModelDir", REMOTE_DIR]

    instrument_args = [
        "-e", "class", TEST_CLASS,
        "-e", "phase0RunId", args.run_id,
        "-e", "phase0Environment", args.environment,
        *model_arg,
    ]
    if args.corpus is not None:
        push_private(adb_path, args.serial, args.corpus, "corpus.jsonl")
        remote_outputs.append("measurement.jsonl")
        instrument_args += [
            "-e", "libreboardRequirePhase0Measurement", "true",
            "-e", "phase0CorpusFile", f"{REMOTE_DIR}/corpus.jsonl",
            "-e", "phase0OutputFile", "measurement.jsonl",
        ]
    if args.swipe_corpus is not None:
        push_private(adb_path, args.serial, args.swipe_corpus, "swipe-corpus.jsonl")
        remote_outputs.append("swipe-measurement.jsonl")
        instrument_args += [
            "-e", "libreboardRequirePhase0SwipeMeasurement", "true",
            "-e", "phase0SwipeCorpusFile", f"{REMOTE_DIR}/swipe-corpus.jsonl",
            "-e", "phase0SwipeOutputFile", "swipe-measurement.jsonl",
        ]
        if args.swipe_model is not None:
            push_private(adb_path, args.serial, args.swipe_model, "swipe.onnx")
            instrument_args += ["-e", "phase0SwipeModelFile", f"{REMOTE_DIR}/swipe.onnx"]
    if args.limit:
        instrument_args += ["-e", "phase0Limit", str(args.limit)]
    if not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("require 0 <= --shard-index < --shard-count")
    if args.shard_count != 1:
        instrument_args += [
            "-e", "phase0ShardCount", str(args.shard_count),
            "-e", "phase0ShardIndex", str(args.shard_index),
        ]
    print(adb(adb_path, args.serial, "shell", "am", "instrument", "-w", "-r",
              *instrument_args, RUNNER, timeout=args.instrument_timeout))

    pulled = b"".join(
        adb(adb_path, args.serial, "exec-out", "run-as", PACKAGE,
            "cat", f"files/{REMOTE_DIR}/{name}").encode()
        for name in remote_outputs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pulled)

    sidecar_raw = adb(adb_path, args.serial, "exec-out", "run-as", PACKAGE,
                      "cat", f"files/{REMOTE_DIR}/environment.json")
    sidecar = json.loads(sidecar_raw)
    if args.metadata is not None:
        if args.metadata.is_file():
            metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        else:
            metadata = {"schemaVersion": 3, "environments": []}
        for key, value in (("appCommit", args.app_commit),
                           ("coreApkSha256", args.core_apk_sha256),
                           ("swipeModelSha256", args.swipe_model_sha256),
                           ("contextModelSha256", args.context_model_sha256)):
            if value is not None:
                metadata[key] = value
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
        "releaseEligible": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
