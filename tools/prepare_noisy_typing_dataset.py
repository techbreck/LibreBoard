#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare bounded human tap replays from the version-pinned OSF noisy-typing archive."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import urllib.request
import zipfile

import model_sources
import prepare_tap_evaluation

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models/evaluation/noisy-typing-v1.json"
DEFAULT_SOURCE_ROOT = ROOT / "build/evaluation-sources"
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT / "noisy-phone-v1"
EXPERIMENTS = {"eyesfree_exp1", "vt_exp1", "vt_exp2", "vt_exp3", "vt_pilot2"}
LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
MAXIMUM_ENTRY_BYTES = 4 * 1024 * 1024
PROTOCOL = (
    "Publisher force-aligned phone test_word recordings, one touch-down sample per physical tap. "
    "Raw text is nearest-key replay using the recorded layout. No resampling, jitter, or synthetic "
    "errors. Only the public reference word is retained; LEFT, RIGHT, ORIG and surrounding prose "
    "are discarded. Conditions/sessions for each experiment-scoped participant stay together. "
    "Pilot 1 is excluded because Pilot 2 reuses its participants; author-only data is excluded."
)
RIGHTS = (
    "The authors released pseudonymous study touch recordings publicly under CC BY 4.0 on OSF. "
    "OSF's project license is recorded in the pinned source manifest. The archive does not include "
    "participant consent forms or a global cross-experiment participant mapping."
)


class NoisyTypingError(ValueError):
    pass


def load_source(path=DEFAULT_MANIFEST):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 65536:
        raise NoisyTypingError("source manifest is missing, linked, or too large")
    value = json.loads(path.read_bytes())
    expected = {"schemaVersion", "datasetId", "sourceUrl", "authors", "license", "licenseEvidenceUrl",
                "licenseId", "licenseUrl", "downloadUrl", "archiveFile", "archiveVersion", "archiveBytes",
                "archiveSha256", "includedExperiments"}
    if not isinstance(value, dict) or set(value) != expected:
        raise NoisyTypingError("invalid noisy-typing source schema")
    fixed = {"schemaVersion": 1, "datasetId": "vertanen-noisy-phone-v1", "sourceUrl": "https://osf.io/5xwng/",
             "authors": ["Keith Vertanen", "Per Ola Kristensson"], "license": "CC-BY-4.0",
             "licenseEvidenceUrl": "https://api.osf.io/v2/nodes/5xwng/",
             "licenseId": "563c1cf88c5e4a3877f9e96a", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
             "downloadUrl": "https://osf.io/download/7ev5z/?revision=1",
             "archiveFile": "vertanen_noisy_typing_iui2023.zip", "archiveVersion": 1,
             "archiveBytes": 18052094, "includedExperiments": sorted(EXPERIMENTS)}
    if any(type(value[key]) is not type(wanted) or value[key] != wanted for key, wanted in fixed.items()):
        raise NoisyTypingError("source does not match the reviewed OSF version/license/experiment scope")
    if not isinstance(value["archiveSha256"], str) or not model_sources.SHA256.fullmatch(value["archiveSha256"]):
        raise NoisyTypingError("invalid source archive hash")
    return value


def verify_archive(path, source):
    if (not path.is_file() or path.is_symlink() or path.stat().st_size != source["archiveBytes"]
            or model_sources.file_sha256(path) != source["archiveSha256"]):
        raise NoisyTypingError("source archive missing or does not match pinned version; use --fetch")


def fetch_archive(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".download")
    staged = pathlib.Path(name)
    try:
        with os.fdopen(descriptor, "wb") as target, urllib.request.urlopen(source["downloadUrl"], timeout=30) as response:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > source["archiveBytes"]:
                    raise NoisyTypingError("download exceeds pinned byte count")
                target.write(chunk)
        verify_archive(staged, source)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def read_entry(archive, name):
    info = archive.getinfo(name)
    if not 0 < info.file_size <= MAXIMUM_ENTRY_BYTES:
        raise NoisyTypingError("archive entry exceeds bounded text size")
    return archive.read(info).decode("utf-8-sig")


def parse_keyboard(text):
    keys = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) != 5 or parts[0] in keys or parts[0] not in LETTERS | {"<sp>", "<b>"}:
            raise NoisyTypingError("invalid or duplicate keyboard key")
        values = tuple(float(v) for v in parts[1:])
        if any(not math.isfinite(v) or abs(v) > 10000 for v in values) or min(values[2:]) <= 0:
            raise NoisyTypingError("invalid keyboard geometry")
        keys[parts[0]] = values
    if not LETTERS.issubset(keys):
        raise NoisyTypingError("keyboard omits letters")
    left = min(keys[k][0] - keys[k][2] / 2 for k in LETTERS)
    right = max(keys[k][0] + keys[k][2] / 2 for k in LETTERS)
    top = min(keys[k][1] - keys[k][3] / 2 for k in LETTERS)
    bottom = max(keys[k][1] + keys[k][3] / 2 for k in LETTERS)
    return keys, (left, top, right - left, bottom - top)


def parse_word(target, event_line, keyboard):
    if not re.fullmatch(r"[a-z]{1,128}", target):
        return None, "non_word_target"
    keys, (left, top, width, height) = keyboard
    taps = event_line.split("|")
    if not 1 <= len(taps) <= 128:
        return None, "tap_count"
    letters, points = [], []
    initial_time = previous_time = None
    for tap in taps:
        samples = tap.split(";")
        if not 1 <= len(samples) <= 1024:
            raise NoisyTypingError("unbounded touch event")
        first = None
        last_time = None
        for sample in samples:
            parts = sample.split(",")
            if len(parts) != 3:
                raise NoisyTypingError("invalid touch sample")
            x, y, timestamp = (float(v) for v in parts)
            if not all(math.isfinite(v) for v in (x, y, timestamp)) or timestamp < 0:
                raise NoisyTypingError("invalid touch coordinate/time")
            if last_time is not None and timestamp < last_time:
                return None, "non_monotonic_time"
            last_time = timestamp
            if first is None:
                first = (x, y, timestamp)
        x, y, timestamp = first
        if previous_time is not None and timestamp < previous_time:
            return None, "non_monotonic_time"
        previous_time = last_time
        initial_time = timestamp if initial_time is None else initial_time
        nx, ny = (x - left) / width, (y - top) / height
        if not -0.5 <= nx <= 1.5 or not -0.5 <= ny <= 1.5:
            return None, "out_of_bounds"
        distances = {k: (v[0] - x) ** 2 + (v[1] - y) ** 2 for k, v in keys.items()}
        nearest = min(distances, key=distances.get)
        if sum(v == distances[nearest] for v in distances.values()) != 1:
            return None, "ambiguous_key"
        if nearest not in LETTERS:
            return None, "control_key"
        letters.append(nearest)
        relative_time = round(timestamp - initial_time)
        if relative_time > 60000:
            return None, "word_duration"
        points.append({"x": nx, "y": ny, "timeMillis": relative_time})
    raw = "".join(letters)
    return ((raw, points), None) if raw != target else (None, "correct_replay")


def rows_from_log(text, name, keyboard, counts):
    parts = pathlib.PurePosixPath(name).parts
    experiment, participant_file = parts[2], parts[-1]
    match = re.match(r"(p[0-9]+)(?:_|\.)", participant_file)
    if not match:
        raise NoisyTypingError("unexpected participant filename")
    session = f"noisy-phone:{experiment}:{match[1]}"
    for block_index, block in enumerate(re.split(r"\r?\n\s*\r?\n", text)):
        fields = collections.defaultdict(list)
        for line in block.splitlines():
            if not line.strip():
                continue
            field, separator, value = line.partition(":")
            if not separator or field not in {"ID", "REF", "LEFT", "RIGHT", "ORIG", "IN"}:
                raise NoisyTypingError("unrecognized log record")
            if field in {"REF", "IN"}:
                if len(value) > 65536:
                    raise NoisyTypingError("unbounded log line")
                fields[field].append(value.strip())
        if not fields:
            continue
        if not fields["REF"]:
            counts["orphan_input"] += 1
            continue
        if len(fields["REF"]) != 1 or len(fields["REF"][0]) > 4096:
            raise NoisyTypingError("invalid reference sentence")
        targets = fields["REF"][0].split()
        if len(targets) != len(fields["IN"]):
            counts["unaligned_sentence"] += 1
            continue
        for word_index, (target, event) in enumerate(zip(targets, fields["IN"])):
            parsed, reason = parse_word(target, event, keyboard)
            if parsed is None:
                counts[reason] += 1
                continue
            raw, touches = parsed
            identity = hashlib.sha256(f"{name}:{block_index}:{word_index}".encode()).hexdigest()
            yield {"schemaVersion": 1, "id": identity, "sessionId": session, "category": "tap_error",
                   "target": target, "raw": raw, "languageTag": "en-US", "precedingContext": "",
                   "fieldClass": "plain", "collectionMethod": "human_replay", "touchPoints": touches}


def prepare(args):
    source = load_source(args.manifest)
    archive_path = args.source_root / source["archiveFile"]
    if args.fetch:
        fetch_archive(archive_path, source)
    verify_archive(archive_path, source)
    counts, experiment_counts, split_counts = collections.Counter(), collections.Counter(), collections.Counter()
    rows, seen = [], set()
    policy = prepare_tap_evaluation.load_policy()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > 5000 or sum(i.file_size for i in archive.infolist()) > 64 * 1024 * 1024:
            raise NoisyTypingError("archive inventory exceeds bounds or repeats names")
        for name in sorted(names):
            parts = pathlib.PurePosixPath(name).parts
            if (len(parts) != 5 or parts[:2] != ("noisy_typing", "phone") or parts[2] not in EXPERIMENTS
                    or parts[3] != "test_word" or not name.endswith(".log")):
                continue
            experiment = parts[2]
            layout = "keyboard_iphone4.txt" if experiment == "eyesfree_exp1" else "keyboard_nexus4.txt"
            if experiment == "vt_exp3":
                condition = pathlib.PurePosixPath(name).stem.rsplit("_", 1)[-1]
                if condition not in {"normal", "small", "tiny"}:
                    raise NoisyTypingError("unrecognized keyboard size condition")
                layout = f"keyboard_nexus4_{condition}.txt"
            keyboard = parse_keyboard(read_entry(archive, "/".join(parts[:3]) + "/" + layout))
            for row in rows_from_log(read_entry(archive, name), name, keyboard, counts):
                signature = json.dumps([row["target"], row["raw"], row["touchPoints"]], sort_keys=True)
                if signature in seen:
                    counts["duplicate_replay"] += 1
                    continue
                seen.add(signature)
                rows.append(row)
                experiment_counts[experiment] += 1
                split_counts[prepare_tap_evaluation.split_for_session(row["sessionId"], policy)] += 1
    if not rows:
        raise NoisyTypingError("source produced no tap-error records")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    data = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    manifest = {"schemaVersion": 1, "datasetId": source["datasetId"], "dataFile": "tap-errors.jsonl",
                "dataSha256": hashlib.sha256(data).hexdigest(), "license": source["license"],
                "collectionProtocol": PROTOCOL, "containsHumanContributions": True, "consentStatement": RIGHTS}
    report = {"schemaVersion": 1, "sourceManifestSha256": model_sources.file_sha256(args.manifest),
              "archiveSha256": source["archiveSha256"], "adapterSha256": model_sources.file_sha256(pathlib.Path(__file__)),
              "records": len(rows), "experiments": dict(experiment_counts), "splits": dict(split_counts),
              "rejected": dict(counts), "globalParticipantDisjointnessVerified": False,
              "releaseEligible": False, "limitations": ["Publisher force alignment supplies word boundaries.",
                  "Participant identifiers are experiment-scoped; global cross-experiment identities are unavailable.",
                  "This tap-only component cannot satisfy the complete Phase 0 corpus gate."]}
    for filename, payload in (("tap-errors.jsonl", data), ("source-manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()),
                              ("preparation-report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())):
        descriptor, temporary = tempfile.mkstemp(dir=output)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, output / filename)
        finally:
            pathlib.Path(temporary).unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fetch", action="store_true")
    try:
        print(json.dumps(prepare(parser.parse_args(argv)), indent=2, sort_keys=True))
        return 0
    except (NoisyTypingError, OSError, ValueError, KeyError, zipfile.BadZipFile) as failure:
        print(f"noisy typing preparation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
