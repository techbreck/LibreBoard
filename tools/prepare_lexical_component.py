#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare project-authored personal/compound lexical replays for the Phase 0 tap corpus."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import random
import sys
import tempfile

import model_sources
import prepare_tap_evaluation as tap

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "build/evaluation-sources/project-lexical-v1"
SEED = 20260911
SESSION_ROWS = 24
SESSIONS_PER_KIND = 60
ADJACENT = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "ryfg", "y": "tugh", "u": "yihj",
    "i": "uojk", "o": "iplk", "p": "ol", "a": "qwsz", "s": "awedzx", "d": "serfcx",
    "f": "drtgvc", "g": "ftyhvb", "h": "gyujnb", "j": "huiknm", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
NAME_SYLLABLES = (
    "an", "ar", "bel", "bris", "cor", "del", "dov", "el", "en", "far", "gal", "har", "in",
    "jan", "jor", "kal", "kel", "lan", "lis", "lor", "mar", "mel", "mir", "nal", "nor",
    "quin", "ren", "sar", "sha", "sor", "tal", "tan", "tor", "val", "ven", "ver", "vin",
    "wen", "wil", "zan", "zel",
)
NAME_ENDINGS = ("a", "e", "i", "o", "ia", "en", "on", "is", "us", "el")
COMPOUNDS = (
    "aircraft", "airport", "background", "bathroom", "bedtime", "breakfast", "butterfly",
    "campground", "classroom", "crossword", "daylight", "deadline", "downtown", "dragonfly",
    "driveway", "earring", "everyday", "farmhouse", "firewood", "flashlight", "football",
    "footprint", "freeway", "gateway", "goldfish", "grandmother", "grasshopper", "greenhouse",
    "haircut", "handbag", "headache", "heartbeat", "highway", "homework", "horseback",
    "hotspot", "housework", "jellyfish", "keyboard", "landlord", "laptop", "lawnmower",
    "lighthouse", "mailbox", "midnight", "motorcycle", "motorway", "network", "newsletter",
    "newspaper", "notebook", "outdoors", "overnight", "paperback", "password", "paycheck",
    "playground", "postcode", "railroad", "railway", "rainbow", "roommate", "seafood",
    "shortcut", "sidewalk", "skateboard", "software", "spaceship", "spotlight", "staircase",
    "sunflower", "sunlight", "teammate", "teaspoon", "textbook", "timeline", "underground",
    "upstairs", "waterfall", "weekday", "weekend", "wildlife", "workshop",
)
PERSONAL_CONTEXTS = (
    "ask ", "call ", "email ", "from ", "invite ", "meet ", "message ", "see ", "text ",
    "thanks ", "with ",
)
COMPOUND_CONTEXTS = (
    "a ", "check the ", "every ", "my ", "near the ", "on the ", "read the ", "the ",
    "this ", "to the ",
)
PROTOCOL = (
    "Project-authored lexical cases. Personal targets are invented synthetic names generated from "
    "a fixed syllable list and carried in each row's bounded personalWords fixture; no real "
    "personal data or device dictionary contents are used. Compound targets are common English "
    "compound words; raw text is a deterministic adjacent-key typo or a split-form variant. "
    "Preceding contexts are fixed synthetic prompt fragments. Rows are grouped into "
    "project-authored sessions so whole sessions share one salted split."
)


def typo(word, rng):
    positions = [i for i, c in enumerate(word) if c in ADJACENT and ADJACENT[c]]
    index = rng.choice(positions)
    replacement = rng.choice(ADJACENT[word[index]])
    return word[:index] + replacement + word[index + 1:]


def synthetic_name(rng):
    name = "".join(rng.choice(NAME_SYLLABLES) for _ in range(rng.randint(2, 3)))
    return name + rng.choice(NAME_ENDINGS)


def personal_rows(session, rng):
    rows = []
    for index in range(SESSION_ROWS):
        target = synthetic_name(rng)
        fixture = list(dict.fromkeys([target] + [synthetic_name(rng) for _ in range(4)]))[:4]
        yield {"schemaVersion": 1,
               "id": hashlib.sha256(f"project-lexical:personal:{session}:{index}".encode()).hexdigest(),
               "sessionId": f"project-lexical:v1:personal:{session:03d}", "category": "lexical",
               "lexicalKind": "personal", "raw": typo(target, rng), "target": target,
               "languageTag": "en-US", "precedingContext": rng.choice(PERSONAL_CONTEXTS),
               "fieldClass": "plain", "collectionMethod": "project_authored", "touchPoints": [],
               "personalWords": fixture}


def compound_rows(session, rng):
    for index in range(SESSION_ROWS):
        target = rng.choice(COMPOUNDS)
        if rng.random() < 0.5:
            split_at = rng.randint(2, len(target) - 2)
            raw = target[:split_at] + " " + target[split_at:]
        else:
            raw = typo(target, rng)
        yield {"schemaVersion": 1,
               "id": hashlib.sha256(f"project-lexical:compound:{session}:{index}".encode()).hexdigest(),
               "sessionId": f"project-lexical:v1:compound:{session:03d}", "category": "lexical",
               "lexicalKind": "compound", "raw": raw, "target": target,
               "languageTag": "en-US", "precedingContext": rng.choice(COMPOUND_CONTEXTS),
               "fieldClass": "plain", "collectionMethod": "project_authored", "touchPoints": []}


def prepare(output_root):
    rng = random.Random(SEED)
    policy = tap.load_policy(tap.DEFAULT_POLICY)
    split_counts, kind_counts = collections.Counter(), collections.Counter()
    rows = []
    for session in range(SESSIONS_PER_KIND):
        for row in personal_rows(session, rng):
            rows.append(row)
        for row in compound_rows(session, rng):
            rows.append(row)
    for row in rows:
        split_counts[tap.split_for_session(row["sessionId"], policy)] += 1
        kind_counts[row["lexicalKind"]] += 1
    output_root.mkdir(parents=True, exist_ok=True)
    data = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    manifest = {"schemaVersion": 1, "datasetId": "project-lexical-en-v1", "dataFile": "lexical.jsonl",
                "dataSha256": hashlib.sha256(data).hexdigest(), "license": "CC-BY-4.0",
                "collectionProtocol": PROTOCOL, "containsHumanContributions": False,
                "consentStatement": ""}
    report = {"schemaVersion": 1, "adapterSha256": model_sources.file_sha256(pathlib.Path(__file__)),
              "seed": SEED, "records": len(rows), "lexicalKinds": dict(kind_counts),
              "splits": dict(split_counts), "releaseEligible": False,
              "limitations": ["Synthetic cases measure fixture/compound handling, not human lexical behavior.",
                              "This component cannot satisfy the complete Phase 0 corpus gate."]}
    for name, value in (("lexical.jsonl", data), ("source-manifest.json", manifest),
                        ("preparation-report.json", report)):
        payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(dir=output_root)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, output_root / name)
        finally:
            pathlib.Path(temporary).unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.output_root), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as failure:
        print(f"project-lexical preparation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
