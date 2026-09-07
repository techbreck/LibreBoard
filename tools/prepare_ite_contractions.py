#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare real contraction replays from the same pinned, participant-grouped ITE source."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import zipfile

import model_sources
import prepare_ite_valid_words as ite
import prepare_tap_evaluation as tap

CONTRACTIONS = frozenset("""aren't can't couldn't didn't doesn't don't hadn't hasn't haven't isn't
mightn't mustn't needn't shan't shouldn't wasn't weren't won't wouldn't i'm i've i'd i'll you're
you've you'd you'll he's he'd he'll she's she'd she'll it's it'd it'll we're we've we'd we'll
they're they've they'd they'll that's that'll there's there'll who's who'll what's what'll let's
where's here's how's when's why's""".split())
DEFAULT_OUTPUT_ROOT = ite.ROOT / "build/evaluation-sources/ite-contractions-v1"
PROTOCOL = (
    "Human pre-autocorrection transcription words from the pinned Aalto ITE English v1 source. "
    "An explicit English contraction list excludes noun possessives. Exact typed-prefix/reference "
    "alignment is required. Publisher autocorrection output and participant metadata are discarded. "
    "All sessions of a participant share the same split identity as the ITE valid-word component; "
    "repeated participant/prefix/raw/target cases are deduplicated. No spatial data are invented."
)


def aligned_contraction(row, sentence):
    raw, target, current = row["TYPED_WORD"], row["ORIGINAL_WORD"], row["CURRENT_INPUT"]
    if target not in CONTRACTIONS:
        return None, "not_contraction"
    if not re.fullmatch("[a-z]+(?:'[a-z]+)?", raw) or len(raw) > 64:
        return None, "invalid_raw"
    if len(current) > 4096 or not current.lower().endswith(raw):
        return None, "raw_suffix"
    prefix = current[:-len(raw)]
    if not prefix.strip() or not prefix.endswith(" ") or not sentence.lower().startswith(prefix.lower()):
        return None, "prefix"
    if not re.match(re.escape(target) + r"(?=$|[^A-Za-z'])", sentence[len(prefix):].lower()):
        return None, "reference_alignment"
    return (raw, target, prefix), None


def prepare(source, source_root, output_root):
    for item in source["files"]:
        if not ite.verify_file(source_root / item["file"], item):
            raise ite.IteDataError("selected source file is missing or mismatched; use --fetch")
    sentences = {r["SENTENCE_ID"]: r["SENTENCE"] for r in ite.rows(source_root / "sentences.csv")}
    sections = {r["TEST_SECTION_ID"]: (r["PARTICIPANT_ID"], r["SENTENCE_ID"])
                for r in ite.rows(source_root / "test_sections_labeled.csv")}
    policy = tap.load_policy(tap.DEFAULT_POLICY)
    seen, records = set(), []
    rejected, counts = collections.Counter(), collections.Counter()
    for row in ite.rows(source_root / "ac_words_en.csv"):
        section = sections.get(row["TEST_SECTION_ID"])
        if not section or section[1] != row["SENTENCE_ID"]:
            rejected["section"] += 1
            continue
        case, reason = aligned_contraction(row, sentences.get(row["SENTENCE_ID"], ""))
        if reason:
            rejected[reason] += 1
            continue
        raw, target, prefix = case
        identity = (section[0], prefix.lower(), raw, target)
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        # Deliberately identical across ITE components: one person's sessions cannot cross splits.
        session = "ite-en:" + section[0]
        counts[(tap.split_for_session(session, policy), "correct" if raw != target else "keep")] += 1
        records.append({"schemaVersion": 1, "id": hashlib.sha256(ite.canonical(identity)).hexdigest(),
                        "sessionId": session, "category": "lexical", "lexicalKind": "contraction",
                        "raw": raw, "target": target, "languageTag": "en-US", "fieldClass": "plain",
                        "precedingContext": prefix[-policy.maximum_context_codepoints:],
                        "collectionMethod": "human_replay", "touchPoints": []})
    records.sort(key=lambda row: row["id"])
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        staged = pathlib.Path(temporary)
        data = staged / "contractions.jsonl"
        with data.open("wb") as stream:
            for record in records:
                stream.write(ite.canonical(record) + b"\n")
        manifest = {"schemaVersion": 1, "datasetId": "aalto-ite-contractions-en-v1", "dataFile": data.name,
                    "dataSha256": model_sources.file_sha256(data), "license": source["license"],
                    "containsHumanContributions": True, "collectionProtocol": PROTOCOL, "consentStatement": ite.CONSENT}
        report = {"schemaVersion": 1, "records": len(records), "releaseEligible": False,
                  "canonicalSourceManifestSha256": hashlib.sha256(ite.canonical(source)).hexdigest(),
                  "adapterSha256": model_sources.file_sha256(pathlib.Path(__file__)),
                  "sharedSourceAdapterSha256": model_sources.file_sha256(pathlib.Path(ite.__file__)),
                  "contractionsSha256": hashlib.sha256(ite.canonical(sorted(CONTRACTIONS))).hexdigest(),
                  "splits": {split: {label: counts[(split, label)] for label in ("correct", "keep")}
                             for split in tap.SPLITS}, "rejected": dict(sorted(rejected.items())),
                  "limitations": ["Selection is conditioned on publisher-detected autocorrection events.",
                                  "Publisher event labels are automatic; exact reference alignment is required.",
                                  "This text-only component supplies contractions, not personal words or compounds.",
                                  "No spatial, device, performance or release-quality evidence is supplied."]}
        for name, value in (("source-manifest.json", manifest), ("preparation-report.json", report)):
            (staged / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        for name in (data.name, "source-manifest.json", "preparation-report.json"):
            os.replace(staged / name, output_root / name)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=ite.DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=pathlib.Path, default=ite.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = ite.load_source(args.manifest)
        if args.fetch:
            ite.fetch_files(args.source_root, source)
        print(json.dumps(prepare(source, args.source_root, args.output_root), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, csv.Error, zipfile.BadZipFile) as failure:
        print(f"ITE contraction preparation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
