#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare word split/join spacing replays from the same pinned, participant-grouped ITE source."""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile

import model_sources
import prepare_ite_valid_words as ite
import prepare_tap_evaluation as tap

DEFAULT_OUTPUT_ROOT = ite.ROOT / "build/evaluation-sources/ite-spacing-v1"
MAXIMUM_SITES_PER_SECTION = 16
WORD = re.compile(r"[a-z]+")
PROTOCOL = (
    "Human word split/join sites from the pinned Aalto ITE English v1 source. A site is a "
    "token-level alignment block where the participant's letter tokens regroup the reference "
    "tokens without changing the concatenated letters. Only all-letter token groups qualify; "
    "punctuation-adjacent regroupings, substitutions, insertions, and deletions are not spacing "
    "rows. The bounded preceding context is the participant's own typed prefix. Participant IDs "
    "keep all sessions together and match the other ITE components so one person's sessions "
    "cannot cross splits; repeated participant/context/raw/target cases are deduplicated. "
    "Submitted sentence metadata beyond the typed text is discarded. No spatial data are invented."
)


def token_spans(text):
    spans = []
    start = 0
    for token in text.split():
        offset = text.index(token, start)
        spans.append((token, offset))
        start = offset + len(token)
    return spans


def spacing_sites(user_tokens, prompt_tokens):
    matcher = difflib.SequenceMatcher(None, [t.lower() for t in user_tokens],
                                      [t.lower() for t in prompt_tokens], autojunk=False)
    for opcode, u_start, u_end, p_start, p_end in matcher.get_opcodes():
        if opcode != "replace":
            continue
        user_group = user_tokens[u_start:u_end]
        prompt_group = prompt_tokens[p_start:p_end]
        if (not user_group or not prompt_group or len(user_group) == len(prompt_group)
                or "".join(user_group).lower() != "".join(prompt_group).lower()):
            continue
        if all(WORD.fullmatch(token.lower()) for token in user_group + prompt_group):
            yield u_start, u_end, p_start, p_end


def prepare(source, source_root, output_root):
    for item in source["files"]:
        if not ite.verify_file(source_root / item["file"], item):
            raise ite.IteDataError("selected source file is missing or mismatched; use --fetch")
    sentences = {r["SENTENCE_ID"]: r["SENTENCE"] for r in ite.rows(source_root / "sentences.csv")}
    policy = tap.load_policy(tap.DEFAULT_POLICY)
    counts, split_counts = collections.Counter(), collections.Counter()
    rows, seen = [], set()
    for section in ite.rows(source_root / "test_sections_labeled.csv"):
        sentence = sentences.get(section["SENTENCE_ID"])
        typed = section["USER_INPUT"]
        if not sentence or not isinstance(typed, str) or len(typed) > 8192:
            counts["unmatched_section"] += 1
            continue
        session = "ite-en:" + section["PARTICIPANT_ID"]
        user_spans = token_spans(typed)
        prompt_tokens = sentence.split()
        emitted = 0
        for u_start, u_end, p_start, p_end in spacing_sites([t for t, _ in user_spans], prompt_tokens):
            if emitted >= MAXIMUM_SITES_PER_SECTION:
                counts["site_cap"] += 1
                break
            raw = " ".join(user_spans[u][0] for u in range(u_start, u_end)).lower()
            target = " ".join(prompt_tokens[p_start:p_end]).lower()
            context = typed[:user_spans[u_start][1]].rstrip()[-200:]
            signature = (session, context, raw, target)
            if signature in seen:
                counts["duplicate_case"] += 1
                continue
            seen.add(signature)
            emitted += 1
            counts["emitted"] += 1
            split_counts[tap.split_for_session(session, policy)] += 1
            identity = hashlib.sha256(
                f"ite-spacing:{section['TEST_SECTION_ID']}:{u_start}:{p_start}".encode()).hexdigest()
            rows.append({"schemaVersion": 1, "id": identity, "sessionId": session,
                         "category": "spacing", "raw": raw, "target": target,
                         "languageTag": "en-US", "precedingContext": context,
                         "fieldClass": "plain", "collectionMethod": "human_replay",
                         "touchPoints": []})
    if not rows:
        raise ite.IteDataError("source produced no spacing records")
    output_root.mkdir(parents=True, exist_ok=True)
    data = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    manifest = {"schemaVersion": 1, "datasetId": "aalto-ite-spacing-en-v1", "dataFile": "spacing.jsonl",
                "dataSha256": hashlib.sha256(data).hexdigest(), "license": source["license"],
                "collectionProtocol": PROTOCOL, "containsHumanContributions": True,
                "consentStatement": ite.CONSENT}
    report = {"schemaVersion": 1, "sourceManifestSha256": model_sources.file_sha256(
                  ite.DEFAULT_MANIFEST), "adapterSha256": model_sources.file_sha256(pathlib.Path(__file__)),
              "records": len(rows), "splits": dict(split_counts), "rejected": dict(counts),
              "releaseEligible": False,
              "limitations": ["Letter-token regroupings only; punctuation-adjacent spacing differences are excluded.",
                              "No touch coordinates are available from this keystroke-level source.",
                              "This component cannot satisfy the complete Phase 0 corpus gate."]}
    for name, value in (("spacing.jsonl", data), ("source-manifest.json", manifest),
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
    parser.add_argument("--manifest", type=pathlib.Path, default=ite.DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=pathlib.Path,
                        default=ite.ROOT / "build/evaluation-sources/ite-selected-v1")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(ite.load_source(args.manifest), args.source_root, args.output_root),
                         sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as failure:
        print(f"ite-spacing preparation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
