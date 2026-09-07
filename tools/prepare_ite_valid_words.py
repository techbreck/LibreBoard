#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Prepare participant-separated real-word typing cases from the pinned Aalto ITE source."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import urllib.request
import zipfile

import model_sources
import prepare_tap_evaluation as tap

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models/evaluation/ite-valid-words-v1.json"
DEFAULT_SOURCE_ROOT = ROOT / "build/evaluation-sources/ite-selected-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "build/evaluation-sources/ite-valid-words-v1"
DEFAULT_VOCABULARY = ROOT / "build/device-evidence/static-swipe-lexicon.json"
PROTOCOL = (
    "Pre-autocorrection human transcription words from Aalto ITE English v1. The typed prefix must "
    "match the public reference prefix and the reference word must match its exact position. Both "
    "words must occur in the pinned native English vocabulary. Autocorrection outputs and demographic "
    "metadata are discarded. Participant IDs keep all sessions together; repeated participant cases "
    "are deduplicated. No spatial coordinates or synthetic errors are generated."
)
CONSENT = (
    "The authors publicly released the pseudonymous transcription dataset under CC BY 4.0 on Zenodo. "
    "Only reference-verified prompt context is retained; the release does not include individual consent forms."
)


class IteDataError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def load_source(path=DEFAULT_MANIFEST):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 65536:
        raise IteDataError("source manifest is missing, linked, or too large")
    value = json.loads(path.read_bytes())
    if (not isinstance(value, dict) or set(value) != {"schemaVersion", "datasetId", "sourceUrl", "license",
            "licenseEvidenceUrl", "authors", "archiveUrl", "archiveBytes", "publisherArchiveMd5",
            "dictionaryAssetSha256", "nativeVocabularySha256", "files"}
            or value.get("schemaVersion") != 1
            or value.get("datasetId") != "aalto-ite-valid-words-en-v1"
            or value.get("license") != "CC-BY-4.0"
            or value.get("sourceUrl") != "https://zenodo.org/records/12528163"
            or value.get("licenseEvidenceUrl") != "https://zenodo.org/api/records/12528163"
            or value.get("archiveUrl") != "https://zenodo.org/records/12528163/files/ite_typing_dataset.zip?download=1"
            or value.get("archiveBytes") != 7309673578
            or value.get("publisherArchiveMd5") != "6bc033900d25a39801a22952c74f8027"
            or value.get("authors") != ["Katri Leino", "Markku Laine", "Mikko Kurimo", "Antti Oulasvirta"]):
        raise IteDataError("incompatible ITE source identity")
    for key in ("dictionaryAssetSha256", "nativeVocabularySha256"):
        if not isinstance(value.get(key), str) or not model_sources.SHA256.fullmatch(value[key]):
            raise IteDataError("invalid vocabulary provenance hash")
    expected = {"ITE_words/ac_words_en.csv", "processed2020/english/sentences.csv",
                "processed2020/english/test_sections_labeled.csv"}
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 3:
        raise IteDataError("missing selected source files")
    for item in files:
        if (not isinstance(item, dict) or set(item) != {"member", "file", "bytes", "sha256"}
                or not isinstance(item["member"], str) or item["member"] not in expected or item["file"] != pathlib.PurePosixPath(item["member"]).name
                or type(item["bytes"]) is not int or not 0 < item["bytes"] <= 128 * 1024 * 1024
                or not isinstance(item["sha256"], str) or not model_sources.SHA256.fullmatch(item["sha256"])):
            raise IteDataError("invalid selected source file")
        expected.remove(item["member"])
    return value


class RangeReader(io.RawIOBase):
    """Seekable, bounded HTTP ranges; never download the unselected multi-gigabyte log tables."""
    def __init__(self, url, length):
        self.url, self.length, self.position = url, length, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise IteDataError("invalid archive seek")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.length)
        if not 0 <= position <= self.length:
            raise IteDataError("archive seek is outside the pinned byte range")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if not 0 <= size <= 8 * 1024 * 1024:
            raise IteDataError("archive range exceeds its size bound")
        if size == 0:
            return b""
        start = self.position
        expected = f"bytes {start}-{start + size - 1}/{self.length}"
        for attempt in range(3):
            try:
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + size - 1}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise IteDataError("server did not return the exact pinned archive range")
                    payload = response.read(size)
                    if len(payload) != size:
                        raise IteDataError("truncated archive range")
                self.position += size
                return payload
            except (OSError, IteDataError):
                if attempt == 2:
                    raise


def verify_file(path, item):
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
            and model_sources.file_sha256(path) == item["sha256"])


def fetch_files(root, source):
    root.mkdir(parents=True, exist_ok=True)
    if all(verify_file(root / item["file"], item) for item in source["files"]):
        return
    with zipfile.ZipFile(RangeReader(source["archiveUrl"], source["archiveBytes"])) as archive:
        for item in source["files"]:
            target = root / item["file"]
            if verify_file(target, item):
                continue
            info = archive.getinfo(item["member"])
            if info.file_size != item["bytes"]:
                raise IteDataError("selected ZIP member size differs from the pin")
            descriptor, name = tempfile.mkstemp(dir=root, suffix=".download")
            staged = pathlib.Path(name)
            try:
                with os.fdopen(descriptor, "wb") as output, archive.open(info) as stream:
                    size = 0
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        if size > item["bytes"]:
                            raise IteDataError("selected ZIP member exceeds its byte bound")
                        output.write(chunk)
                if not verify_file(staged, item):
                    raise IteDataError("selected ZIP member hash differs from the pin")
                os.replace(staged, target)
            finally:
                staged.unlink(missing_ok=True)


def load_vocabulary(path, source):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
        raise IteDataError("native dictionary export is missing, linked, or oversized")
    value = json.loads(path.read_bytes())
    if (not isinstance(value, dict) or value.get("dictionarySha256") != source["dictionaryAssetSha256"]
            or not isinstance(value.get("words"), list)
            or hashlib.sha256(canonical(value["words"])).hexdigest() != source["nativeVocabularySha256"]):
        raise IteDataError("native dictionary export differs from the pinned vocabulary")
    return {word["word"].lower() for word in value["words"] if not word["possiblyOffensive"]}


def aligned_case(row, sentence, vocabulary):
    raw, target, current = row["TYPED_WORD"], row["ORIGINAL_WORD"], row["CURRENT_INPUT"]
    if (not re.fullmatch("[a-z]{2,64}", raw) or not re.fullmatch("[a-z]{2,64}", target)
            or raw not in vocabulary or target not in vocabulary):
        return None, "not_valid_word"
    if len(current) > 4096 or not current.lower().endswith(raw):
        return None, "raw_suffix"
    prefix = current[:-len(raw)]
    if not prefix.strip() or not prefix.endswith(" ") or not sentence.lower().startswith(prefix.lower()):
        return None, "prefix"
    match = re.match(r"([A-Za-z]+)(?=$|[^A-Za-z])", sentence[len(prefix):])
    if not match or match[1].lower() != target:
        return None, "reference_alignment"
    return (raw, target, prefix), None


def rows(path):
    csv.field_size_limit(16384)
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def prepare(source, source_root, vocabulary_path, output_root):
    for item in source["files"]:
        if not verify_file(source_root / item["file"], item):
            raise IteDataError("selected source file is missing or mismatched; use --fetch")
    vocabulary = load_vocabulary(vocabulary_path, source)
    sentences = {r["SENTENCE_ID"]: r["SENTENCE"] for r in rows(source_root / "sentences.csv")}
    sections = {r["TEST_SECTION_ID"]: (r["PARTICIPANT_ID"], r["SENTENCE_ID"])
                for r in rows(source_root / "test_sections_labeled.csv")}
    policy = tap.load_policy(tap.DEFAULT_POLICY)
    seen, records = set(), []
    rejected, counts = collections.Counter(), collections.Counter()
    for row in rows(source_root / "ac_words_en.csv"):
        section = sections.get(row["TEST_SECTION_ID"])
        if not section or section[1] != row["SENTENCE_ID"]:
            rejected["section"] += 1
            continue
        case, reason = aligned_case(row, sentences.get(row["SENTENCE_ID"], ""), vocabulary)
        if reason:
            rejected[reason] += 1
            continue
        raw, target, prefix = case
        identity = (section[0], prefix.lower(), raw, target)
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        session = "ite-en:" + section[0]
        correct = raw != target
        counts[(tap.split_for_session(session, policy), "correct" if correct else "keep")] += 1
        records.append({"schemaVersion": 1, "id": hashlib.sha256(canonical(identity)).hexdigest(),
                        "sessionId": session, "category": "valid_word", "raw": raw, "target": target,
                        "shouldCorrect": correct, "languageTag": "en-US", "fieldClass": "plain",
                        "precedingContext": prefix[-policy.maximum_context_codepoints:],
                        "collectionMethod": "human_replay", "touchPoints": []})
    records.sort(key=lambda row: row["id"])
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root) as temporary:
        staged = pathlib.Path(temporary)
        data = staged / "valid-words.jsonl"
        with data.open("wb") as stream:
            for record in records:
                stream.write(canonical(record) + b"\n")
        manifest = {"schemaVersion": 1, "datasetId": source["datasetId"], "dataFile": data.name,
                    "dataSha256": model_sources.file_sha256(data), "license": source["license"],
                    "containsHumanContributions": True, "collectionProtocol": PROTOCOL, "consentStatement": CONSENT}
        report = {"schemaVersion": 1, "records": len(records), "releaseEligible": False,
                  "canonicalSourceManifestSha256": hashlib.sha256(canonical(source)).hexdigest(),
                  "adapterSha256": model_sources.file_sha256(pathlib.Path(__file__)),
                  "nativeVocabularySha256": source["nativeVocabularySha256"],
                  "splits": {split: {label: counts[(split, label)] for label in ("correct", "keep")}
                             for split in tap.SPLITS}, "rejected": dict(sorted(rejected.items())),
                  "limitations": ["Selection is conditioned on publisher-detected autocorrection events.",
                                  "Publisher word-event labels are automatically inferred; strict reference alignment is required.",
                                  "No touch geometry is available; these rows are not spatial tap-error evidence.",
                                  "This English valid-word component cannot satisfy the complete Phase 0 gate."]}
        for name, value in (("source-manifest.json", manifest), ("preparation-report.json", report)):
            (staged / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        for name in (data.name, "source-manifest.json", "preparation-report.json"):
            os.replace(staged / name, output_root / name)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dictionary-lexicon", type=pathlib.Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = load_source(args.manifest)
        if args.fetch:
            fetch_files(args.source_root, source)
        report = prepare(source, args.source_root, args.dictionary_lexicon, args.output_root)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, KeyError, csv.Error, zipfile.BadZipFile) as failure:
        print(f"ITE preparation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
