#!/usr/bin/env python3
"""Prepare a deterministic, session-separated bilingual context sentence corpus."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import pathlib
import re
import sqlite3
import string
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator

import model_sources


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "context" / "data-policy.json"
DEFAULT_CORPUS_MANIFEST = ROOT / "models" / "context" / "corpus-manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-data" / "context-en-de-v1"
SPLITS = ("train", "validation", "test")
FIELD_CLASSES = {"plain", "short_message", "search"}
POLICY_KEYS = {
    "schemaVersion", "sourceId", "dataArtifacts", "projectAuthoredData", "splitSalt",
    "splitBasisPoints", "sentenceSampleBasisPoints", "maximumLineBytes",
    "maximumSentenceCodePoints", "maximumSentenceTokens", "minimumSentenceTokens",
    "maximumAuthoredExpansionsPerTemplate", "minimumAcceptedSentences",
    "minimumGermanSentences", "maximumRejectedFraction",
}
PROJECT_KEYS = {
    "schemaVersion", "id", "locale", "license", "slots", "templates",
    "confusionSets", "aliases",
}
TEMPLATE_KEYS = {"id", "fieldClass", "text", "slots", "sessionBuckets"}
PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAXIMUM_POLICY_BYTES = 256 * 1024
MAXIMUM_PROJECT_BYTES = 2 * 1024 * 1024


class ContextDataError(ValueError):
    pass


@dataclass(frozen=True)
class Policy:
    path: pathlib.Path
    sha256: str
    source_id: str
    data_artifacts: tuple[str, ...]
    project_authored_data: str
    split_salt: str
    split_basis_points: dict[str, int]
    sentence_sample_basis_points: int
    maximum_line_bytes: int
    maximum_sentence_codepoints: int
    maximum_sentence_tokens: int
    minimum_sentence_tokens: int
    maximum_authored_expansions_per_template: int
    minimum_accepted_sentences: int
    minimum_german_sentences: int
    maximum_rejected_fraction: float


@dataclass(frozen=True)
class ProjectCorpus:
    path: pathlib.Path
    sha256: str
    raw: dict[str, Any]


PREPARED_MANIFEST_KEYS = {
    "schemaVersion", "modelId", "minimumsEnforced", "sourceManifestSha256",
    "policySha256", "projectDataSha256", "toolSha256", "sources", "counts", "outputs",
}
PREPARED_OUTPUT_KEYS = {"bytes", "sha256", "records", "sessions", "languages"}


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: pathlib.Path, maximum_bytes: int, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= maximum_bytes:
            raise ContextDataError(f"{label} is missing, linked, empty, or too large")
        payload = path.read_bytes()
        value = json.loads(payload)
    except ContextDataError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextDataError(f"cannot read {label}: {failure}") from failure
    if not isinstance(value, dict):
        raise ContextDataError(f"{label} must be a JSON object")
    return value, payload


def _bounded_int(raw: dict[str, Any], field: str, minimum: int, maximum: int) -> int:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContextDataError(f"context data policy has invalid {field}")
    return value


def load_policy(path: pathlib.Path = DEFAULT_POLICY) -> Policy:
    path = path.resolve()
    raw, payload = _load_json(path, MAXIMUM_POLICY_BYTES, "context data policy")
    if set(raw) != POLICY_KEYS or raw.get("schemaVersion") != 1:
        raise ContextDataError("context data policy has an unexpected schema")
    source_id = raw["sourceId"]
    if not isinstance(source_id, str) or not model_sources.SOURCE_ID.fullmatch(source_id):
        raise ContextDataError("context data policy has an invalid sourceId")
    artifacts = raw["dataArtifacts"]
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or len(artifacts) != len(set(artifacts))
        or any(not isinstance(value, str) or not value for value in artifacts)
    ):
        raise ContextDataError("context data policy requires unique data artifacts")
    project_data = raw["projectAuthoredData"]
    if not isinstance(project_data, str) or pathlib.PurePosixPath(project_data).is_absolute():
        raise ContextDataError("context project-authored path must be repository-relative")
    if any(part in {"", ".", ".."} for part in pathlib.PurePosixPath(project_data).parts):
        raise ContextDataError("context project-authored path is not normalized")
    split_salt = raw["splitSalt"]
    if not isinstance(split_salt, str) or not 16 <= len(split_salt) <= 128:
        raise ContextDataError("context split salt must be a stable bounded string")
    split_points = raw["splitBasisPoints"]
    if not isinstance(split_points, dict) or set(split_points) != set(SPLITS):
        raise ContextDataError("context split policy must define train, validation, and test")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in split_points.values()):
        raise ContextDataError("context split sizes must be positive integers")
    if sum(split_points.values()) != 10_000:
        raise ContextDataError("context split basis points must total 10000")

    integers = {
        "sentenceSampleBasisPoints": _bounded_int(raw, "sentenceSampleBasisPoints", 1, 10_000),
        "maximumLineBytes": _bounded_int(raw, "maximumLineBytes", 1024, 16 * 1024 * 1024),
        "maximumSentenceCodePoints": _bounded_int(raw, "maximumSentenceCodePoints", 32, 4096),
        "maximumSentenceTokens": _bounded_int(raw, "maximumSentenceTokens", 4, 512),
        "minimumSentenceTokens": _bounded_int(raw, "minimumSentenceTokens", 1, 64),
        "maximumAuthoredExpansionsPerTemplate": _bounded_int(
            raw, "maximumAuthoredExpansionsPerTemplate", 1, 1_000_000,
        ),
        "minimumAcceptedSentences": _bounded_int(raw, "minimumAcceptedSentences", 1, 10_000_000),
        "minimumGermanSentences": _bounded_int(raw, "minimumGermanSentences", 1, 10_000_000),
    }
    if integers["minimumSentenceTokens"] > integers["maximumSentenceTokens"]:
        raise ContextDataError("context sentence token limits are inverted")
    rejected_fraction = raw["maximumRejectedFraction"]
    if (
        isinstance(rejected_fraction, bool)
        or not isinstance(rejected_fraction, (int, float))
        or not 0 <= float(rejected_fraction) < 1
    ):
        raise ContextDataError("context maximumRejectedFraction must be in [0, 1)")
    return Policy(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        source_id=source_id,
        data_artifacts=tuple(artifacts),
        project_authored_data=project_data,
        split_salt=split_salt,
        split_basis_points=dict(split_points),
        sentence_sample_basis_points=integers["sentenceSampleBasisPoints"],
        maximum_line_bytes=integers["maximumLineBytes"],
        maximum_sentence_codepoints=integers["maximumSentenceCodePoints"],
        maximum_sentence_tokens=integers["maximumSentenceTokens"],
        minimum_sentence_tokens=integers["minimumSentenceTokens"],
        maximum_authored_expansions_per_template=integers["maximumAuthoredExpansionsPerTemplate"],
        minimum_accepted_sentences=integers["minimumAcceptedSentences"],
        minimum_german_sentences=integers["minimumGermanSentences"],
        maximum_rejected_fraction=float(rejected_fraction),
    )


def _normalize_sentence(value: str, policy: Policy) -> tuple[str | None, str | None]:
    normalized = unicodedata.normalize("NFKC", value).replace("\u2019", "'")
    normalized = " ".join(normalized.split()).strip()
    if not normalized:
        return None, "empty_sentence"
    if len(normalized) > policy.maximum_sentence_codepoints:
        return None, "sentence_too_long"
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in normalized):
        return None, "unsafe_character"
    tokens = normalized.split(" ")
    if not policy.minimum_sentence_tokens <= len(tokens) <= policy.maximum_sentence_tokens:
        return None, "invalid_token_count"
    if not all(any(character.isalpha() for character in token) for token in tokens):
        return None, "nonlexical_token"
    return normalized, None


def session_hash(namespace: str, session: str, policy: Policy) -> str:
    return hashlib.sha256((policy.split_salt + "\0" + namespace + "\0" + session).encode()).hexdigest()


def split_for_hashed_session(hashed_session: str, policy: Policy) -> str:
    bucket = int(hashed_session[:8], 16) % 10_000
    cursor = 0
    for split in SPLITS:
        cursor += policy.split_basis_points[split]
        if bucket < cursor:
            return split
    raise AssertionError("context split basis points did not cover the hash space")


def split_for_session(namespace: str, session: str, policy: Policy) -> str:
    return split_for_hashed_session(session_hash(namespace, session, policy), policy)


def _sampled(source_revision: str, hashed_session: str, sentence: str, policy: Policy) -> bool:
    digest = hashlib.sha256((source_revision + "\0" + hashed_session + "\0" + sentence).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10_000 < policy.sentence_sample_basis_points


def _verified_json_lines(
    path: pathlib.Path,
    artifact: model_sources.Artifact,
    maximum_line_bytes: int,
) -> Iterator[tuple[int, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != artifact.bytes:
        raise ContextDataError(f"context source is missing or has the wrong size: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(maximum_line_bytes + 1)
            if not line:
                break
            line_number += 1
            digest.update(line)
            if len(line) > maximum_line_bytes:
                raise ContextDataError(f"{artifact.path} line {line_number} exceeds the size limit")
            try:
                yield line_number, json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise ContextDataError(f"{artifact.path} line {line_number} is invalid JSON: {failure}") from failure
    if digest.hexdigest() != artifact.sha256:
        raise ContextDataError(f"context source has wrong SHA-256: {path}")


def load_project_corpus(path: pathlib.Path, policy: Policy) -> ProjectCorpus:
    path = path.resolve()
    raw, payload = _load_json(path, MAXIMUM_PROJECT_BYTES, "project-authored context data")
    if set(raw) != PROJECT_KEYS or raw.get("schemaVersion") != 1:
        raise ContextDataError("project-authored context data has an unexpected schema")
    if (
        not isinstance(raw["id"], str)
        or not PROJECT_ID.fullmatch(raw["id"])
        or raw["locale"] != "de"
        or raw["license"] != "Apache-2.0"
    ):
        raise ContextDataError("project-authored context identity, locale, or license is invalid")
    slots = raw["slots"]
    if not isinstance(slots, dict) or not slots or len(slots) > 128:
        raise ContextDataError("project-authored context slots are invalid")
    for name, values in slots.items():
        if not isinstance(name, str) or not PROJECT_ID.fullmatch(name.replace("_", "-")):
            raise ContextDataError("project-authored context slot name is invalid")
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 256
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value.strip() or len(value) > 512 for value in values)
        ):
            raise ContextDataError(f"project-authored context slot {name} is invalid")
    templates = raw["templates"]
    if not isinstance(templates, list) or not templates or len(templates) > 256:
        raise ContextDataError("project-authored context templates are invalid")
    seen_ids = set()
    formatter = string.Formatter()
    for index, template in enumerate(templates):
        if not isinstance(template, dict) or set(template) != TEMPLATE_KEYS:
            raise ContextDataError(f"project-authored template {index + 1} has an unexpected schema")
        identifier = template["id"]
        if not isinstance(identifier, str) or not PROJECT_ID.fullmatch(identifier) or identifier in seen_ids:
            raise ContextDataError("project-authored template id is invalid or duplicated")
        seen_ids.add(identifier)
        if template["fieldClass"] not in FIELD_CLASSES:
            raise ContextDataError(f"project-authored template {identifier} has an invalid field class")
        text = template["text"]
        template_slots = template["slots"]
        if not isinstance(text, str) or not text or len(text) > 2048:
            raise ContextDataError(f"project-authored template {identifier} has invalid text")
        if (
            not isinstance(template_slots, list)
            or len(template_slots) != len(set(template_slots))
            or any(name not in slots for name in template_slots)
        ):
            raise ContextDataError(f"project-authored template {identifier} has invalid slots")
        parsed_slots = [name for _literal, name, format_spec, conversion in formatter.parse(text) if name is not None]
        if any(format_spec or conversion for _literal, name, format_spec, conversion in formatter.parse(text) if name is not None):
            raise ContextDataError(f"project-authored template {identifier} uses formatting operators")
        if set(parsed_slots) != set(template_slots):
            raise ContextDataError(f"project-authored template {identifier} placeholders do not match slots")
        buckets = template["sessionBuckets"]
        if isinstance(buckets, bool) or not isinstance(buckets, int) or not 1 <= buckets <= 4096:
            raise ContextDataError(f"project-authored template {identifier} has invalid session buckets")
        expansions = 1
        for name in template_slots:
            expansions *= len(slots[name])
        if expansions > policy.maximum_authored_expansions_per_template:
            raise ContextDataError(f"project-authored template {identifier} exceeds its expansion ceiling")

    confusion_sets = raw["confusionSets"]
    if (
        not isinstance(confusion_sets, list)
        or len(confusion_sets) > 256
        or any(
            not isinstance(values, list)
            or len(values) < 2
            or len(values) > 16
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
            for values in confusion_sets
        )
    ):
        raise ContextDataError("project-authored confusion sets are invalid")
    aliases = raw["aliases"]
    if (
        not isinstance(aliases, dict)
        or len(aliases) > 256
        or any(
            not isinstance(source, str) or not source
            or not isinstance(target, str) or not target
            for source, target in aliases.items()
        )
    ):
        raise ContextDataError("project-authored aliases are invalid")
    return ProjectCorpus(path=path, sha256=hashlib.sha256(payload).hexdigest(), raw=raw)


def _record(
    *,
    identifier_seed: str,
    hashed_session: str,
    split: str,
    language: str,
    field_class: str,
    text: str,
    source_id: str,
) -> dict[str, Any]:
    identifier = hashlib.sha256((identifier_seed + "\0" + hashed_session + "\0" + text).encode()).hexdigest()
    return {
        "schemaVersion": 1,
        "id": identifier,
        "sessionId": hashed_session,
        "split": split,
        "language": language,
        "fieldClass": field_class,
        "text": text,
        "sourceId": source_id,
    }


def _write_record(
    handles: dict[str, BinaryIO],
    database: sqlite3.Connection,
    counters: collections.Counter[str],
    record: dict[str, Any],
) -> bool:
    fingerprint = hashlib.sha256(_canonical_line({
        "sessionId": record["sessionId"],
        "language": record["language"],
        "text": record["text"],
    })).hexdigest()
    inserted = database.execute(
        "INSERT OR IGNORE INTO fingerprints(fingerprint) VALUES (?)",
        (fingerprint,),
    ).rowcount
    if inserted == 0:
        counters["duplicateSentences"] += 1
        return False
    database.execute(
        "INSERT OR IGNORE INTO sessions(session_id, split) VALUES (?, ?)",
        (record["sessionId"], record["split"]),
    )
    handles[record["split"]].write(_canonical_line(record))
    counters["acceptedSentences"] += 1
    counters[f"accepted:{record['split']}"] += 1
    counters[f"language:{record['language']}"] += 1
    counters[f"language:{record['split']}:{record['language']}"] += 1
    return True


def _source_sentences(
    *,
    source_root: pathlib.Path,
    source: model_sources.Source,
    policy: Policy,
    handles: dict[str, BinaryIO],
    database: sqlite3.Connection,
    counters: collections.Counter[str],
) -> None:
    for artifact_name in policy.data_artifacts:
        artifact = source.artifact(artifact_name)
        path = model_sources.artifact_path(source_root, source, artifact)
        for line_number, raw in _verified_json_lines(path, artifact, policy.maximum_line_bytes):
            counters["sourceRows"] += 1
            if not isinstance(raw, dict):
                counters["invalidSourceRows"] += 1
                counters["rejection:not_object"] += 1
                continue
            session = raw.get("session")
            sentence = raw.get("sentence")
            invalid_sentence = raw.get("potentially_invalid_sentence")
            if not isinstance(session, str) or not session or len(session) > 256:
                counters["invalidSourceRows"] += 1
                counters["rejection:invalid_session"] += 1
                continue
            if not isinstance(sentence, str) or not isinstance(invalid_sentence, bool) or invalid_sentence:
                counters["invalidSourceRows"] += 1
                counters["rejection:invalid_sentence_flag"] += 1
                continue
            normalized, rejection = _normalize_sentence(sentence, policy)
            if normalized is None:
                counters["invalidSourceRows"] += 1
                counters[f"rejection:{rejection}"] += 1
                continue
            hashed_session = session_hash(source.identifier, session, policy)
            if not _sampled(source.revision, hashed_session, normalized, policy):
                counters["sampledOutSourceRows"] += 1
                continue
            split = split_for_hashed_session(hashed_session, policy)
            accepted = _write_record(
                handles,
                database,
                counters,
                _record(
                    identifier_seed=f"{source.revision}\0{artifact.path}\0{line_number}",
                    hashed_session=hashed_session,
                    split=split,
                    language="en-US",
                    field_class="plain",
                    text=normalized,
                    source_id=source.identifier,
                ),
            )
            if accepted:
                counters["acceptedSourceSentences"] += 1


def _project_sentences(
    *,
    project: ProjectCorpus,
    policy: Policy,
    handles: dict[str, BinaryIO],
    database: sqlite3.Connection,
    counters: collections.Counter[str],
) -> None:
    slots = project.raw["slots"]
    for template in project.raw["templates"]:
        names = template["slots"]
        combinations = itertools.product(*(slots[name] for name in names)) if names else [()]
        for index, values in enumerate(combinations):
            substitutions = dict(zip(names, values, strict=True))
            rendered = template["text"].format_map(substitutions)
            normalized, rejection = _normalize_sentence(rendered, policy)
            if normalized is None:
                raise ContextDataError(
                    f"project-authored template {template['id']} produced invalid text: {rejection}"
                )
            bucket_digest = hashlib.sha256((template["id"] + "\0" + normalized).encode()).digest()
            bucket = int.from_bytes(bucket_digest[:4], "big") % template["sessionBuckets"]
            raw_session = f"{template['id']}:{bucket}"
            hashed_session = session_hash(project.raw["id"], raw_session, policy)
            split = split_for_hashed_session(hashed_session, policy)
            accepted = _write_record(
                handles,
                database,
                counters,
                _record(
                    identifier_seed=f"{project.sha256}\0{template['id']}\0{index}",
                    hashed_session=hashed_session,
                    split=split,
                    language="de",
                    field_class=template["fieldClass"],
                    text=normalized,
                    source_id=project.raw["id"],
                ),
            )
            if accepted:
                counters["acceptedProjectSentences"] += 1


def _output_metadata(path: pathlib.Path) -> dict[str, Any]:
    records = 0
    sessions = set()
    languages: collections.Counter[str] = collections.Counter()
    with path.open("rb") as stream:
        for line in stream:
            value = json.loads(line)
            records += 1
            sessions.add(value["sessionId"])
            languages[value["language"]] += 1
    return {
        "bytes": path.stat().st_size,
        "sha256": model_sources.file_sha256(path),
        "records": records,
        "sessions": len(sessions),
        "languages": dict(sorted(languages.items())),
    }


def load_prepared_manifest(
    data_root: pathlib.Path,
    *,
    require_pinned: bool = False,
    pinned_path: pathlib.Path = DEFAULT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    manifest_path = data_root / "split-manifest.json"
    try:
        if (
            not manifest_path.is_file()
            or manifest_path.is_symlink()
            or not 0 < manifest_path.stat().st_size <= 1024 * 1024
        ):
            raise ContextDataError("prepared context manifest is missing, linked, empty, or too large")
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except ContextDataError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextDataError(f"cannot read prepared context manifest: {failure}") from failure
    if not isinstance(manifest, dict) or set(manifest) != PREPARED_MANIFEST_KEYS:
        raise ContextDataError("prepared context manifest has an unexpected schema")
    if (
        manifest["schemaVersion"] != 1
        or manifest["modelId"] != "context-en-de-v1"
        or not isinstance(manifest["minimumsEnforced"], bool)
    ):
        raise ContextDataError("prepared context manifest has an incompatible identity")
    outputs = manifest["outputs"]
    expected_outputs = {f"{split}.sentences.jsonl" for split in SPLITS}
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ContextDataError("prepared context outputs are incomplete or unexpected")
    for filename in sorted(expected_outputs):
        details = outputs[filename]
        if not isinstance(details, dict) or set(details) != PREPARED_OUTPUT_KEYS:
            raise ContextDataError(f"prepared context output metadata is invalid: {filename}")
        path = data_root / filename
        if not path.is_file() or path.is_symlink() or path.stat().st_size != details["bytes"]:
            raise ContextDataError(f"prepared context output is missing or has the wrong size: {filename}")
        digest = details["sha256"]
        if not isinstance(digest, str) or not model_sources.SHA256.fullmatch(digest):
            raise ContextDataError(f"prepared context output has an invalid hash: {filename}")
        if model_sources.file_sha256(path) != digest:
            raise ContextDataError(f"prepared context output hash does not match: {filename}")
        for field in ("records", "sessions"):
            value = details[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContextDataError(f"prepared context output has an invalid {field} count: {filename}")
        languages = details["languages"]
        expected_languages = {"en-US", "de"}
        if (
            not isinstance(languages, dict)
            or not languages
            or not set(languages).issubset(expected_languages)
            or (manifest["minimumsEnforced"] and set(languages) != expected_languages)
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in languages.values())
            or sum(languages.values()) != details["records"]
        ):
            raise ContextDataError(f"prepared context output has invalid language counts: {filename}")
    if require_pinned:
        try:
            if (
                not pinned_path.is_file()
                or pinned_path.is_symlink()
                or not 0 < pinned_path.stat().st_size <= 1024 * 1024
            ):
                raise ContextDataError("pinned context corpus manifest is missing, linked, empty, or too large")
            pinned_payload = pinned_path.read_bytes()
            json.loads(pinned_payload)
        except ContextDataError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
            raise ContextDataError(f"cannot read pinned context corpus manifest: {failure}") from failure
        if payload != pinned_payload:
            raise ContextDataError("prepared context corpus does not match the pinned manifest")
    return manifest


def prepare(
    *,
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    manifest_path: pathlib.Path = model_sources.DEFAULT_MANIFEST,
    policy_path: pathlib.Path = DEFAULT_POLICY,
    project_path: pathlib.Path | None = None,
    enforce_minimums: bool = True,
) -> dict[str, Any]:
    manifest = model_sources.load_manifest(manifest_path)
    policy = load_policy(policy_path)
    source = manifest.source(policy.source_id)
    if source.kind != "dataset" or source.license != "MIT":
        raise ContextDataError("context sentence source must be the pinned MIT dataset")
    for artifact_name in policy.data_artifacts:
        source.artifact(artifact_name)
    project = load_project_corpus(
        project_path if project_path is not None else ROOT / policy.project_authored_data,
        policy,
    )
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    counters: collections.Counter[str] = collections.Counter()

    with tempfile.TemporaryDirectory(prefix="libreboard-context-", dir=output_root.parent) as temporary:
        staging = pathlib.Path(temporary)
        paths = {split: staging / f"{split}.sentences.jsonl" for split in SPLITS}
        handles = {split: path.open("wb") for split, path in paths.items()}
        database = sqlite3.connect(staging / "dedup.sqlite3")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE fingerprints (fingerprint TEXT PRIMARY KEY)")
        database.execute(
            "CREATE TABLE sessions (session_id TEXT NOT NULL, split TEXT NOT NULL, PRIMARY KEY(session_id, split))"
        )
        try:
            try:
                _source_sentences(
                    source_root=source_root,
                    source=source,
                    policy=policy,
                    handles=handles,
                    database=database,
                    counters=counters,
                )
                _project_sentences(
                    project=project,
                    policy=policy,
                    handles=handles,
                    database=database,
                    counters=counters,
                )
            finally:
                for handle in handles.values():
                    if not handle.closed:
                        handle.flush()
                        os.fsync(handle.fileno())
                        handle.close()
            overlap = database.execute(
                "SELECT session_id FROM sessions GROUP BY session_id "
                "HAVING COUNT(DISTINCT split) > 1 LIMIT 1"
            ).fetchone()
        finally:
            database.close()
        if overlap is not None:
            raise ContextDataError("context session leaked across data splits")
        if counters["sourceRows"] == 0:
            raise ContextDataError("context source contained no rows")
        invalid_fraction = counters["invalidSourceRows"] / counters["sourceRows"]
        if invalid_fraction > policy.maximum_rejected_fraction:
            raise ContextDataError(
                f"context source rejection fraction {invalid_fraction:.6f} exceeds policy"
            )
        if enforce_minimums and counters["acceptedSentences"] < policy.minimum_accepted_sentences:
            raise ContextDataError(
                f"context corpus has too few accepted sentences: "
                f"{counters['acceptedSentences']} < {policy.minimum_accepted_sentences}"
            )
        if enforce_minimums and counters["language:de"] < policy.minimum_german_sentences:
            raise ContextDataError(
                f"context corpus has too few German sentences: "
                f"{counters['language:de']} < {policy.minimum_german_sentences}"
            )
        outputs = {path.name: _output_metadata(path) for path in paths.values()}
        if enforce_minimums and any(outputs[f"{split}.sentences.jsonl"]["records"] == 0 for split in SPLITS):
            raise ContextDataError("context corpus has an empty split")
        report = {
            "schemaVersion": 1,
            "modelId": "context-en-de-v1",
            "minimumsEnforced": enforce_minimums,
            "sourceManifestSha256": manifest.sha256,
            "policySha256": policy.sha256,
            "projectDataSha256": project.sha256,
            "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
            "sources": {
                "external": {
                    "id": source.identifier,
                    "revision": source.revision,
                    "license": source.license,
                    "sourceUrl": source.source_url,
                },
                "projectAuthored": {
                    "id": project.raw["id"],
                    "path": policy.project_authored_data,
                    "sha256": project.sha256,
                    "license": project.raw["license"],
                },
            },
            "counts": {
                **dict(sorted(counters.items())),
                "invalidSourceFraction": round(invalid_fraction, 8),
            },
            "outputs": outputs,
        }
        manifest_output = staging / "split-manifest.json"
        manifest_output.write_bytes(_canonical_line(report))
        with manifest_output.open("rb") as stream:
            os.fsync(stream.fileno())
        output_root.mkdir(parents=True, exist_ok=True)
        for path in (*paths.values(), manifest_output):
            os.replace(path, output_root / path.name)
        return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=model_sources.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=pathlib.Path, default=model_sources.DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--allow-small-corpus",
        action="store_true",
        help="development only: publish measured outputs without the release-size minimums",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = prepare(
            source_root=args.source_root,
            output_root=args.output_root,
            manifest_path=args.manifest,
            policy_path=args.policy,
            enforce_minimums=not args.allow_small_corpus,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ContextDataError, model_sources.ModelSourceError) as failure:
        print(f"context data error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
