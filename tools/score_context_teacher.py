#!/usr/bin/env python3
"""Build deterministic candidate slates and score them with the pinned Hanse2 teacher."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterator

import context_tokenizer_contract
import model_sources
import prepare_context_dataset


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "context" / "distillation-policy.json"
DEFAULT_DATA_ROOT = ROOT / "build" / "model-data" / "context-en-de-v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "teacher-scored-v1"
DEFAULT_SOURCE_ROOT = model_sources.DEFAULT_SOURCE_ROOT
INPUT_KEYS = {
    "schemaVersion", "id", "sessionId", "split", "language", "fieldClass", "text", "sourceId",
}
POLICY_KEYS = {
    "schemaVersion", "modelId", "teacherSourceId", "tokenizerSha256",
    "maximumPrefixTeacherTokens", "maximumCandidateTeacherTokens",
    "maximumPrefixStudentTokens", "maximumCandidateStudentTokens",
    "minimumCandidates", "maximumCandidates", "inferenceBatchRows", "chunkRecords",
    "lexiconWordsPerLanguage", "maximumWordCodePoints", "phraseLengthBasisPoints",
    "minimumScoredRecords", "minimumGermanRecords", "minimumHeldOutRecords",
    "minimumHeldOutEnglishRecords", "minimumHeldOutGermanRecords",
}
OUTPUT_KEYS = {
    "schemaVersion", "id", "split", "language", "fieldClassId", "sourceId",
    "prefixTokenIds", "candidates", "observedCandidateIndex",
}
CANDIDATE_KEYS = {"surface", "normalized", "sources", "tokenIds", "teacherMeanLogProbability"}
FIELD_CLASS_IDS = {"plain": 0, "short_message": 1, "search": 2}
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*", re.UNICODE)
MAXIMUM_POLICY_BYTES = 256 * 1024
MAXIMUM_PROGRESS_BYTES = 4 * 1024 * 1024
TEACHER_PARAMETER_COUNT = 99_144_320
KEYBOARD_NEIGHBOR = {
    "q": "w", "w": "e", "e": "r", "r": "t", "t": "z", "z": "u", "u": "i",
    "i": "o", "o": "p", "p": "o", "a": "s", "s": "d", "d": "f", "f": "g",
    "g": "h", "h": "j", "j": "k", "k": "l", "l": "k", "y": "x", "x": "c",
    "c": "v", "v": "b", "b": "n", "n": "m", "m": "n", "ä": "ö", "ö": "ä",
    "ü": "u", "ß": "s",
}


class ContextTeacherError(ValueError):
    pass


@dataclass(frozen=True)
class DistillationPolicy:
    path: pathlib.Path
    sha256: str
    teacher_source_id: str
    tokenizer_sha256: str
    maximum_prefix_teacher_tokens: int
    maximum_candidate_teacher_tokens: int
    maximum_prefix_student_tokens: int
    maximum_candidate_student_tokens: int
    minimum_candidates: int
    maximum_candidates: int
    inference_batch_rows: int
    chunk_records: int
    lexicon_words_per_language: int
    maximum_word_codepoints: int
    phrase_length_basis_points: dict[int, int]
    minimum_scored_records: int
    minimum_german_records: int
    minimum_held_out_records: int
    minimum_held_out_english_records: int
    minimum_held_out_german_records: int


@dataclass(frozen=True)
class PreparedCandidate:
    surface: str
    normalized: str
    sources: tuple[str, ...]
    student_token_ids: tuple[int, ...]
    teacher_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedExample:
    identifier: str
    split: str
    language: str
    field_class_id: int
    source_id: str
    prefix_student_token_ids: tuple[int, ...]
    prefix_teacher_token_ids: tuple[int, ...]
    candidates: tuple[PreparedCandidate, ...]

    def binding_document(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.identifier,
            "split": self.split,
            "language": self.language,
            "fieldClassId": self.field_class_id,
            "sourceId": self.source_id,
            "prefixTokenIds": list(self.prefix_student_token_ids),
            "teacherPrefixTokenIds": list(self.prefix_teacher_token_ids),
            "candidates": [{
                "surface": candidate.surface,
                "normalized": candidate.normalized,
                "sources": list(candidate.sources),
                "tokenIds": list(candidate.student_token_ids),
                "teacherTokenIds": list(candidate.teacher_token_ids),
            } for candidate in self.candidates],
            "observedCandidateIndex": 0,
        }

    def output_document(self, scores: list[float]) -> dict[str, Any]:
        if len(scores) != len(self.candidates):
            raise ContextTeacherError("teacher score count does not match the candidate slate")
        return {
            "schemaVersion": 1,
            "id": self.identifier,
            "split": self.split,
            "language": self.language,
            "fieldClassId": self.field_class_id,
            "sourceId": self.source_id,
            "prefixTokenIds": list(self.prefix_student_token_ids),
            "candidates": [{
                "surface": candidate.surface,
                "normalized": candidate.normalized,
                "sources": list(candidate.sources),
                "tokenIds": list(candidate.student_token_ids),
                "teacherMeanLogProbability": score,
            } for candidate, score in zip(self.candidates, scores, strict=True)],
            "observedCandidateIndex": 0,
        }


def _canonical_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _load_json(path: pathlib.Path, maximum_bytes: int, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= maximum_bytes:
            raise ContextTeacherError(f"{label} is missing, linked, empty, or too large")
        payload = path.read_bytes()
        value = json.loads(payload)
    except ContextTeacherError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTeacherError(f"cannot read {label}: {failure}") from failure
    if not isinstance(value, dict):
        raise ContextTeacherError(f"{label} must be a JSON object")
    return value, payload


def _bounded_int(raw: dict[str, Any], name: str, minimum: int, maximum: int) -> int:
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContextTeacherError(f"context distillation policy has invalid {name}")
    return value


def load_policy(path: pathlib.Path = DEFAULT_POLICY) -> DistillationPolicy:
    path = path.resolve()
    raw, payload = _load_json(path, MAXIMUM_POLICY_BYTES, "context distillation policy")
    if set(raw) != POLICY_KEYS or raw.get("schemaVersion") != 1 or raw.get("modelId") != "context-en-de-v1":
        raise ContextTeacherError("context distillation policy has an unexpected schema or identity")
    teacher_source_id = raw["teacherSourceId"]
    tokenizer_sha256 = raw["tokenizerSha256"]
    if not isinstance(teacher_source_id, str) or not model_sources.SOURCE_ID.fullmatch(teacher_source_id):
        raise ContextTeacherError("context distillation policy has an invalid teacher source")
    if not isinstance(tokenizer_sha256, str) or not model_sources.SHA256.fullmatch(tokenizer_sha256):
        raise ContextTeacherError("context distillation policy has an invalid tokenizer hash")
    values = {
        "maximumPrefixTeacherTokens": _bounded_int(raw, "maximumPrefixTeacherTokens", 1, 256),
        "maximumCandidateTeacherTokens": _bounded_int(raw, "maximumCandidateTeacherTokens", 1, 64),
        "maximumPrefixStudentTokens": _bounded_int(raw, "maximumPrefixStudentTokens", 1, 22),
        "maximumCandidateStudentTokens": _bounded_int(raw, "maximumCandidateStudentTokens", 1, 8),
        "minimumCandidates": _bounded_int(raw, "minimumCandidates", 2, 32),
        "maximumCandidates": _bounded_int(raw, "maximumCandidates", 2, 32),
        "inferenceBatchRows": _bounded_int(raw, "inferenceBatchRows", 1, 128),
        "chunkRecords": _bounded_int(raw, "chunkRecords", 1, 4096),
        "lexiconWordsPerLanguage": _bounded_int(raw, "lexiconWordsPerLanguage", 64, 100_000),
        "maximumWordCodePoints": _bounded_int(raw, "maximumWordCodePoints", 8, 256),
        "minimumScoredRecords": _bounded_int(raw, "minimumScoredRecords", 1, 10_000_000),
        "minimumGermanRecords": _bounded_int(raw, "minimumGermanRecords", 1, 10_000_000),
        "minimumHeldOutRecords": _bounded_int(raw, "minimumHeldOutRecords", 1, 1_000_000),
        "minimumHeldOutEnglishRecords": _bounded_int(raw, "minimumHeldOutEnglishRecords", 1, 1_000_000),
        "minimumHeldOutGermanRecords": _bounded_int(raw, "minimumHeldOutGermanRecords", 1, 1_000_000),
    }
    if values["minimumCandidates"] > values["maximumCandidates"]:
        raise ContextTeacherError("context distillation candidate limits are inverted")
    if values["inferenceBatchRows"] < values["maximumCandidates"]:
        raise ContextTeacherError("context distillation batch cannot hold one complete candidate slate")
    phrase_points = raw["phraseLengthBasisPoints"]
    if (
        not isinstance(phrase_points, dict)
        or set(phrase_points) != {"1", "2", "3"}
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in phrase_points.values())
        or sum(phrase_points.values()) != 10_000
    ):
        raise ContextTeacherError("context phrase-length basis points must define 1-3 words and total 10000")
    return DistillationPolicy(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        teacher_source_id=teacher_source_id,
        tokenizer_sha256=tokenizer_sha256,
        maximum_prefix_teacher_tokens=values["maximumPrefixTeacherTokens"],
        maximum_candidate_teacher_tokens=values["maximumCandidateTeacherTokens"],
        maximum_prefix_student_tokens=values["maximumPrefixStudentTokens"],
        maximum_candidate_student_tokens=values["maximumCandidateStudentTokens"],
        minimum_candidates=values["minimumCandidates"],
        maximum_candidates=values["maximumCandidates"],
        inference_batch_rows=values["inferenceBatchRows"],
        chunk_records=values["chunkRecords"],
        lexicon_words_per_language=values["lexiconWordsPerLanguage"],
        maximum_word_codepoints=values["maximumWordCodePoints"],
        phrase_length_basis_points={int(key): value for key, value in phrase_points.items()},
        minimum_scored_records=values["minimumScoredRecords"],
        minimum_german_records=values["minimumGermanRecords"],
        minimum_held_out_records=values["minimumHeldOutRecords"],
        minimum_held_out_english_records=values["minimumHeldOutEnglishRecords"],
        minimum_held_out_german_records=values["minimumHeldOutGermanRecords"],
    )


def _dependencies():
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as failure:
        raise ContextTeacherError(
            "install the pinned dependencies from models/training/requirements-context-linux-x86_64.lock"
        ) from failure
    versions = {
        "torch": str(torch.__version__).split("+", 1)[0],
        "transformers": transformers.__version__,
    }
    if versions != {"torch": "2.8.0", "transformers": "4.57.6"}:
        raise ContextTeacherError(f"context teacher dependency versions do not match the lock: {versions}")
    return torch, AutoModelForCausalLM, AutoTokenizer, versions


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    if not model_sources.REVISION.fullmatch(commit):
        raise ContextTeacherError("context teacher scoring requires a full Git commit")
    return commit, dirty


def _input_record(line: bytes, split: str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTeacherError(f"prepared {split} context data contains invalid JSON: {failure}") from failure
    if not isinstance(value, dict) or set(value) != INPUT_KEYS:
        raise ContextTeacherError(f"prepared {split} context record has an unexpected schema")
    if value["schemaVersion"] != 1 or value["split"] != split:
        raise ContextTeacherError(f"prepared {split} context record has an incompatible identity")
    if (
        not isinstance(value["id"], str)
        or not model_sources.SHA256.fullmatch(value["id"])
        or not isinstance(value["sessionId"], str)
        or not model_sources.SHA256.fullmatch(value["sessionId"])
        or value["language"] not in {"en-US", "de"}
        or value["fieldClass"] not in FIELD_CLASS_IDS
        or not isinstance(value["text"], str)
        or not value["text"]
        or not isinstance(value["sourceId"], str)
        or not model_sources.SOURCE_ID.fullmatch(value["sourceId"])
    ):
        raise ContextTeacherError(f"prepared {split} context record has invalid fields")
    return value


def _iter_records(path: pathlib.Path, split: str, expected: int) -> Iterator[dict[str, Any]]:
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            count += 1
            yield _input_record(line, split)
    if count != expected:
        raise ContextTeacherError(f"prepared {split} context count changed while scoring")


def _normalize_candidate(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().replace("\u2019", "'").split())


def _build_lexicons(
    data_root: pathlib.Path,
    manifest: dict[str, Any],
    policy: DistillationPolicy,
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for split in prepare_context_dataset.SPLITS:
        name = f"{split}.sentences.jsonl"
        counters = {"en-US": collections.Counter(), "de": collections.Counter()}
        for record in _iter_records(data_root / name, split, manifest["outputs"][name]["records"]):
            for match in WORD_PATTERN.finditer(record["text"]):
                word = _normalize_candidate(match.group())
                if 0 < len(word) <= policy.maximum_word_codepoints:
                    counters[record["language"]][word] += 1
        result[split] = {}
        for language, counter in counters.items():
            words = [
                word for word, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            ][:policy.lexicon_words_per_language]
            if len(words) < policy.minimum_candidates:
                raise ContextTeacherError(f"prepared {split} {language} lexicon is too small")
            result[split][language] = words
    return result


def _desired_phrase_length(identifier: str, policy: DistillationPolicy) -> int:
    bucket = int(hashlib.sha256((identifier + "\0phrase-length").encode()).hexdigest()[:8], 16) % 10_000
    cursor = 0
    for length in (1, 2, 3):
        cursor += policy.phrase_length_basis_points[length]
        if bucket < cursor:
            return length
    raise AssertionError("phrase-length basis points do not cover the hash space")


def _edit_variants(word: str, identifier: str) -> list[tuple[str, str]]:
    characters = list(word)
    if not characters:
        return []
    seed = int(hashlib.sha256((identifier + "\0edits").encode()).hexdigest()[:16], 16)
    variants: list[tuple[str, str]] = []
    if len(characters) >= 4:
        index = seed % len(characters)
        variants.append(("".join(characters[:index] + characters[index + 1:]), "delete"))
        indexes = [index for index in range(len(characters) - 1) if characters[index] != characters[index + 1]]
        if indexes:
            index = indexes[(seed // 7) % len(indexes)]
            swapped = characters.copy()
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            variants.append(("".join(swapped), "transpose"))
    neighbor_indexes = [index for index, character in enumerate(characters) if character.lower() in KEYBOARD_NEIGHBOR]
    if neighbor_indexes:
        index = neighbor_indexes[(seed // 13) % len(neighbor_indexes)]
        replacement = KEYBOARD_NEIGHBOR[characters[index].lower()]
        if characters[index].isupper():
            replacement = replacement.upper()
        changed = characters.copy()
        changed[index] = replacement
        variants.append(("".join(changed), "neighbor"))
    compact = word.replace("'", "").replace("\u2019", "").replace("-", "")
    if compact != word:
        variants.append((compact, "punctuation-drop"))
    return variants


def _prepare_example(
    record: dict[str, Any],
    *,
    lexicon: list[str],
    confusion_map: dict[str, tuple[str, ...]],
    aliases: dict[str, str],
    student_tokenizer: context_tokenizer_contract.ContextTokenizer,
    teacher_tokenizer,
    policy: DistillationPolicy,
) -> tuple[PreparedExample | None, str | None]:
    text = record["text"]
    matches = list(WORD_PATTERN.finditer(text))
    if len(matches) < 2:
        return None, "insufficient_context"
    desired_length = _desired_phrase_length(record["id"], policy)
    starts = list(range(1, len(matches)))
    offset = int(record["id"][:8], 16) % len(starts)
    starts = starts[offset:] + starts[:offset]

    for start in starts:
        for length in tuple(range(desired_length, 0, -1)):
            end = start + length - 1
            if end >= len(matches):
                continue
            if any(not text[matches[index].end():matches[index + 1].start()].isspace() for index in range(start, end)):
                continue
            prefix = text[:matches[start].start()].rstrip()
            observed = text[matches[start].start():matches[end].end()]
            if not prefix or len(observed) > policy.maximum_word_codepoints * length:
                continue
            prefix_student_ids = tuple(context_tokenizer_contract.encode_text(student_tokenizer, prefix))
            prefix_student_ids = prefix_student_ids[-policy.maximum_prefix_student_tokens:]
            if not prefix_student_ids:
                continue
            prefix_teacher_ids = tuple(teacher_tokenizer.encode(prefix, add_special_tokens=False))
            prefix_teacher_ids = prefix_teacher_ids[-policy.maximum_prefix_teacher_tokens:]
            if not prefix_teacher_ids:
                continue

            candidates: list[PreparedCandidate] = []
            candidate_indexes: dict[str, int] = {}

            def add_candidate(surface: str, source: str) -> bool:
                surface = " ".join(surface.split())
                normalized = _normalize_candidate(surface)
                if not surface or not normalized or len(surface) > policy.maximum_word_codepoints * 3:
                    return False
                existing = candidate_indexes.get(normalized)
                if existing is not None:
                    prior = candidates[existing]
                    candidates[existing] = PreparedCandidate(
                        surface=prior.surface,
                        normalized=prior.normalized,
                        sources=tuple(sorted(set(prior.sources) | {source})),
                        student_token_ids=prior.student_token_ids,
                        teacher_token_ids=prior.teacher_token_ids,
                    )
                    return True
                student_ids = tuple(context_tokenizer_contract.encode_text(student_tokenizer, surface))
                teacher_ids = tuple(teacher_tokenizer.encode(" " + surface, add_special_tokens=False))
                if (
                    not student_ids
                    or len(student_ids) > policy.maximum_candidate_student_tokens
                    or not teacher_ids
                    or len(teacher_ids) > policy.maximum_candidate_teacher_tokens
                ):
                    return False
                candidate_indexes[normalized] = len(candidates)
                candidates.append(PreparedCandidate(
                    surface=surface,
                    normalized=normalized,
                    sources=(source,),
                    student_token_ids=student_ids,
                    teacher_token_ids=teacher_ids,
                ))
                return True

            if not add_candidate(observed, "observed"):
                continue
            observed_words = list(WORD_PATTERN.finditer(observed))
            final_match = observed_words[-1]
            final_word = final_match.group()

            def replace_final(replacement: str) -> str:
                return observed[:final_match.start()] + replacement + observed[final_match.end():]

            normalized_word = _normalize_candidate(final_word)
            for replacement in confusion_map.get(normalized_word, ()):
                add_candidate(replace_final(replacement), "confusion")
            for source, target in aliases.items():
                if normalized_word == _normalize_candidate(source):
                    add_candidate(replace_final(target), "alias")
                elif normalized_word == _normalize_candidate(target):
                    add_candidate(replace_final(source), "alias")
            for replacement, source in _edit_variants(final_word, record["id"]):
                add_candidate(replace_final(replacement), source)
            if " " in observed:
                add_candidate(observed.replace(" ", ""), "missed-space")
            elif record["language"] == "de" and len(final_word) >= 12:
                midpoint = len(final_word) // 2
                add_candidate(replace_final(final_word[:midpoint] + " " + final_word[midpoint:]), "compound-split")

            lexicon_offset = int(hashlib.sha256((record["id"] + "\0distractor").encode()).hexdigest()[:16], 16)
            for index in range(len(lexicon)):
                if len(candidates) >= policy.maximum_candidates:
                    break
                replacement = lexicon[(lexicon_offset + index) % len(lexicon)]
                if abs(len(replacement) - len(normalized_word)) <= 2:
                    add_candidate(replace_final(replacement), "lexicon")
            if len(candidates) < policy.minimum_candidates:
                continue
            candidates = candidates[:policy.maximum_candidates]
            identifier = hashlib.sha256(_canonical_line({
                "sourceRecordId": record["id"],
                "prefixTokenIds": list(prefix_student_ids),
                "candidates": [candidate.normalized for candidate in candidates],
                "policySha256": policy.sha256,
            })).hexdigest()
            return PreparedExample(
                identifier=identifier,
                split=record["split"],
                language=record["language"],
                field_class_id=FIELD_CLASS_IDS[record["fieldClass"]],
                source_id=record["sourceId"],
                prefix_student_token_ids=prefix_student_ids,
                prefix_teacher_token_ids=prefix_teacher_ids,
                candidates=tuple(candidates),
            ), None
    return None, "no_eligible_target"


def _confusions(project: prepare_context_dataset.ProjectCorpus) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    result: dict[str, set[str]] = collections.defaultdict(set)
    for values in project.raw["confusionSets"]:
        normalized = [_normalize_candidate(value) for value in values]
        for source in normalized:
            result[source].update(value for value in normalized if value != source)
    return {key: tuple(sorted(values)) for key, values in result.items()}, dict(project.raw["aliases"])


def _score_rows(examples: list[PreparedExample], model, teacher_tokenizer, torch) -> list[list[float]]:
    rows: list[list[int]] = []
    starts: list[int] = []
    targets: list[tuple[int, ...]] = []
    owners: list[int] = []
    bos = teacher_tokenizer.bos_token_id
    pad = teacher_tokenizer.pad_token_id
    if not isinstance(bos, int) or bos < 0 or not isinstance(pad, int) or pad < 0:
        raise ContextTeacherError("teacher tokenizer is missing bounded BOS or padding tokens")
    for owner, example in enumerate(examples):
        for candidate in example.candidates:
            start = 1 + len(example.prefix_teacher_token_ids)
            rows.append([bos, *example.prefix_teacher_token_ids, *candidate.teacher_token_ids])
            starts.append(start)
            targets.append(candidate.teacher_token_ids)
            owners.append(owner)
    maximum = max(map(len, rows))
    input_ids = torch.full((len(rows), maximum), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), maximum), dtype=torch.long)
    for row, values in enumerate(rows):
        input_ids[row, :len(values)] = torch.tensor(values, dtype=torch.long)
        attention_mask[row, :len(values)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ContextTeacherError("teacher returned invalid causal-language-model logits")
    result: list[list[float]] = [[] for _example in examples]
    for row, (start, target, owner) in enumerate(zip(starts, targets, owners, strict=True)):
        positions = torch.arange(start - 1, start + len(target) - 1)
        selected = logits[row, positions, :].float()
        if not torch.isfinite(selected).all():
            raise ContextTeacherError("teacher returned non-finite candidate logits")
        labels = torch.tensor(target, dtype=torch.long)
        token_scores = torch.log_softmax(selected, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        score = float(token_scores.mean().item())
        if not math.isfinite(score):
            raise ContextTeacherError("teacher produced a non-finite candidate score")
        result[owner].append(round(score, 7))
    return result


def _score_chunk(
    examples: list[PreparedExample],
    *,
    model,
    teacher_tokenizer,
    torch,
    maximum_batch_rows: int,
) -> tuple[bytes, dict[str, Any]]:
    output = bytearray()
    candidate_rows = 0
    observed_top1 = 0
    slate_sizes: collections.Counter[str] = collections.Counter()
    languages: collections.Counter[str] = collections.Counter()
    cursor = 0
    while cursor < len(examples):
        end = cursor
        rows = 0
        while end < len(examples):
            next_rows = len(examples[end].candidates)
            if end > cursor and rows + next_rows > maximum_batch_rows:
                break
            if next_rows > maximum_batch_rows:
                raise ContextTeacherError("candidate slate exceeds the teacher inference batch")
            rows += next_rows
            end += 1
        batch = examples[cursor:end]
        scores = _score_rows(batch, model, teacher_tokenizer, torch)
        for example, example_scores in zip(batch, scores, strict=True):
            output.extend(_canonical_line(example.output_document(example_scores)))
            candidate_rows += len(example.candidates)
            observed_top1 += max(range(len(example_scores)), key=example_scores.__getitem__) == 0
            slate_sizes[str(len(example.candidates))] += 1
            languages[example.language] += 1
        cursor = end
    return bytes(output), {
        "records": len(examples),
        "candidateRows": candidate_rows,
        "observedTeacherTop1": observed_top1,
        "slateSizes": dict(sorted(slate_sizes.items())),
        "languages": dict(sorted(languages.items())),
    }


def _atomic_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    try:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary.close()
        os.replace(temporary_path, path)
    finally:
        if not temporary.closed:
            temporary.close()
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _chunk_paths(chunk_root: pathlib.Path, split: str, index: int) -> tuple[pathlib.Path, pathlib.Path]:
    stem = f"{split}-{index:06d}"
    return chunk_root / f"{stem}.jsonl", chunk_root / f"{stem}.json"


def _write_or_reuse_chunk(
    examples: list[PreparedExample],
    *,
    split: str,
    index: int,
    chunk_root: pathlib.Path,
    bindings_sha256: str,
    model,
    teacher_tokenizer,
    torch,
    policy: DistillationPolicy,
) -> tuple[pathlib.Path, dict[str, Any], bool]:
    chunk_path, metadata_path = _chunk_paths(chunk_root, split, index)
    input_digest = hashlib.sha256()
    for example in examples:
        input_digest.update(_canonical_line(example.binding_document()))
    expected_input_sha = input_digest.hexdigest()
    if metadata_path.exists():
        metadata, _payload = _load_json(metadata_path, MAXIMUM_PROGRESS_BYTES, "context teacher chunk metadata")
        expected_keys = {
            "schemaVersion", "split", "chunkIndex", "bindingsSha256", "inputSha256",
            "file", "bytes", "sha256", "counts",
        }
        if (
            set(metadata) != expected_keys
            or metadata["schemaVersion"] != 1
            or metadata["split"] != split
            or metadata["chunkIndex"] != index
            or metadata["bindingsSha256"] != bindings_sha256
            or metadata["inputSha256"] != expected_input_sha
            or metadata["file"] != chunk_path.name
            or not chunk_path.is_file()
            or chunk_path.is_symlink()
            or chunk_path.stat().st_size != metadata["bytes"]
            or model_sources.file_sha256(chunk_path) != metadata["sha256"]
            or metadata.get("counts", {}).get("records") != len(examples)
        ):
            raise ContextTeacherError(f"existing teacher chunk does not match its inputs: {chunk_path.name}")
        return chunk_path, metadata, True
    if chunk_path.exists() and (not chunk_path.is_file() or chunk_path.is_symlink()):
        raise ContextTeacherError(f"incomplete teacher chunk is not a regular file: {chunk_path.name}")
    payload, counts = _score_chunk(
        examples,
        model=model,
        teacher_tokenizer=teacher_tokenizer,
        torch=torch,
        maximum_batch_rows=policy.inference_batch_rows,
    )
    _atomic_bytes(chunk_path, payload)
    metadata = {
        "schemaVersion": 1,
        "split": split,
        "chunkIndex": index,
        "bindingsSha256": bindings_sha256,
        "inputSha256": expected_input_sha,
        "file": chunk_path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "counts": counts,
    }
    _atomic_json(metadata_path, metadata)
    return chunk_path, metadata, False


def _load_teacher(
    *,
    source_root: pathlib.Path,
    source: model_sources.Source,
    threads: int,
    torch,
    AutoModelForCausalLM,
    AutoTokenizer,
):
    model_sources.verify_source(source_root, source)
    teacher_root = source_root.resolve() / source.identifier
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_root,
        local_files_only=True,
        dtype=torch.float32,
    ).eval()
    if (
        type(model).__name__ != "LlamaForCausalLM"
        or getattr(model.config, "vocab_size", None) != 32_000
        or sum(parameter.numel() for parameter in model.parameters()) != TEACHER_PARAMETER_COUNT
        or getattr(teacher_tokenizer, "vocab_size", None) != 32_000
    ):
        raise ContextTeacherError("pinned context teacher architecture is incompatible")
    return model, teacher_tokenizer


def score(args: argparse.Namespace) -> dict[str, Any]:
    policy = load_policy(args.policy)
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    source_root = args.source_root.resolve()
    maximum_records = args.maximum_records_per_split
    if maximum_records is not None and (not args.development or maximum_records <= 0):
        raise ContextTeacherError("bounded scoring is permitted only for a positive development run")
    app_commit, dirty = _git_state()
    if dirty and not args.development:
        raise ContextTeacherError("release teacher scoring requires a clean Git worktree")

    corpus_manifest = prepare_context_dataset.load_prepared_manifest(data_root, require_pinned=not args.development)
    corpus_manifest_path = data_root / "split-manifest.json"
    corpus_manifest_sha = model_sources.file_sha256(corpus_manifest_path)
    context_policy = prepare_context_dataset.load_policy()
    project = prepare_context_dataset.load_project_corpus(
        ROOT / context_policy.project_authored_data,
        context_policy,
    )
    tokenizer_path = (args.tokenizer or data_root / "tokenizer.json").resolve()
    student_tokenizer = context_tokenizer_contract.load_tokenizer(tokenizer_path, expected_vocabulary_size=16_384)
    if student_tokenizer.sha256 != policy.tokenizer_sha256:
        raise ContextTeacherError("context tokenizer does not match the distillation policy")

    source_manifest = model_sources.load_manifest(args.source_manifest)
    teacher_source = source_manifest.source(policy.teacher_source_id)
    if teacher_source.repository != "Evicka/Hanse2-100M-Base" or teacher_source.license != "Apache-2.0":
        raise ContextTeacherError("context teacher is not the approved Apache-2.0 Hanse2 model")
    if (
        corpus_manifest.get("sourceManifestSha256") != source_manifest.sha256
        or corpus_manifest.get("policySha256") != context_policy.sha256
        or corpus_manifest.get("projectDataSha256") != project.sha256
        or corpus_manifest.get("toolSha256") != model_sources.file_sha256(
            ROOT / "tools/prepare_context_dataset.py"
        )
    ):
        raise ContextTeacherError("prepared context corpus is not bound to the current source contracts")
    torch, AutoModelForCausalLM, AutoTokenizer, versions = _dependencies()
    model, teacher_tokenizer = _load_teacher(
        source_root=source_root,
        source=teacher_source,
        threads=args.threads,
        torch=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )
    lexicons = _build_lexicons(data_root, corpus_manifest, policy)
    confusion_map, aliases = _confusions(project)

    bindings = {
        "schemaVersion": 1,
        "modelId": "context-en-de-v1",
        "development": bool(args.development),
        "maximumRecordsPerSplit": maximum_records,
        "appCommit": app_commit,
        "dataManifestSha256": corpus_manifest_sha,
        "distillationPolicySha256": policy.sha256,
        "tokenizerSha256": student_tokenizer.sha256,
        "teacherModelSha256": teacher_source.artifact("model.safetensors").sha256,
        "sourceManifestSha256": source_manifest.sha256,
        "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
        "tokenizerContractToolSha256": model_sources.file_sha256(
            ROOT / "tools/context_tokenizer_contract.py"
        ),
        "threads": args.threads,
        "toolchain": versions,
    }
    bindings_payload = _canonical_line(bindings)
    bindings_sha = hashlib.sha256(bindings_payload).hexdigest()
    chunk_root = output_root / ".chunks"
    bindings_path = chunk_root / "bindings.json"
    if bindings_path.exists():
        existing, existing_payload = _load_json(bindings_path, MAXIMUM_PROGRESS_BYTES, "context teacher bindings")
        if existing != bindings or existing_payload != json.dumps(existing, indent=2, sort_keys=True).encode() + b"\n":
            raise ContextTeacherError("existing teacher-scoring workspace has incompatible bindings")
    else:
        _atomic_json(bindings_path, bindings)

    generation_counts: collections.Counter[str] = collections.Counter()
    chunk_metadata: dict[str, list[tuple[pathlib.Path, dict[str, Any]]]] = {}
    for split in prepare_context_dataset.SPLITS:
        name = f"{split}.sentences.jsonl"
        expected = corpus_manifest["outputs"][name]["records"]
        chunk: list[PreparedExample] = []
        chunks: list[tuple[pathlib.Path, dict[str, Any]]] = []
        scored_for_split = 0
        chunk_index = 0
        for record in _iter_records(data_root / name, split, expected):
            if maximum_records is not None and scored_for_split >= maximum_records:
                break
            generation_counts[f"source:{split}"] += 1
            example, rejection = _prepare_example(
                record,
                lexicon=lexicons[split][record["language"]],
                confusion_map=confusion_map if record["language"] == "de" else {},
                aliases=aliases if record["language"] == "de" else {},
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizer,
                policy=policy,
            )
            if example is None:
                generation_counts[f"rejection:{split}:{rejection}"] += 1
                continue
            chunk.append(example)
            scored_for_split += 1
            generation_counts[f"scored:{split}"] += 1
            generation_counts[f"language:{split}:{example.language}"] += 1
            if len(chunk) == policy.chunk_records:
                path, metadata, reused = _write_or_reuse_chunk(
                    chunk,
                    split=split,
                    index=chunk_index,
                    chunk_root=chunk_root,
                    bindings_sha256=bindings_sha,
                    model=model,
                    teacher_tokenizer=teacher_tokenizer,
                    torch=torch,
                    policy=policy,
                )
                chunks.append((path, metadata))
                print(json.dumps({
                    "split": split, "chunk": chunk_index, "records": len(chunk), "reused": reused,
                }, sort_keys=True), flush=True)
                chunk = []
                chunk_index += 1
        if chunk:
            path, metadata, reused = _write_or_reuse_chunk(
                chunk,
                split=split,
                index=chunk_index,
                chunk_root=chunk_root,
                bindings_sha256=bindings_sha,
                model=model,
                teacher_tokenizer=teacher_tokenizer,
                torch=torch,
                policy=policy,
            )
            chunks.append((path, metadata))
            print(json.dumps({
                "split": split, "chunk": chunk_index, "records": len(chunk), "reused": reused,
            }, sort_keys=True), flush=True)
        chunk_metadata[split] = chunks

    total_scored = sum(generation_counts[f"scored:{split}"] for split in prepare_context_dataset.SPLITS)
    total_german = sum(
        generation_counts[f"language:{split}:de"] for split in prepare_context_dataset.SPLITS
    )
    if not args.development:
        test_records = generation_counts["scored:test"]
        test_english = generation_counts["language:test:en-US"]
        test_german = generation_counts["language:test:de"]
        if (
            total_scored < policy.minimum_scored_records
            or total_german < policy.minimum_german_records
            or test_records < policy.minimum_held_out_records
            or test_english < policy.minimum_held_out_english_records
            or test_german < policy.minimum_held_out_german_records
        ):
            raise ContextTeacherError("teacher-scored corpus does not satisfy the release data floors")

    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="libreboard-teacher-output-", dir=output_root) as temporary:
        staging = pathlib.Path(temporary)
        for split in prepare_context_dataset.SPLITS:
            filename = f"{split}.scored.jsonl"
            path = staging / filename
            with path.open("wb") as output:
                for chunk_path, _metadata in chunk_metadata[split]:
                    with chunk_path.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            output.write(block)
                output.flush()
                os.fsync(output.fileno())
            languages = {
                language: generation_counts[f"language:{split}:{language}"]
                for language in ("en-US", "de")
                if generation_counts[f"language:{split}:{language}"]
            }
            outputs[filename] = {
                "bytes": path.stat().st_size,
                "sha256": model_sources.file_sha256(path),
                "records": generation_counts[f"scored:{split}"],
                "languages": languages,
            }

        teacher_metrics = {}
        for split, chunks in chunk_metadata.items():
            records = candidate_rows = observed_top1 = 0
            slate_sizes: collections.Counter[str] = collections.Counter()
            for _path, metadata in chunks:
                counts = metadata["counts"]
                records += counts["records"]
                candidate_rows += counts["candidateRows"]
                observed_top1 += counts["observedTeacherTop1"]
                slate_sizes.update(counts["slateSizes"])
            teacher_metrics[split] = {
                "records": records,
                "candidateRows": candidate_rows,
                "observedTop1": observed_top1,
                "observedTop1Rate": 0.0 if records == 0 else observed_top1 / records,
                "slateSizes": dict(sorted(slate_sizes.items())),
            }

        report = {
            "schemaVersion": 1,
            "modelId": "context-en-de-v1",
            "releaseEligible": not args.development and not dirty,
            "appCommit": app_commit,
            "dataManifestSha256": corpus_manifest_sha,
            "distillationPolicySha256": policy.sha256,
            "tokenizerSha256": student_tokenizer.sha256,
            "teacher": {
                "sourceId": teacher_source.identifier,
                "revision": teacher_source.revision,
                "license": teacher_source.license,
                "sourceUrl": teacher_source.source_url,
                "modelSha256": teacher_source.artifact("model.safetensors").sha256,
            },
            "toolSha256": bindings["toolSha256"],
            "tokenizerContractToolSha256": bindings["tokenizerContractToolSha256"],
            "toolchain": versions,
            "generationCounts": dict(sorted(generation_counts.items())),
            "teacherMetrics": teacher_metrics,
            "outputs": outputs,
        }
        report_name = "distillation-manifest-development.json" if args.development else "distillation-manifest.json"
        report_path = staging / report_name
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in sorted(staging.iterdir()):
            os.replace(path, output_root / path.name)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=pathlib.Path, default=model_sources.DEFAULT_MANIFEST)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--tokenizer", type=pathlib.Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--maximum-records-per-split", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 64:
        parser.error("--threads must be between 1 and 64")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        report = score(parse_args(argv))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        ContextTeacherError,
        context_tokenizer_contract.ContextTokenizerContractError,
        model_sources.ModelSourceError,
        prepare_context_dataset.ContextDataError,
        OSError,
        subprocess.CalledProcessError,
    ) as failure:
        print(f"context teacher error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
