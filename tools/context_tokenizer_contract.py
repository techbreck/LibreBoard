#!/usr/bin/env python3
"""Dependency-free validation for LibreBoard's bilingual BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


MAXIMUM_TOKENIZER_BYTES = 2 * 1024 * 1024
MAXIMUM_TOKEN_LENGTH_UTF16 = 128
MAXIMUM_MERGES = 65_536
WORD_START = "▁"
TOP_LEVEL_KEYS = {"schemaVersion", "normalization", "vocabulary", "merges", "specialTokens"}
SPECIAL_TOKEN_KEYS = {"padding", "beginningOfSequence", "unknown", "languages"}
EXPECTED_LANGUAGES = {"en", "de"}


class ContextTokenizerContractError(ValueError):
    pass


@dataclass(frozen=True)
class ContextTokenizer:
    path: pathlib.Path
    sha256: str
    raw: dict[str, Any]


def encode_text(tokenizer: ContextTokenizer, text: str) -> list[int]:
    """Mirror the bounded Kotlin BPE algorithm for build-time parity checks."""
    vocabulary = tokenizer.raw["vocabulary"]
    special = tokenizer.raw["specialTokens"]
    unknown_id = vocabulary[special["unknown"]]
    merge_ranks = {
        (merge[0], merge[1]): rank
        for rank, merge in enumerate(tokenizer.raw["merges"])
    }
    normalized = unicodedata.normalize("NFKC", text).lower()
    output: list[int] = []
    for word in re.split(r"\s+", normalized.strip()):
        if not word:
            continue
        symbols = [WORD_START, *word]
        while len(symbols) > 1:
            selected_index = -1
            selected_rank = len(merge_ranks) + 1
            for index in range(len(symbols) - 1):
                rank = merge_ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and rank < selected_rank:
                    selected_index = index
                    selected_rank = rank
            if selected_index < 0:
                break
            symbols[selected_index:selected_index + 2] = [
                symbols[selected_index] + symbols[selected_index + 1]
            ]
        output.extend(vocabulary.get(symbol, unknown_id) for symbol in symbols)
    return output


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContextTokenizerContractError(f"context tokenizer contains duplicate key: {key}")
        result[key] = value
    return result


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def load_tokenizer(
    path: pathlib.Path,
    *,
    expected_vocabulary_size: int | None = None,
) -> ContextTokenizer:
    path = path.resolve()
    try:
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAXIMUM_TOKENIZER_BYTES:
            raise ContextTokenizerContractError("context tokenizer is missing, linked, empty, or too large")
        payload = path.read_bytes()
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    except ContextTokenizerContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTokenizerContractError(f"cannot read context tokenizer: {failure}") from failure

    if not isinstance(raw, dict) or set(raw) != TOP_LEVEL_KEYS:
        raise ContextTokenizerContractError("context tokenizer has an unexpected schema")
    if raw["schemaVersion"] != 1 or raw["normalization"] != "NFKC_LOWER":
        raise ContextTokenizerContractError("context tokenizer has an unsupported schema or normalization")

    vocabulary = raw["vocabulary"]
    if not isinstance(vocabulary, dict) or not 4 <= len(vocabulary) <= 16_384:
        raise ContextTokenizerContractError("context tokenizer vocabulary size is invalid")
    if expected_vocabulary_size is not None and len(vocabulary) != expected_vocabulary_size:
        raise ContextTokenizerContractError(
            f"context tokenizer must contain exactly {expected_vocabulary_size} tokens"
        )
    if any(
        not isinstance(token, str)
        or not token
        or _utf16_length(token) > MAXIMUM_TOKEN_LENGTH_UTF16
        for token in vocabulary
    ):
        raise ContextTokenizerContractError("context tokenizer contains an invalid token")
    ids = list(vocabulary.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
        raise ContextTokenizerContractError("context tokenizer IDs must be integers")
    if sorted(ids) != list(range(len(vocabulary))):
        raise ContextTokenizerContractError("context tokenizer IDs must be unique and dense")
    if WORD_START not in vocabulary:
        raise ContextTokenizerContractError("context tokenizer is missing the word-start token")

    merges = raw["merges"]
    if not isinstance(merges, list) or len(merges) > MAXIMUM_MERGES:
        raise ContextTokenizerContractError("context tokenizer merge list is invalid")
    seen_merges: set[tuple[str, str]] = set()
    for index, merge in enumerate(merges):
        if (
            not isinstance(merge, list)
            or len(merge) != 2
            or any(not isinstance(part, str) or not part for part in merge)
        ):
            raise ContextTokenizerContractError(f"context tokenizer merge {index + 1} is invalid")
        pair = (merge[0], merge[1])
        if pair in seen_merges:
            raise ContextTokenizerContractError("context tokenizer contains a duplicate merge")
        seen_merges.add(pair)
        if merge[0] + merge[1] not in vocabulary:
            raise ContextTokenizerContractError("context tokenizer merge output is absent from the vocabulary")

    special = raw["specialTokens"]
    if not isinstance(special, dict) or set(special) != SPECIAL_TOKEN_KEYS:
        raise ContextTokenizerContractError("context tokenizer special tokens have an unexpected schema")
    languages = special["languages"]
    if not isinstance(languages, dict) or set(languages) != EXPECTED_LANGUAGES:
        raise ContextTokenizerContractError("context tokenizer must declare exactly English and German")
    special_values = [special["padding"], special["beginningOfSequence"], special["unknown"]]
    special_values.extend(languages.values())
    if any(not isinstance(token, str) or token not in vocabulary for token in special_values):
        raise ContextTokenizerContractError("context tokenizer references a missing special token")
    if len({vocabulary[token] for token in special_values}) != len(special_values):
        raise ContextTokenizerContractError("context tokenizer special tokens must use distinct IDs")

    return ContextTokenizer(path=path, sha256=hashlib.sha256(payload).hexdigest(), raw=raw)
