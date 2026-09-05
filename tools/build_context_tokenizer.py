#!/usr/bin/env python3
"""Train and verify LibreBoard's deterministic data-only bilingual BPE tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Iterator

import context_tokenizer_contract
import model_sources
import prepare_context_dataset


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "models" / "context" / "tokenizer-policy.json"
DEFAULT_DATA_ROOT = ROOT / "build" / "model-data" / "context-en-de-v1"
POLICY_KEYS = {
    "schemaVersion", "modelId", "normalization", "vocabularySize", "minimumFrequency",
    "alphabetLimit", "specialTokens", "languages",
}
MAXIMUM_POLICY_BYTES = 256 * 1024
RECORD_KEYS = {
    "schemaVersion", "id", "sessionId", "split", "language", "fieldClass", "text", "sourceId",
}


class ContextTokenizerBuildError(ValueError):
    pass


@dataclass(frozen=True)
class TokenizerPolicy:
    path: pathlib.Path
    sha256: str
    vocabulary_size: int
    minimum_frequency: int
    alphabet_limit: int
    special_tokens: tuple[str, ...]
    languages: dict[str, str]


def load_policy(path: pathlib.Path = DEFAULT_POLICY) -> TokenizerPolicy:
    path = path.resolve()
    try:
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAXIMUM_POLICY_BYTES:
            raise ContextTokenizerBuildError("context tokenizer policy is missing, linked, empty, or too large")
        payload = path.read_bytes()
        raw = json.loads(payload)
    except ContextTokenizerBuildError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTokenizerBuildError(f"cannot read context tokenizer policy: {failure}") from failure
    if not isinstance(raw, dict) or set(raw) != POLICY_KEYS:
        raise ContextTokenizerBuildError("context tokenizer policy has an unexpected schema")
    if raw["schemaVersion"] != 1 or raw["modelId"] != "context-en-de-v1":
        raise ContextTokenizerBuildError("context tokenizer policy has an incompatible identity")
    if raw["normalization"] != "NFKC_LOWER":
        raise ContextTokenizerBuildError("context tokenizer policy has an incompatible normalization")
    integer_bounds = {
        "vocabularySize": (32, 16_384),
        "minimumFrequency": (1, 1_000_000),
        "alphabetLimit": (32, 16_384),
    }
    values = {}
    for field, (minimum, maximum) in integer_bounds.items():
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ContextTokenizerBuildError(f"context tokenizer policy has invalid {field}")
        values[field] = value
    special_tokens = raw["specialTokens"]
    expected_specials = ["<pad>", "<bos>", "<unk>", "<lang:en>", "<lang:de>"]
    if special_tokens != expected_specials:
        raise ContextTokenizerBuildError("context tokenizer special-token order is incompatible")
    languages = raw["languages"]
    if languages != {"en": "<lang:en>", "de": "<lang:de>"}:
        raise ContextTokenizerBuildError("context tokenizer language mapping is incompatible")
    return TokenizerPolicy(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        vocabulary_size=values["vocabularySize"],
        minimum_frequency=values["minimumFrequency"],
        alphabet_limit=values["alphabetLimit"],
        special_tokens=tuple(special_tokens),
        languages=dict(languages),
    )


def _dependencies():
    try:
        import tokenizers
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    except ImportError as failure:
        raise ContextTokenizerBuildError(
            "install the pinned dependencies from "
            "models/training/requirements-context-linux-x86_64.lock"
        ) from failure
    if tokenizers.__version__ != "0.22.2":
        raise ContextTokenizerBuildError(
            f"context tokenizer builder requires tokenizers 0.22.2, found {tokenizers.__version__}"
        )
    return tokenizers, Tokenizer, models, normalizers, pre_tokenizers, trainers


def _training_sentences(path: pathlib.Path, expected_records: int) -> Iterator[str]:
    records = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as failure:
                raise ContextTokenizerBuildError(
                    f"context training sentence {line_number} is invalid JSON: {failure}"
                ) from failure
            if not isinstance(value, dict) or set(value) != RECORD_KEYS:
                raise ContextTokenizerBuildError("context training sentence has an unexpected schema")
            if value["schemaVersion"] != 1 or value["split"] != "train":
                raise ContextTokenizerBuildError("context tokenizer input is not the training split")
            text = value["text"]
            if not isinstance(text, str) or not text:
                raise ContextTokenizerBuildError("context tokenizer input contains invalid text")
            records += 1
            yield text
    if records != expected_records:
        raise ContextTokenizerBuildError(
            f"context tokenizer read {records} records, expected {expected_records}"
        )


def _parity_probes(path: pathlib.Path, maximum: int = 256) -> list[str]:
    probes = [
        "ＨELLO LibreBoard",
        "Datenschutzgrundverordnung",
        "don't split contractions",
        "Ärger über größere Geräte",
    ]
    with path.open("rb") as stream:
        for line in stream:
            value = json.loads(line)
            probes.append(value["text"])
            if len(probes) >= maximum:
                break
    return probes


def build(
    *,
    data_root: pathlib.Path,
    policy_path: pathlib.Path = DEFAULT_POLICY,
    development: bool = False,
    pinned_manifest_path: pathlib.Path = prepare_context_dataset.DEFAULT_CORPUS_MANIFEST,
) -> dict[str, Any]:
    tokenizers, Tokenizer, models, normalizers, pre_tokenizers, trainers = _dependencies()
    policy = load_policy(policy_path)
    data_root = data_root.resolve()
    manifest = prepare_context_dataset.load_prepared_manifest(
        data_root,
        require_pinned=not development,
        pinned_path=pinned_manifest_path,
    )
    if not development and manifest["minimumsEnforced"] is not True:
        raise ContextTokenizerBuildError("a development-sized corpus cannot produce a release tokenizer")
    train_name = "train.sentences.jsonl"
    train_path = data_root / train_name
    train_records = manifest["outputs"][train_name]["records"]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
        split=True,
    )
    trainer = trainers.BpeTrainer(
        vocab_size=policy.vocabulary_size,
        min_frequency=policy.minimum_frequency,
        show_progress=False,
        special_tokens=list(policy.special_tokens),
        limit_alphabet=policy.alphabet_limit,
        initial_alphabet=["▁"],
    )
    tokenizer.train_from_iterator(
        _training_sentences(train_path, train_records),
        trainer=trainer,
        length=train_records,
    )
    try:
        state = json.loads(tokenizer.model.__getstate__())
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTokenizerBuildError(f"cannot inspect trained BPE model: {failure}") from failure
    if not isinstance(state, dict) or not isinstance(state.get("vocab"), dict) or not isinstance(state.get("merges"), list):
        raise ContextTokenizerBuildError("trained BPE model has an unexpected schema")
    if len(state["vocab"]) != policy.vocabulary_size:
        raise ContextTokenizerBuildError(
            f"trained BPE vocabulary has {len(state['vocab'])} tokens, expected {policy.vocabulary_size}"
        )
    document = {
        "schemaVersion": 1,
        "normalization": "NFKC_LOWER",
        "vocabulary": state["vocab"],
        "merges": state["merges"],
        "specialTokens": {
            "padding": "<pad>",
            "beginningOfSequence": "<bos>",
            "unknown": "<unk>",
            "languages": policy.languages,
        },
    }
    payload = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest_path = data_root / "split-manifest.json"
    data_manifest_sha = model_sources.file_sha256(manifest_path)
    tokenizer_name = "tokenizer-development.json" if development else "tokenizer.json"
    report_name = "tokenizer-report-development.json" if development else "tokenizer-report.json"
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libreboard-tokenizer-", dir=data_root) as temporary:
        staging = pathlib.Path(temporary)
        tokenizer_path = staging / tokenizer_name
        tokenizer_path.write_bytes(payload)
        validated = context_tokenizer_contract.load_tokenizer(
            tokenizer_path,
            expected_vocabulary_size=policy.vocabulary_size,
        )
        probes = _parity_probes(train_path)
        for probe in probes:
            generated_ids = tokenizer.encode(probe, add_special_tokens=False).ids
            runtime_ids = context_tokenizer_contract.encode_text(validated, probe)
            if generated_ids != runtime_ids:
                raise ContextTokenizerBuildError(
                    f"generated tokenizer disagrees with the runtime BPE algorithm for probe: {probe!r}"
                )
        report = {
            "schemaVersion": 1,
            "modelId": "context-en-de-v1",
            "releaseEligible": not development,
            "dataManifestSha256": data_manifest_sha,
            "tokenizerPolicySha256": policy.sha256,
            "toolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
            "trainingSplitSha256": manifest["outputs"][train_name]["sha256"],
            "trainingRecords": train_records,
            "runtimeParityProbes": len(probes),
            "toolchain": {"tokenizers": tokenizers.__version__},
            "tokenizer": {
                "file": tokenizer_name,
                "bytes": tokenizer_path.stat().st_size,
                "sha256": validated.sha256,
                "vocabularySize": len(document["vocabulary"]),
                "mergeCount": len(document["merges"]),
            },
        }
        report_path = staging / report_name
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tokenizer_path, data_root / tokenizer_name)
        os.replace(report_path, data_root / report_name)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--policy", type=pathlib.Path, default=DEFAULT_POLICY)
    parser.add_argument("--development", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build(
            data_root=args.data_root,
            policy_path=args.policy,
            development=args.development,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        ContextTokenizerBuildError,
        context_tokenizer_contract.ContextTokenizerContractError,
        prepare_context_dataset.ContextDataError,
    ) as failure:
        print(f"context tokenizer error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
