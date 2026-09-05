#!/usr/bin/env python3
"""Train LibreBoard's bilingual candidate rescorer from pinned teacher-scored slates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterator

import canonical_safetensors
import context_model_contract
import context_tokenizer_contract
import model_sources
import score_context_teacher


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SCORED_ROOT = ROOT / "build" / "model-data" / "context-en-de-v1" / "teacher-scored-v1"
DEFAULT_PINNED_MANIFEST = ROOT / "models" / "context" / "distillation-manifest.json"
DEFAULT_TOKENIZER = ROOT / "build" / "model-data" / "context-en-de-v1" / "tokenizer.json"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-training" / "context-en-de-v1"
MANIFEST_KEYS = {
    "schemaVersion", "modelId", "releaseEligible", "appCommit", "dataManifestSha256",
    "distillationPolicySha256", "tokenizerSha256", "teacher", "toolSha256",
    "tokenizerContractToolSha256", "toolchain", "generationCounts", "teacherMetrics", "outputs",
}
MANIFEST_OUTPUT_KEYS = {"bytes", "sha256", "records", "languages"}
TEACHER_KEYS = {"sourceId", "revision", "license", "sourceUrl", "modelSha256"}
CHECKPOINT_METADATA_KEYS = {
    "schemaVersion", "appCommit", "dataManifestSha256", "modelSpecSha256",
    "trainingToolSha256", "modelImplementationSha256", "tokenizerSha256",
    "teacherModelSha256", "deviceType", "torchVersion", "threads", "currentEpoch",
    "completedExamples", "optimizerSteps", "epochReports", "lossTotals",
}
LOSS_NAMES = ("combined", "teacherCrossEntropy", "observedCrossEntropy", "ranking")
MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
MAXIMUM_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024


class ContextTrainingError(ValueError):
    pass


def _dependencies():
    try:
        import safetensors
        import torch
        from safetensors.torch import save_file
    except ImportError as failure:
        raise ContextTrainingError(
            "install the pinned dependencies from models/training/requirements-context-linux-x86_64.lock"
        ) from failure
    versions = {
        "torch": str(torch.__version__).split("+", 1)[0],
        "safetensors": safetensors.__version__,
    }
    if versions != {"torch": "2.8.0", "safetensors": "0.6.2"}:
        raise ContextTrainingError(f"context training dependencies do not match the lock: {versions}")
    return torch, safetensors, save_file, versions


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
        raise ContextTrainingError("context training requires a full Git commit")
    return commit, dirty


def _load_manifest(
    path: pathlib.Path,
    *,
    scored_root: pathlib.Path,
    development: bool,
    pinned_path: pathlib.Path = DEFAULT_PINNED_MANIFEST,
) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    try:
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAXIMUM_MANIFEST_BYTES:
            raise ContextTrainingError("context distillation manifest is missing, linked, empty, or too large")
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except ContextTrainingError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTrainingError(f"cannot read context distillation manifest: {failure}") from failure
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ContextTrainingError("context distillation manifest has an unexpected schema")
    if (
        manifest["schemaVersion"] != 1
        or manifest["modelId"] != "context-en-de-v1"
        or not isinstance(manifest["releaseEligible"], bool)
        or (not development and manifest["releaseEligible"] is not True)
    ):
        raise ContextTrainingError("context distillation manifest has an incompatible release identity")
    for field in (
        "appCommit", "dataManifestSha256", "distillationPolicySha256", "tokenizerSha256",
        "toolSha256", "tokenizerContractToolSha256",
    ):
        value = manifest[field]
        pattern = model_sources.REVISION if field == "appCommit" else model_sources.SHA256
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ContextTrainingError(f"context distillation manifest has invalid {field}")
    teacher = manifest["teacher"]
    if not isinstance(teacher, dict) or set(teacher) != TEACHER_KEYS:
        raise ContextTrainingError("context distillation manifest has invalid teacher metadata")
    if (
        teacher["sourceId"] != "hanse2-100m-base-teacher-v1"
        or teacher["license"] != "Apache-2.0"
        or not isinstance(teacher["revision"], str)
        or not model_sources.REVISION.fullmatch(teacher["revision"])
        or not isinstance(teacher["modelSha256"], str)
        or not model_sources.SHA256.fullmatch(teacher["modelSha256"])
        or not isinstance(teacher["sourceUrl"], str)
        or not teacher["sourceUrl"].startswith("https://")
    ):
        raise ContextTrainingError("context distillation teacher is not the approved model")
    outputs = manifest["outputs"]
    expected_outputs = {f"{split}.scored.jsonl" for split in ("train", "validation", "test")}
    if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
        raise ContextTrainingError("context distillation outputs are incomplete")
    for filename, details in outputs.items():
        if (
            not isinstance(details, dict)
            or set(details) != MANIFEST_OUTPUT_KEYS
            or isinstance(details["bytes"], bool)
            or not isinstance(details["bytes"], int)
            or details["bytes"] <= 0
            or not isinstance(details["sha256"], str)
            or not model_sources.SHA256.fullmatch(details["sha256"])
            or isinstance(details["records"], bool)
            or not isinstance(details["records"], int)
            or details["records"] <= 0
            or not isinstance(details["languages"], dict)
            or not details["languages"]
            or not set(details["languages"]).issubset({"en-US", "de"})
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in details["languages"].values()
            )
            or sum(details["languages"].values()) != details["records"]
        ):
            raise ContextTrainingError(f"context distillation output metadata is invalid: {filename}")
        output_path = scored_root / filename
        if (
            not output_path.is_file()
            or output_path.is_symlink()
            or output_path.stat().st_size != details["bytes"]
            or model_sources.file_sha256(output_path) != details["sha256"]
        ):
            raise ContextTrainingError(f"context distillation output does not match its manifest: {filename}")
    if not development:
        try:
            if not pinned_path.is_file() or pinned_path.is_symlink() or pinned_path.read_bytes() != payload:
                raise ContextTrainingError("context distillation manifest does not match the pinned release manifest")
        except OSError as failure:
            raise ContextTrainingError(f"cannot read pinned context distillation manifest: {failure}") from failure
    return manifest, hashlib.sha256(payload).hexdigest()


def _parse_record(
    line: bytes,
    *,
    split: str,
    vocabulary_size: int,
    maximum_prefix_tokens: int,
    maximum_candidate_tokens: int,
    maximum_candidates: int,
) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise ContextTrainingError(f"scored {split} data contains invalid JSON: {failure}") from failure
    if not isinstance(value, dict) or set(value) != score_context_teacher.OUTPUT_KEYS:
        raise ContextTrainingError(f"scored {split} record has an unexpected schema")
    if (
        value["schemaVersion"] != 1
        or value["split"] != split
        or value["language"] not in {"en-US", "de"}
        or value["fieldClassId"] not in {0, 1, 2}
        or value["observedCandidateIndex"] != 0
        or not isinstance(value["id"], str)
        or not model_sources.SHA256.fullmatch(value["id"])
        or not isinstance(value["sourceId"], str)
        or not model_sources.SOURCE_ID.fullmatch(value["sourceId"])
    ):
        raise ContextTrainingError(f"scored {split} record has incompatible identity fields")
    prefix = value["prefixTokenIds"]
    if (
        not isinstance(prefix, list)
        or not 0 < len(prefix) <= maximum_prefix_tokens
        or any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocabulary_size for token in prefix)
    ):
        raise ContextTrainingError(f"scored {split} record has invalid prefix tokens")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= maximum_candidates:
        raise ContextTrainingError(f"scored {split} record has an invalid candidate slate")
    normalized_values = set()
    allowed_sources = {
        "observed", "confusion", "alias", "delete", "transpose", "neighbor",
        "punctuation-drop", "missed-space", "compound-split", "lexicon",
    }
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != score_context_teacher.CANDIDATE_KEYS:
            raise ContextTrainingError(f"scored {split} candidate has an unexpected schema")
        surface = candidate["surface"]
        normalized = candidate["normalized"]
        sources = candidate["sources"]
        tokens = candidate["tokenIds"]
        score = candidate["teacherMeanLogProbability"]
        if (
            not isinstance(surface, str)
            or not surface
            or not isinstance(normalized, str)
            or normalized != score_context_teacher._normalize_candidate(surface)
            or normalized in normalized_values
            or not isinstance(sources, list)
            or not sources
            or len(sources) != len(set(sources))
            or any(source not in allowed_sources for source in sources)
            or not isinstance(tokens, list)
            or not 0 < len(tokens) <= maximum_candidate_tokens
            or any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocabulary_size for token in tokens)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not -1_000.0 <= float(score) <= 0.0
        ):
            raise ContextTrainingError(f"scored {split} candidate is invalid")
        normalized_values.add(normalized)
        if index == 0 and "observed" not in sources:
            raise ContextTrainingError(f"scored {split} slate does not preserve the observed candidate")
    return value


def _records(
    path: pathlib.Path,
    *,
    split: str,
    expected_records: int,
    seed: int,
    shuffle_buffer: int,
    parser_arguments: dict[str, int],
) -> Iterator[dict[str, Any]]:
    randomizer = random.Random(seed)
    buffer: list[dict[str, Any]] = []
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            count += 1
            record = _parse_record(line, split=split, **parser_arguments)
            if shuffle_buffer <= 1:
                yield record
            elif len(buffer) < shuffle_buffer:
                buffer.append(record)
            else:
                index = randomizer.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = record
    if count != expected_records:
        raise ContextTrainingError(f"scored {split} record count changed while training")
    if shuffle_buffer > 1:
        randomizer.shuffle(buffer)
        yield from buffer


def _model_inputs(record: dict[str, Any], tokenizer, spec, torch, device, candidate_rows: int):
    architecture = spec.architecture
    candidates = record["candidates"]
    if not len(candidates) <= candidate_rows <= architecture["maximumCandidates"]:
        raise ContextTrainingError("scored candidate slate cannot fit the training batch")
    rows = candidate_rows
    sequence = architecture["sequenceLength"]
    candidate_start = sequence - architecture["maximumCandidateTokens"]
    padding = tokenizer.raw["vocabulary"][tokenizer.raw["specialTokens"]["padding"]]
    beginning = tokenizer.raw["vocabulary"][tokenizer.raw["specialTokens"]["beginningOfSequence"]]
    language_key = "en" if record["language"] == "en-US" else "de"
    language_token = tokenizer.raw["specialTokens"]["languages"][language_key]
    language = tokenizer.raw["vocabulary"][language_token]
    prefix = [beginning, language, *record["prefixTokenIds"]]
    if len(prefix) > candidate_start:
        raise ContextTrainingError("scored context prefix exceeds the model ABI")
    input_ids = torch.full((rows, sequence), padding, dtype=torch.long, device=device)
    attention_mask = torch.zeros((rows, sequence), dtype=torch.long, device=device)
    candidate_mask = torch.zeros((rows, sequence), dtype=torch.float32, device=device)
    prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device)
    input_ids[:, :len(prefix)] = prefix_tensor
    attention_mask[:, :len(prefix)] = 1
    for row, candidate in enumerate(candidates):
        tokens = candidate["tokenIds"]
        input_ids[row, candidate_start:candidate_start + len(tokens)] = torch.tensor(
            tokens, dtype=torch.long, device=device,
        )
        attention_mask[row, candidate_start:candidate_start + len(tokens)] = 1
        candidate_mask[row, candidate_start:candidate_start + len(tokens)] = 1.0
    field_class = torch.full((rows,), record["fieldClassId"], dtype=torch.long, device=device)
    teacher_scores = torch.full((rows,), -1_000.0, dtype=torch.float32, device=device)
    teacher_scores[:len(candidates)] = torch.tensor(
        [candidate["teacherMeanLogProbability"] for candidate in candidates],
        dtype=torch.float32,
        device=device,
    )
    valid_candidates = torch.zeros((rows,), dtype=torch.bool, device=device)
    valid_candidates[:len(candidates)] = True
    return input_ids, attention_mask, candidate_mask, field_class, teacher_scores, valid_candidates


def _batched_inputs(records, tokenizer, spec, torch, device, candidate_rows: int):
    prepared = [
        _model_inputs(record, tokenizer, spec, torch, device, candidate_rows)
        for record in records
    ]
    return tuple(
        torch.stack([item[index] for item in prepared], dim=0)
        for index in range(6)
    )


def _forward_batch(model, inputs, torch, *, training: bool):
    randomness = "different" if training else "error"
    return torch.vmap(model, in_dims=(0, 0, 0, 0), randomness=randomness)(*inputs[:4])


def _batch_losses(student_scores, teacher_scores, valid_candidates, training, torch):
    losses = {name: [] for name in LOSS_NAMES}
    for row in range(student_scores.shape[0]):
        count = int(valid_candidates[row].sum().item())
        values = _losses(
            student_scores[row, :count],
            teacher_scores[row, :count],
            0,
            training,
            torch,
        )
        for name, value in values.items():
            losses[name].append(value)
    return {name: torch.stack(values).mean() for name, values in losses.items()}


def _losses(student_scores, teacher_scores, observed_index: int, training: dict[str, Any], torch):
    temperature = training["teacherTemperature"]
    teacher_distribution = torch.softmax(teacher_scores / temperature, dim=0)
    teacher_loss = -(
        teacher_distribution * torch.log_softmax(student_scores / temperature, dim=0)
    ).sum() * temperature * temperature
    observed_loss = -torch.log_softmax(student_scores, dim=0)[observed_index]
    negative_mask = torch.ones_like(student_scores, dtype=torch.bool)
    negative_mask[observed_index] = False
    best_negative = student_scores.masked_fill(~negative_mask, torch.finfo(student_scores.dtype).min).max()
    ranking_loss = torch.relu(
        torch.as_tensor(training["rankingMargin"], dtype=student_scores.dtype, device=student_scores.device)
        - student_scores[observed_index]
        + best_negative
    )
    combined = (
        training["teacherLossWeight"] * teacher_loss
        + training["observedLossWeight"] * observed_loss
        + training["rankingLossWeight"] * ranking_loss
    )
    return {
        "combined": combined,
        "teacherCrossEntropy": teacher_loss,
        "observedCrossEntropy": observed_loss,
        "ranking": ranking_loss,
    }


def _evaluate(
    model,
    *,
    path: pathlib.Path,
    manifest: dict[str, Any],
    split: str,
    tokenizer,
    spec,
    torch,
    device,
    parser_arguments: dict[str, int],
    maximum_examples: int | None,
    examples_per_batch: int,
    candidate_rows: int,
) -> dict[str, Any]:
    details = manifest["outputs"][f"{split}.scored.jsonl"]
    examples = candidates = teacher_top1 = observed_top1 = pairwise_correct = pairwise_total = 0
    loss_totals = {name: 0.0 for name in LOSS_NAMES}
    model.eval()

    def evaluate_batch(records: list[dict[str, Any]]) -> None:
        nonlocal examples, candidates, teacher_top1, observed_top1, pairwise_correct, pairwise_total
        inputs = _batched_inputs(records, tokenizer, spec, torch, device, candidate_rows)
        student_batch = _forward_batch(model, inputs, torch, training=False)
        losses = _batch_losses(student_batch, inputs[4], inputs[5], spec.training, torch)
        if any(not torch.isfinite(value) for value in losses.values()):
            raise ContextTrainingError(f"non-finite {split} evaluation loss")
        for name, value in losses.items():
            loss_totals[name] += float(value.item()) * len(records)
        for row, record in enumerate(records):
            count = len(record["candidates"])
            student_scores = student_batch[row, :count]
            teacher_scores = inputs[4][row, :count]
            student_order = student_scores.argsort(descending=True).tolist()
            teacher_order = teacher_scores.argsort(descending=True).tolist()
            teacher_top1 += student_order[0] == teacher_order[0]
            observed_top1 += student_order[0] == 0
            for first in range(count):
                for second in range(first + 1, count):
                    teacher_delta = float(teacher_scores[first] - teacher_scores[second])
                    if teacher_delta == 0:
                        continue
                    student_delta = float(student_scores[first] - student_scores[second])
                    pairwise_correct += (teacher_delta > 0) == (student_delta > 0)
                    pairwise_total += 1
            examples += 1
            candidates += count

    pending: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in _records(
            path,
            split=split,
            expected_records=details["records"],
            seed=spec.training["seed"],
            shuffle_buffer=1,
            parser_arguments=parser_arguments,
        ):
            if maximum_examples is not None and examples + len(pending) >= maximum_examples:
                break
            pending.append(record)
            if len(pending) == examples_per_batch:
                evaluate_batch(pending)
                pending = []
        if pending:
            evaluate_batch(pending)
    if examples == 0:
        raise ContextTrainingError(f"{split} evaluation produced no examples")
    return {
        "examples": examples,
        "candidateRows": candidates,
        "teacherTop1Agreement": teacher_top1 / examples,
        "observedTop1Accuracy": observed_top1 / examples,
        "teacherPairwiseAgreement": 0.0 if pairwise_total == 0 else pairwise_correct / pairwise_total,
        "meanLosses": {name: loss_totals[name] / examples for name in LOSS_NAMES},
    }


def _checkpoint_bindings(
    *,
    app_commit: str,
    data_manifest_sha256: str,
    model_spec_sha256: str,
    tokenizer_sha256: str,
    teacher_model_sha256: str,
    device_type: str,
    torch_version: str,
    threads: int,
) -> dict[str, str]:
    return {
        "schemaVersion": "1",
        "appCommit": app_commit,
        "dataManifestSha256": data_manifest_sha256,
        "modelSpecSha256": model_spec_sha256,
        "trainingToolSha256": model_sources.file_sha256(pathlib.Path(__file__)),
        "modelImplementationSha256": model_sources.file_sha256(
            ROOT / "models" / "training" / "context_model.py"
        ),
        "tokenizerSha256": tokenizer_sha256,
        "teacherModelSha256": teacher_model_sha256,
        "deviceType": device_type,
        "torchVersion": torch_version,
        "threads": str(threads),
    }


def _save_checkpoint(
    path: pathlib.Path,
    *,
    model,
    optimizer,
    bindings: dict[str, str],
    current_epoch: int,
    completed_examples: int,
    optimizer_steps: int,
    epoch_reports: list[dict[str, Any]],
    loss_totals: dict[str, float],
    torch,
    save_file,
) -> None:
    tensors = {
        f"model.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in sorted(model.state_dict().items())
    }
    for name, parameter in sorted(model.named_parameters()):
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ContextTrainingError(f"optimizer checkpoint state is incomplete for {name}")
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            value = state[state_name]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise ContextTrainingError(f"optimizer checkpoint state is invalid for {name}.{state_name}")
            tensors[f"optimizer.{name}.{state_name}"] = value.detach().cpu().contiguous()
    tensors["rng.cpu"] = torch.get_rng_state().cpu().contiguous()
    metadata = dict(bindings)
    metadata.update({
        "currentEpoch": str(current_epoch),
        "completedExamples": str(completed_examples),
        "optimizerSteps": str(optimizer_steps),
        "epochReports": json.dumps(epoch_reports, separators=(",", ":"), sort_keys=True),
        "lossTotals": json.dumps(loss_totals, separators=(",", ":"), sort_keys=True),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    temporary.close()
    try:
        save_file(tensors, temporary_path, metadata=metadata)
        canonical_safetensors.canonicalize(
            temporary_path,
            maximum_bytes=MAXIMUM_CHECKPOINT_BYTES,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _restore_checkpoint(path, *, model, optimizer, bindings, torch):
    try:
        from safetensors import safe_open

        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAXIMUM_CHECKPOINT_BYTES:
            raise ContextTrainingError("context training checkpoint is missing, linked, empty, or too large")
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
            tensor_names = set(checkpoint.keys())
            tensors = {name: checkpoint.get_tensor(name) for name in tensor_names}
    except ContextTrainingError:
        raise
    except Exception as failure:
        raise ContextTrainingError(f"cannot read context training checkpoint: {failure}") from failure
    if not isinstance(metadata, dict) or set(metadata) != CHECKPOINT_METADATA_KEYS:
        raise ContextTrainingError("context training checkpoint metadata has an unexpected schema")
    for key, value in bindings.items():
        if key != "appCommit" and metadata.get(key) != value:
            raise ContextTrainingError(f"context training checkpoint binding does not match: {key}")
    app_commit = metadata.get("appCommit", "")
    if not model_sources.REVISION.fullmatch(app_commit):
        raise ContextTrainingError("context training checkpoint has an invalid app commit")
    try:
        current_epoch = int(metadata["currentEpoch"])
        completed_examples = int(metadata["completedExamples"])
        optimizer_steps = int(metadata["optimizerSteps"])
        epoch_reports = json.loads(metadata["epochReports"])
        loss_totals = json.loads(metadata["lossTotals"])
    except (TypeError, ValueError, json.JSONDecodeError) as failure:
        raise ContextTrainingError("context training checkpoint has invalid progress metadata") from failure
    if (
        current_epoch < 0
        or completed_examples < 0
        or optimizer_steps < 0
        or not isinstance(epoch_reports, list)
        or len(epoch_reports) != current_epoch
        or not isinstance(loss_totals, dict)
        or set(loss_totals) != set(LOSS_NAMES)
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in loss_totals.values())
    ):
        raise ContextTrainingError("context training checkpoint progress is inconsistent")
    model_state = model.state_dict()
    expected_tensors = {f"model.{name}" for name in model_state} | {"rng.cpu"}
    for name, _parameter in model.named_parameters():
        expected_tensors.update({
            f"optimizer.{name}.step", f"optimizer.{name}.exp_avg", f"optimizer.{name}.exp_avg_sq",
        })
    if tensor_names != expected_tensors:
        raise ContextTrainingError("context training checkpoint tensor inventory does not match the model")
    try:
        model.load_state_dict({name: tensors[f"model.{name}"] for name in model_state}, strict=True)
        optimizer.state.clear()
        for name, parameter in model.named_parameters():
            optimizer.state[parameter] = {
                "step": tensors[f"optimizer.{name}.step"].cpu(),
                "exp_avg": tensors[f"optimizer.{name}.exp_avg"],
                "exp_avg_sq": tensors[f"optimizer.{name}.exp_avg_sq"],
            }
        torch.set_rng_state(tensors["rng.cpu"].to(dtype=torch.uint8))
    except (KeyError, RuntimeError, ValueError) as failure:
        raise ContextTrainingError(f"context training checkpoint state is incompatible: {failure}") from failure
    return current_epoch, completed_examples, optimizer_steps, epoch_reports, {
        name: float(loss_totals[name]) for name in LOSS_NAMES
    }, app_commit


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, safetensors, save_file, versions = _dependencies()
    spec = context_model_contract.load_spec(args.spec)
    scored_root = args.scored_root.resolve()
    manifest_path = (
        args.distillation_manifest.resolve()
        if args.distillation_manifest is not None
        else scored_root / ("distillation-manifest-development.json" if args.development else "distillation-manifest.json")
    )
    release_attempt = (
        not args.development
        and args.maximum_training_examples is None
        and args.maximum_validation_examples is None
        and args.maximum_testing_examples is None
        and args.epochs is None
        and args.device == "cpu"
    )
    manifest, data_manifest_sha = _load_manifest(
        manifest_path,
        scored_root=scored_root,
        development=not release_attempt,
        pinned_path=args.pinned_manifest,
    )
    tokenizer_path = args.tokenizer.resolve()
    tokenizer = context_tokenizer_contract.load_tokenizer(
        tokenizer_path,
        expected_vocabulary_size=spec.architecture["vocabularySize"],
    )
    policy = score_context_teacher.load_policy()
    source_manifest = model_sources.load_manifest()
    teacher_source = source_manifest.source(policy.teacher_source_id)
    if (
        tokenizer.sha256 != manifest["tokenizerSha256"]
        or tokenizer.sha256 != policy.tokenizer_sha256
        or manifest["distillationPolicySha256"] != policy.sha256
        or manifest["toolSha256"] != model_sources.file_sha256(ROOT / "tools/score_context_teacher.py")
        or manifest["tokenizerContractToolSha256"]
        != model_sources.file_sha256(ROOT / "tools/context_tokenizer_contract.py")
        or manifest["teacher"] != {
            "sourceId": teacher_source.identifier,
            "revision": teacher_source.revision,
            "license": teacher_source.license,
            "sourceUrl": teacher_source.source_url,
            "modelSha256": teacher_source.artifact("model.safetensors").sha256,
        }
    ):
        raise ContextTrainingError("context training inputs do not match the tokenizer/teacher policy")
    commit, dirty = _git_state()
    if dirty and not args.allow_dirty:
        raise ContextTrainingError("commit or stash the worktree before release-eligible context training")
    if args.resume_checkpoint is not None and not release_attempt:
        raise ContextTrainingError("checkpoint resume is available only for complete release training")

    sys.path.insert(0, str(ROOT))
    from models.training.context_model import ContextCandidateModel, trainable_parameter_count

    if args.device != "cpu":
        raise ContextTrainingError("context-en-de-v1 release training currently supports deterministic CPU only")
    device = torch.device("cpu")
    torch.manual_seed(spec.training["seed"])
    random.seed(spec.training["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    model = ContextCandidateModel(spec.architecture).to(device)
    parameter_count = trainable_parameter_count(model)
    if parameter_count != spec.raw["parameterCount"]:
        raise ContextTrainingError("implemented context model parameter count does not match the spec")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.training["learningRate"],
        weight_decay=spec.training["weightDecay"],
    )
    parser_arguments = {
        "vocabulary_size": spec.architecture["vocabularySize"],
        "maximum_prefix_tokens": policy.maximum_prefix_student_tokens,
        "maximum_candidate_tokens": spec.architecture["maximumCandidateTokens"],
        "maximum_candidates": policy.maximum_candidates,
    }
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.resume_checkpoint.resolve()
        if args.resume_checkpoint is not None
        else output_root / "checkpoint.safetensors"
    )
    if release_attempt and args.resume_checkpoint is None and checkpoint_path.exists():
        raise ContextTrainingError("context release checkpoint exists; pass --resume-checkpoint or use a clean output")
    bindings = _checkpoint_bindings(
        app_commit=commit,
        data_manifest_sha256=data_manifest_sha,
        model_spec_sha256=spec.sha256,
        tokenizer_sha256=tokenizer.sha256,
        teacher_model_sha256=manifest["teacher"]["modelSha256"],
        device_type=device.type,
        torch_version=str(torch.__version__),
        threads=args.threads,
    )
    epoch_reports: list[dict[str, Any]] = []
    current_epoch = completed_examples = optimizer_steps = 0
    loss_totals = {name: 0.0 for name in LOSS_NAMES}
    if args.resume_checkpoint is not None:
        (
            current_epoch, completed_examples, optimizer_steps, epoch_reports, loss_totals, commit,
        ) = _restore_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            bindings=bindings,
            torch=torch,
        )
    configured_epochs = args.epochs if args.epochs is not None else spec.training["epochs"]
    if configured_epochs <= 0 or (release_attempt and configured_epochs != spec.training["epochs"]):
        raise ContextTrainingError("context training epoch count is invalid")
    if current_epoch > configured_epochs:
        raise ContextTrainingError("context checkpoint exceeds the configured epoch count")

    train_details = manifest["outputs"]["train.scored.jsonl"]
    examples_per_batch = max(1, spec.training["batchSize"] // policy.maximum_candidates)
    if completed_examples > train_details["records"] or (
        completed_examples > 0
        and optimizer_steps != completed_examples // examples_per_batch
    ):
        raise ContextTrainingError("context checkpoint progress exceeds or disagrees with the training data")
    for epoch in range(current_epoch, configured_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        seen = 0
        pending: list[dict[str, Any]] = []
        if epoch != current_epoch:
            completed_examples = optimizer_steps = 0
            loss_totals = {name: 0.0 for name in LOSS_NAMES}
        records = _records(
            scored_root / "train.scored.jsonl",
            split="train",
            expected_records=train_details["records"],
            seed=spec.training["seed"] + epoch,
            shuffle_buffer=spec.training["shuffleBufferRecords"],
            parser_arguments=parser_arguments,
        )
        for record in records:
            if seen < completed_examples:
                seen += 1
                continue
            if args.maximum_training_examples is not None and seen >= args.maximum_training_examples:
                break
            pending.append(record)
            seen += 1
            completed_examples = seen
            if len(pending) < examples_per_batch:
                continue
            inputs = _batched_inputs(
                pending, tokenizer, spec, torch, device, policy.maximum_candidates,
            )
            student_scores = _forward_batch(model, inputs, torch, training=True)
            losses = _batch_losses(student_scores, inputs[4], inputs[5], spec.training, torch)
            if any(not torch.isfinite(value) for value in losses.values()):
                raise ContextTrainingError(f"non-finite context training loss at epoch {epoch + 1}")
            losses["combined"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.training["gradientClip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            for name, value in losses.items():
                loss_totals[name] += float(value.detach().item()) * len(pending)
            pending = []
            if (
                release_attempt
                and completed_examples % spec.training["checkpointEveryExamples"] == 0
            ):
                checkpoint_bindings = dict(bindings)
                checkpoint_bindings["appCommit"] = commit
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    bindings=checkpoint_bindings,
                    current_epoch=epoch,
                    completed_examples=completed_examples,
                    optimizer_steps=optimizer_steps,
                    epoch_reports=epoch_reports,
                    loss_totals=loss_totals,
                    torch=torch,
                    save_file=save_file,
                )
        if pending:
            inputs = _batched_inputs(
                pending, tokenizer, spec, torch, device, policy.maximum_candidates,
            )
            student_scores = _forward_batch(model, inputs, torch, training=True)
            losses = _batch_losses(student_scores, inputs[4], inputs[5], spec.training, torch)
            if any(not torch.isfinite(value) for value in losses.values()):
                raise ContextTrainingError(f"non-finite context training loss at epoch {epoch + 1}")
            losses["combined"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.training["gradientClip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            for name, value in losses.items():
                loss_totals[name] += float(value.detach().item()) * len(pending)
        trained_examples = completed_examples
        if trained_examples == 0:
            raise ContextTrainingError("context training produced no examples")
        validation = _evaluate(
            model,
            path=scored_root / "validation.scored.jsonl",
            manifest=manifest,
            split="validation",
            tokenizer=tokenizer,
            spec=spec,
            torch=torch,
            device=device,
            parser_arguments=parser_arguments,
            maximum_examples=args.maximum_validation_examples,
            examples_per_batch=examples_per_batch,
            candidate_rows=policy.maximum_candidates,
        )
        epoch_report = {
            "epoch": epoch + 1,
            "examples": trained_examples,
            "optimizerSteps": optimizer_steps,
            "meanLosses": {name: loss_totals[name] / trained_examples for name in LOSS_NAMES},
            "validation": validation,
        }
        epoch_reports.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        current_epoch = epoch + 1
        completed_examples = optimizer_steps = 0
        loss_totals = {name: 0.0 for name in LOSS_NAMES}
        if release_attempt:
            checkpoint_bindings = dict(bindings)
            checkpoint_bindings["appCommit"] = commit
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                bindings=checkpoint_bindings,
                current_epoch=current_epoch,
                completed_examples=0,
                optimizer_steps=0,
                epoch_reports=epoch_reports,
                loss_totals=loss_totals,
                torch=torch,
                save_file=save_file,
            )
        if args.maximum_training_examples is not None:
            break

    test_evaluation = _evaluate(
        model,
        path=scored_root / "test.scored.jsonl",
        manifest=manifest,
        split="test",
        tokenizer=tokenizer,
        spec=spec,
        torch=torch,
        device=device,
        parser_arguments=parser_arguments,
        maximum_examples=args.maximum_testing_examples,
        examples_per_batch=examples_per_batch,
        candidate_rows=policy.maximum_candidates,
    )
    model = model.to("cpu").eval()
    development = not release_attempt or dirty or len(epoch_reports) != spec.training["epochs"]
    weights_name = "context-en-de-v1-development.safetensors" if development else "context-en-de-v1.weights.safetensors"
    weights_path = output_root / weights_name
    metadata = {
        "appCommit": commit,
        "dataManifestSha256": data_manifest_sha,
        "modelSpecSha256": spec.sha256,
        "teacherModelSha256": manifest["teacher"]["modelSha256"],
        "tokenizerSha256": tokenizer.sha256,
    }
    temporary = tempfile.NamedTemporaryFile(dir=output_root, suffix=".safetensors", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    temporary.close()
    try:
        save_file(
            {name: tensor.detach().contiguous() for name, tensor in sorted(model.state_dict().items())},
            temporary_path,
            metadata=metadata,
        )
        canonical_safetensors.canonicalize(
            temporary_path,
            maximum_bytes=MAXIMUM_CHECKPOINT_BYTES,
        )
        os.replace(temporary_path, weights_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    tokenizer_output = output_root / "tokenizer.json"
    shutil.copyfile(tokenizer_path, tokenizer_output)
    provenance = [
        {
            "name": "FUTO swipe sentence corpus",
            "revision": source_manifest.source("futo-swipe-dataset-v1").revision,
            "license": "MIT",
            "source_url": source_manifest.source("futo-swipe-dataset-v1").source_url,
        },
        {
            "name": "Hanse2-100M-Base teacher",
            "revision": teacher_source.revision,
            "license": teacher_source.license,
            "source_url": teacher_source.source_url,
        },
    ]
    report = {
        "schemaVersion": 1,
        "modelId": spec.raw["modelId"],
        "releaseEligible": not development,
        "appCommit": commit,
        "modelSpecSha256": spec.sha256,
        "dataManifestSha256": data_manifest_sha,
        "parameterCount": parameter_count,
        "teacher": {"modelSha256": manifest["teacher"]["modelSha256"]},
        "provenance": provenance,
        "weights": {
            "file": weights_name,
            "bytes": weights_path.stat().st_size,
            "sha256": model_sources.file_sha256(weights_path),
        },
        "tokenizer": {
            "file": tokenizer_output.name,
            "bytes": tokenizer_output.stat().st_size,
            "sha256": model_sources.file_sha256(tokenizer_output),
        },
        "toolchain": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "safetensors": safetensors.__version__,
            "deviceType": device.type,
            "deterministicAlgorithms": True,
            "threads": args.threads,
        },
        "epochs": epoch_reports,
        "test": test_evaluation,
    }
    report_name = "training-report-development.json" if development else "training-report.json"
    report_path = output_root / report_name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(manifest_path, output_root / manifest_path.name)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=context_model_contract.DEFAULT_SPEC)
    parser.add_argument("--scored-root", type=pathlib.Path, default=DEFAULT_SCORED_ROOT)
    parser.add_argument("--distillation-manifest", type=pathlib.Path)
    parser.add_argument("--pinned-manifest", type=pathlib.Path, default=DEFAULT_PINNED_MANIFEST)
    parser.add_argument("--tokenizer", type=pathlib.Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--maximum-training-examples", type=int)
    parser.add_argument("--maximum-validation-examples", type=int)
    parser.add_argument("--maximum-testing-examples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if (
        not 1 <= args.threads <= 64
        or args.maximum_training_examples == 0
        or args.maximum_validation_examples == 0
        or args.maximum_testing_examples == 0
        or args.epochs == 0
    ):
        parser.error("threads and optional example/epoch counts must be positive")
    if any(value is not None and value < 0 for value in (
        args.maximum_training_examples, args.maximum_validation_examples,
        args.maximum_testing_examples, args.epochs,
    )):
        parser.error("optional example/epoch counts cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        report = train(parse_args(argv))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        ContextTrainingError,
        canonical_safetensors.CanonicalSafetensorsError,
        context_model_contract.ContextModelContractError,
        context_tokenizer_contract.ContextTokenizerContractError,
        model_sources.ModelSourceError,
        score_context_teacher.ContextTeacherError,
        OSError,
        subprocess.CalledProcessError,
    ) as failure:
        print(f"context training error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
