#!/usr/bin/env python3
"""Train LibreBoard's from-scratch layout-conditioned CTC swipe encoder."""

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
from collections.abc import Iterator
from typing import Any

import model_sources
import swipe_model_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "build" / "model-data" / "swipe-latin-v1"
DEFAULT_OUTPUT_ROOT = ROOT / "build" / "model-training" / "swipe-latin-v1"


class SwipeTrainingError(ValueError):
    pass


CHECKPOINT_METADATA_KEYS = {
    "schemaVersion",
    "appCommit",
    "dataManifestSha256",
    "modelSpecSha256",
    "trainingToolSha256",
    "modelImplementationSha256",
    "completedEpoch",
    "epochReports",
    "deviceType",
    "torchVersion",
    "threads",
}


def _dependencies():
    try:
        import torch
        import safetensors
        from safetensors.torch import save_file
        from torch import nn
        from torch.utils.data import DataLoader, IterableDataset
    except ImportError as failure:
        raise SwipeTrainingError(
            "install the pinned build-only dependencies from models/training/requirements-linux-x86_64.lock"
        ) from failure
    if str(torch.__version__).split("+", 1)[0] != "2.8.0":
        raise SwipeTrainingError(f"training requires torch 2.8.0, found {torch.__version__}")
    return torch, safetensors, save_file, nn, DataLoader, IterableDataset


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip())
    return commit, dirty


def _load_layout(data_root: pathlib.Path, spec: swipe_model_contract.SwipeModelSpec) -> tuple[list[float], list[float]]:
    try:
        value = json.loads((data_root / "layout.json").read_bytes())
    except (OSError, json.JSONDecodeError) as failure:
        raise SwipeTrainingError(f"cannot read prepared layout: {failure}") from failure
    architecture = spec.architecture
    if value.get("schemaVersion") != 1 or value.get("pathShape") != [1, architecture["pathPoints"], 2]:
        raise SwipeTrainingError("prepared layout has an incompatible path ABI")
    if value.get("keyCentersShape") != [1, architecture["keySlots"], 2]:
        raise SwipeTrainingError("prepared layout has an incompatible key-center ABI")
    if value.get("keyMaskShape") != [1, architecture["keySlots"]]:
        raise SwipeTrainingError("prepared layout has an incompatible key-mask ABI")
    centers = value.get("keyCenters")
    mask = value.get("keyMask")
    if (
        not isinstance(centers, list)
        or len(centers) != architecture["keySlots"] * 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in centers)
    ):
        raise SwipeTrainingError("prepared key centers are invalid")
    if not isinstance(mask, list) or len(mask) != architecture["keySlots"] or any(item not in {0, 1} for item in mask):
        raise SwipeTrainingError("prepared key mask is invalid")
    if not any(mask):
        raise SwipeTrainingError("prepared layout contains no enabled keys")
    return [float(item) for item in centers], [float(item) for item in mask]


def _parse_record(line: bytes, split: str, path_points: int, key_slots: int, output_frames: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        raise SwipeTrainingError(f"prepared {split} data contains invalid JSON: {failure}") from failure
    if not isinstance(value, dict) or value.get("schemaVersion") != 1 or value.get("split") != split:
        raise SwipeTrainingError(f"prepared {split} record has an incompatible schema or split")
    identifier = value.get("id")
    path = value.get("pathCoordinates")
    labels = value.get("ctcLabels")
    if not isinstance(identifier, str) or not model_sources.SHA256.fullmatch(identifier):
        raise SwipeTrainingError(f"prepared {split} record has an invalid id")
    if (
        not isinstance(path, list)
        or len(path) != path_points * 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in path)
    ):
        raise SwipeTrainingError(f"prepared {split} record has an invalid path tensor")
    if (
        not isinstance(labels, list)
        or not labels
        or any(isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= key_slots for item in labels)
    ):
        raise SwipeTrainingError(f"prepared {split} record has invalid CTC labels")
    required_frames = len(labels) + sum(labels[index] == labels[index - 1] for index in range(1, len(labels)))
    if required_frames > output_frames:
        raise SwipeTrainingError(f"prepared {split} record cannot align within the output frames")
    return {"id": identifier, "path": [float(item) for item in path], "labels": labels}


def _shuffled_records(
    path: pathlib.Path,
    *,
    split: str,
    seed: int,
    buffer_size: int,
    path_points: int,
    key_slots: int,
    output_frames: int,
) -> Iterator[dict[str, Any]]:
    randomizer = random.Random(seed)
    buffer: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line in stream:
            record = _parse_record(line, split, path_points, key_slots, output_frames)
            if len(buffer) < buffer_size:
                buffer.append(record)
                continue
            index = randomizer.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = record
    randomizer.shuffle(buffer)
    yield from buffer


def _collate(records, torch):
    paths = torch.tensor([record["path"] for record in records], dtype=torch.float32).reshape(len(records), 64, 2)
    lengths = torch.tensor([len(record["labels"]) for record in records], dtype=torch.long)
    targets = torch.tensor([label for record in records for label in record["labels"]], dtype=torch.long)
    return paths, targets, lengths


def _augment(paths, key_centers, key_mask, config: dict[str, Any], generator, torch):
    batch = paths.shape[0]
    scale = 1.0 + torch.randn((batch, 1, 1), generator=generator, device=paths.device) * config["geometryScaleStd"]
    translation = torch.randn((batch, 1, 2), generator=generator, device=paths.device) * config["geometryTranslationStd"]
    noise = torch.randn(paths.shape, generator=generator, device=paths.device) * config["pathNoiseStd"]
    augmented_paths = (paths - 0.5) * scale + 0.5 + translation + noise
    augmented_keys = (key_centers - 0.5) * scale + 0.5 + translation
    augmented_paths = augmented_paths.clamp(-0.5, 1.5)
    augmented_keys = augmented_keys.clamp(-0.5, 1.5) * key_mask.unsqueeze(-1)
    return augmented_paths, augmented_keys


def _collapse_ctc(classes: list[int]) -> list[int]:
    result = []
    previous = None
    for output_class in classes:
        if output_class != 0 and output_class != previous:
            result.append(output_class)
        previous = output_class
    return result


def _edit_distance(first: list[int], second: list[int]) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_value in enumerate(first, 1):
        current = [first_index]
        for second_index, second_value in enumerate(second, 1):
            current.append(min(
                current[-1] + 1,
                previous[second_index] + 1,
                previous[second_index - 1] + (first_value != second_value),
            ))
        previous = current
    return previous[-1]


def _evaluate(model, path, split, layout, spec, device, batch_size, maximum_steps, torch, DataLoader, IterableDataset):
    architecture = spec.architecture

    class EvaluationDataset(IterableDataset):
        def __iter__(self):
            yield from _shuffled_records(
                path,
                split=split,
                seed=spec.training["seed"],
                buffer_size=1,
                path_points=architecture["pathPoints"],
                key_slots=architecture["keySlots"],
                output_frames=architecture["outputFrames"],
            )

    loader = DataLoader(
        EvaluationDataset(),
        batch_size=batch_size,
        num_workers=0,
        collate_fn=lambda values: _collate(values, torch),
    )
    centers, mask = layout
    center_tensor = torch.tensor(centers, dtype=torch.float32, device=device).reshape(1, 64, 2)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).reshape(1, 64)
    exact = edits = target_characters = examples = 0
    model.eval()
    with torch.inference_mode():
        for step, (paths, targets, lengths) in enumerate(loader):
            if maximum_steps is not None and step >= maximum_steps:
                break
            paths = paths.to(device)
            batch = paths.shape[0]
            logits = model(paths, center_tensor.expand(batch, -1, -1), mask_tensor.expand(batch, -1))
            greedy = logits.argmax(dim=-1).cpu().tolist()
            offset = 0
            for predictions, length in zip(greedy, lengths.tolist(), strict=True):
                expected = targets[offset: offset + length].tolist()
                offset += length
                decoded = _collapse_ctc(predictions)
                exact += decoded == expected
                edits += _edit_distance(decoded, expected)
                target_characters += len(expected)
                examples += 1
    if examples == 0:
        raise SwipeTrainingError(f"{split} evaluation produced no examples")
    return {
        "examples": examples,
        "greedyExactAccuracy": exact / examples,
        "greedyCharacterErrorRate": edits / target_characters,
    }


def _checkpoint_bindings(
    *,
    app_commit: str,
    data_manifest_sha256: str,
    model_spec_sha256: str,
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
            ROOT / "models" / "training" / "swipe_model.py"
        ),
        "deviceType": device_type,
        "torchVersion": torch_version,
        "threads": str(threads),
    }


def _save_checkpoint(
    path: pathlib.Path,
    *,
    model,
    optimizer,
    epoch_reports: list[dict[str, Any]],
    bindings: dict[str, str],
    device,
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
            raise SwipeTrainingError(f"optimizer checkpoint state is incomplete for {name}")
        for state_name in ("step", "exp_avg", "exp_avg_sq"):
            value = state[state_name]
            if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                raise SwipeTrainingError(f"optimizer checkpoint state is invalid for {name}.{state_name}")
            tensors[f"optimizer.{name}.{state_name}"] = value.detach().cpu().contiguous()
    tensors["rng.cpu"] = torch.get_rng_state().cpu().contiguous()
    if device.type == "cuda":
        tensors["rng.cuda"] = torch.cuda.get_rng_state(device).cpu().contiguous()

    metadata = dict(bindings)
    metadata.update({
        "completedEpoch": str(len(epoch_reports)),
        "epochReports": json.dumps(epoch_reports, separators=(",", ":"), sort_keys=True),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    temporary.close()
    try:
        save_file(tensors, temporary_path, metadata=metadata)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _restore_checkpoint(
    path: pathlib.Path,
    *,
    model,
    optimizer,
    expected_bindings: dict[str, str],
    device,
    torch,
) -> tuple[int, list[dict[str, Any]], str]:
    try:
        from safetensors import safe_open

        if not path.is_file() or path.is_symlink() or path.stat().st_size > 128 * 1024 * 1024:
            raise SwipeTrainingError("training checkpoint is missing, linked, or too large")
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata()
            tensor_names = set(checkpoint.keys())
            tensors = {name: checkpoint.get_tensor(name) for name in tensor_names}
    except SwipeTrainingError:
        raise
    except Exception as failure:
        raise SwipeTrainingError(f"cannot read training checkpoint: {failure}") from failure
    if not isinstance(metadata, dict) or set(metadata) != CHECKPOINT_METADATA_KEYS:
        raise SwipeTrainingError("training checkpoint metadata has an unexpected schema")
    for key, value in expected_bindings.items():
        if key == "appCommit":
            continue
        if metadata.get(key) != value:
            raise SwipeTrainingError(f"training checkpoint binding does not match: {key}")
    app_commit = metadata.get("appCommit", "")
    if len(app_commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in app_commit):
        raise SwipeTrainingError("training checkpoint has an invalid app commit")
    try:
        completed_epoch = int(metadata["completedEpoch"])
        epoch_reports = json.loads(metadata["epochReports"])
    except (TypeError, ValueError, json.JSONDecodeError) as failure:
        raise SwipeTrainingError("training checkpoint has invalid progress metadata") from failure
    if (
        not isinstance(epoch_reports, list)
        or completed_epoch != len(epoch_reports)
        or completed_epoch <= 0
    ):
        raise SwipeTrainingError("training checkpoint epoch reports do not match its progress")

    model_state = model.state_dict()
    expected_tensors = {f"model.{name}" for name in model_state}
    for name, _parameter in model.named_parameters():
        expected_tensors.update({
            f"optimizer.{name}.step",
            f"optimizer.{name}.exp_avg",
            f"optimizer.{name}.exp_avg_sq",
        })
    expected_tensors.add("rng.cpu")
    if device.type == "cuda":
        expected_tensors.add("rng.cuda")
    if tensor_names != expected_tensors:
        raise SwipeTrainingError("training checkpoint tensor inventory does not match the model")

    try:
        model.load_state_dict(
            {name: tensors[f"model.{name}"] for name in model_state},
            strict=True,
        )
        optimizer.state.clear()
        for name, parameter in model.named_parameters():
            optimizer.state[parameter] = {
                "step": tensors[f"optimizer.{name}.step"].cpu(),
                "exp_avg": tensors[f"optimizer.{name}.exp_avg"].to(device),
                "exp_avg_sq": tensors[f"optimizer.{name}.exp_avg_sq"].to(device),
            }
        torch.set_rng_state(tensors["rng.cpu"].to(dtype=torch.uint8, device="cpu"))
        if device.type == "cuda":
            torch.cuda.set_rng_state(tensors["rng.cuda"].to(dtype=torch.uint8, device="cpu"), device)
    except (KeyError, RuntimeError, ValueError) as failure:
        raise SwipeTrainingError(f"training checkpoint state is incompatible: {failure}") from failure
    return completed_epoch, epoch_reports, app_commit


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, safetensors, save_file, nn, DataLoader, IterableDataset = _dependencies()
    spec = swipe_model_contract.load_spec(args.spec)
    data_root = args.data_root.resolve()
    release_attempt = args.max_train_steps is None and args.max_validation_steps is None
    data_manifest = swipe_model_contract.load_prepared_manifest(data_root, require_pinned=release_attempt)
    layout = _load_layout(data_root, spec)
    commit, dirty = _git_state()
    if dirty and not args.allow_dirty:
        raise SwipeTrainingError("commit or stash the worktree before a release-eligible training run")

    sys.path.insert(0, str(ROOT))
    from models.training.swipe_model import SwipeCtcModel, trainable_parameter_count

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SwipeTrainingError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(spec.training["seed"])
    random.seed(spec.training["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)

    model = SwipeCtcModel(spec.architecture).to(device)
    actual_parameters = trainable_parameter_count(model)
    if actual_parameters != spec.raw["parameterCount"]:
        raise SwipeTrainingError(
            f"implemented parameter count {actual_parameters} does not match spec {spec.raw['parameterCount']}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.training["learningRate"],
        weight_decay=spec.training["weightDecay"],
    )
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    centers, mask = layout
    center_tensor = torch.tensor(centers, dtype=torch.float32, device=device).reshape(1, 64, 2)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).reshape(1, 64)

    architecture = spec.architecture

    class TrainingDataset(IterableDataset):
        def __init__(self):
            super().__init__()
            self.epoch = 0

        def __iter__(self):
            yield from _shuffled_records(
                data_root / "train.jsonl",
                split="train",
                seed=spec.training["seed"] + self.epoch,
                buffer_size=spec.training["shuffleBuffer"],
                path_points=architecture["pathPoints"],
                key_slots=architecture["keySlots"],
                output_frames=architecture["outputFrames"],
            )

    dataset = TrainingDataset()
    loader = DataLoader(
        dataset,
        batch_size=spec.training["batchSize"],
        num_workers=0,
        collate_fn=lambda values: _collate(values, torch),
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    checkpoint_path = (
        resume_checkpoint.resolve()
        if resume_checkpoint is not None
        else output_root / "checkpoint.safetensors"
    )
    if resume_checkpoint is not None and not release_attempt:
        raise SwipeTrainingError("checkpoint resume is available only for complete release training")
    if release_attempt and resume_checkpoint is None and checkpoint_path.exists():
        raise SwipeTrainingError("release checkpoint already exists; pass --resume-checkpoint or use a clean output")
    data_manifest_sha256 = model_sources.file_sha256(data_root / "split-manifest.json")
    bindings = _checkpoint_bindings(
        app_commit=commit,
        data_manifest_sha256=data_manifest_sha256,
        model_spec_sha256=spec.sha256,
        device_type=device.type,
        torch_version=str(torch.__version__),
        threads=args.threads,
    )
    epoch_reports: list[dict[str, Any]] = []
    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch, epoch_reports, commit = _restore_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            expected_bindings=bindings,
            device=device,
            torch=torch,
        )
        if start_epoch > spec.training["epochs"]:
            raise SwipeTrainingError("training checkpoint exceeds the configured epoch count")
    for epoch in range(start_epoch, spec.training["epochs"]):
        dataset.epoch = epoch
        generator = torch.Generator(device=device).manual_seed(spec.training["seed"] + epoch)
        model.train()
        total_loss = 0.0
        examples = steps = 0
        for step, (paths, targets, target_lengths) in enumerate(loader):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            paths = paths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            batch = paths.shape[0]
            keys = center_tensor.expand(batch, -1, -1)
            key_mask = mask_tensor.expand(batch, -1)
            paths, keys = _augment(paths, keys, key_mask, spec.training, generator, torch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(paths, keys, key_mask)
            log_probabilities = logits.log_softmax(dim=-1).transpose(0, 1)
            input_lengths = torch.full(
                (batch,), architecture["outputFrames"], dtype=torch.long, device=device
            )
            loss = loss_function(log_probabilities, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise SwipeTrainingError(f"non-finite training loss at epoch {epoch + 1}, step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.training["gradientClip"])
            optimizer.step()
            total_loss += loss.item() * batch
            examples += batch
            steps += 1
        if examples == 0:
            raise SwipeTrainingError("training produced no examples")
        validation = _evaluate(
            model,
            data_root / "validation.jsonl",
            "validation",
            layout,
            spec,
            device,
            spec.training["batchSize"],
            args.max_validation_steps,
            torch,
            DataLoader,
            IterableDataset,
        )
        epoch_reports.append({
            "epoch": epoch + 1,
            "examples": examples,
            "steps": steps,
            "meanCtcLoss": total_loss / examples,
            "validation": validation,
        })
        if release_attempt:
            checkpoint_bindings = dict(bindings)
            checkpoint_bindings["appCommit"] = commit
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch_reports=epoch_reports,
                bindings=checkpoint_bindings,
                device=device,
                torch=torch,
                save_file=save_file,
            )
        print(json.dumps(epoch_reports[-1], sort_keys=True), flush=True)
        if args.max_train_steps is not None:
            break

    model = model.to("cpu").eval()
    development = args.max_train_steps is not None or args.max_validation_steps is not None or dirty
    weights_name = "swipe-latin-v1-development.safetensors" if development else "swipe-latin-v1.weights.safetensors"
    weights_path = output_root / weights_name
    state = {name: tensor.detach().contiguous() for name, tensor in sorted(model.state_dict().items())}
    metadata = {
        "appCommit": commit,
        "dataManifestSha256": model_sources.file_sha256(data_root / "split-manifest.json"),
        "modelSpecSha256": spec.sha256,
    }
    temporary = tempfile.NamedTemporaryFile(dir=output_root, suffix=".safetensors", delete=False)
    temporary_path = pathlib.Path(temporary.name)
    temporary.close()
    try:
        save_file(state, temporary_path, metadata=metadata)
        os.replace(temporary_path, weights_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report = {
        "schemaVersion": 1,
        "modelId": spec.raw["modelId"],
        "appCommit": commit,
        "releaseEligible": not development,
        "parameterCount": actual_parameters,
        "modelSpecSha256": spec.sha256,
        "dataManifestSha256": metadata["dataManifestSha256"],
        "weights": {
            "file": weights_name,
            "bytes": weights_path.stat().st_size,
            "sha256": model_sources.file_sha256(weights_path),
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
    }
    report_name = "training-report-development.json" if development else "training-report.json"
    (output_root / report_name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(data_root / "split-manifest.json", output_root / "split-manifest.json")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=swipe_model_contract.DEFAULT_SPEC)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-validation-steps", type=int)
    parser.add_argument("--resume-checkpoint", type=pathlib.Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.max_train_steps == 0 or args.max_validation_steps == 0:
        parser.error("thread and optional step counts must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        report = train(parse_args(argv))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (SwipeTrainingError, swipe_model_contract.SwipeModelContractError) as failure:
        print(f"swipe training error: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
