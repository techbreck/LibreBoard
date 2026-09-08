#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Measure exported INT4 context rankings on provenance-bound held-out slates.

Host distillation diagnostics only: never substitutes for real tap errors, complete candidate
fusion, or Android latency/memory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
from collections import defaultdict

import audit_context_prompt_overlap
import audit_joint_model_splits
import context_model_contract
import context_tokenizer_contract
import evaluate_swipe_ctc
import export_context_model
import export_swipe_model
import model_sources
import train_context_model


class ContextEvaluationError(ValueError):
    pass


def _artifact(root, metadata, maximum_bytes, label):
    if not isinstance(metadata, dict) or set(metadata) != {"file", "bytes", "sha256"}:
        raise ContextEvaluationError(f"invalid {label} metadata")
    size, digest = metadata["bytes"], metadata["sha256"]
    if (isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= maximum_bytes
            or not isinstance(digest, str) or not model_sources.SHA256.fullmatch(digest)):
        raise ContextEvaluationError(f"invalid {label} size/hash")
    path = evaluate_swipe_ctc._safe_child(root, metadata["file"], label)
    if path.stat().st_size != size or model_sources.file_sha256(path) != digest:
        raise ContextEvaluationError(f"{label} does not match its export report")
    return path


def load_export(path, spec, development):
    report, payload = evaluate_swipe_ctc._read_json(path, 4 * 1024 * 1024, "context export report")
    if (report.get("schemaVersion") != 1 or report.get("modelId") != "context-en-de-v1"
            or report.get("modelSpecSha256") != spec.sha256
            or not isinstance(report.get("releaseEligible"), bool)):
        raise ContextEvaluationError("incompatible context export identity")
    if not development and report["releaseEligible"] is not True:
        raise ContextEvaluationError("a development export requires --development")
    for field in ("dataManifestSha256", "trainingReportSha256", "weightsSha256"):
        if not isinstance(report.get(field), str) or not model_sources.SHA256.fullmatch(report[field]):
            raise ContextEvaluationError(f"invalid export {field}")
    if not isinstance(report.get("appCommit"), str) or not model_sources.REVISION.fullmatch(report["appCommit"]):
        raise ContextEvaluationError("invalid export app commit")
    model = _artifact(path.parent, report.get("model"), spec.export["maximumModelBytes"], "context model")
    tokenizer = _artifact(path.parent, report.get("tokenizer"), 2 * 1024 * 1024, "context tokenizer")
    training_name = "training-report.json" if report["releaseEligible"] else "training-report-development.json"
    training_path = evaluate_swipe_ctc._safe_child(path.parent, training_name, "training report")
    if model_sources.file_sha256(training_path) != report["trainingReportSha256"]:
        raise ContextEvaluationError("training report hash differs from export")
    training = export_context_model._load_training_report(training_path, development, spec)
    if (training["appCommit"] != report["appCommit"]
            or training["dataManifestSha256"] != report["dataManifestSha256"]
            or training["weights"]["sha256"] != report["weightsSha256"]
            or training["tokenizer"] != report["tokenizer"]):
        raise ContextEvaluationError("export and training provenance disagree")
    return report, model, tokenizer, hashlib.sha256(payload).hexdigest()


def ranking_metrics(scores, teacher_scores):
    """Candidate zero is the observed label, never an input ranking advantage."""
    if (len(scores) < 2 or len(scores) != len(teacher_scores)
            or any(not math.isfinite(float(x)) for x in (*scores, *teacher_scores))):
        raise ContextEvaluationError("invalid candidate scores")
    # Rank ties against the observed candidate: a constant model must not score 100% merely
    # because the preparation contract always places that candidate first.
    better_or_equal = sum(value >= scores[0] for value in scores[1:])
    winners = [i for i, value in enumerate(scores) if value == max(scores)]
    teacher_winners = [i for i, value in enumerate(teacher_scores) if value == max(teacher_scores)]
    return {
        "examples": 1,
        "observedTop1": int(better_or_equal == 0),
        "observedTop3": int(better_or_equal < 3),
        "teacherTop1Agreement": int(len(winners) == len(teacher_winners) == 1 and winners == teacher_winners),
        "tiedTop1": int(len(winners) != 1),
    }


def bind_prompt_exclusions(records, source_ids, policy_sha256):
    """Reconstruct teacher identifiers; never guess a scored row from textual similarity."""
    remaining = set(source_ids)
    matches = {}
    for record in records:
        for source_id in tuple(remaining):
            binding = {
                "sourceRecordId": source_id, "prefixTokenIds": record["prefixTokenIds"],
                "candidates": [candidate["normalized"] for candidate in record["candidates"]],
                "policySha256": policy_sha256,
            }
            identifier = hashlib.sha256(train_context_model.score_context_teacher._canonical_line(binding)).hexdigest()
            if identifier == record["id"]:
                matches[source_id] = identifier
                remaining.remove(source_id)
                break
    if remaining:
        raise ContextEvaluationError("prompt exclusions could not be bound to every scored record")
    return matches


def metric_report(counts):
    metrics = {}
    for group, values in sorted(counts.items()):
        metrics[group] = dict(values)
        for name in ("observedTop1", "observedTop3", "teacherTop1Agreement"):
            metrics[group][name + "Rate"] = values[name] / values["examples"]
    return metrics


def evaluate(args):
    if args.maximum_examples is not None and (not args.development or args.maximum_examples < 1):
        raise ContextEvaluationError("a positive --maximum-examples requires --development")
    spec = context_model_contract.load_spec()
    report, model_path, tokenizer_path, report_hash = load_export(args.export_report, spec, args.development)
    root = args.scored_root
    manifest_path = args.distillation_manifest or root / (
        "distillation-manifest-development.json" if args.development else "distillation-manifest.json"
    )
    manifest, manifest_hash = train_context_model._load_manifest(
        manifest_path, scored_root=root, development=args.development,
    )
    if manifest_hash != report["dataManifestSha256"]:
        raise ContextEvaluationError("evaluation data differs from trained corpus")
    prompt_corpus = getattr(args, "prompt_audit_corpus_manifest", None)
    prompt_root = getattr(args, "prompt_audit_data_root", None)
    if bool(prompt_corpus) != bool(prompt_root):
        raise ContextEvaluationError("prompt audit requires both corpus manifest and prepared data root")
    prompt_audit = None
    prompt_excluded = set()
    if prompt_corpus is not None:
        prompt_audit = audit_context_prompt_overlap.run(prompt_corpus, prompt_root)
        if prompt_audit["contextCorpusManifestSha256"] != manifest["dataManifestSha256"]:
            raise ContextEvaluationError("prompt audit corpus differs from the trained distillation corpus")
        mappings = bind_prompt_exclusions(
            audit_joint_model_splits.verified_rows(
                root / f"{args.split}.scored.jsonl", manifest["outputs"][f"{args.split}.scored.jsonl"], args.split,
            ), prompt_audit["excludedIds"][args.split], manifest["distillationPolicySha256"],
        )
        prompt_excluded = set(mappings.values())
    tokenizer = context_tokenizer_contract.load_tokenizer(
        tokenizer_path, expected_vocabulary_size=spec.architecture["vocabularySize"],
    )
    if tokenizer.sha256 != manifest["tokenizerSha256"]:
        raise ContextEvaluationError("evaluation tokenizer differs from trained corpus")
    torch, numpy, onnx, ort, _, _, _, _, versions = export_context_model._dependencies()
    if report.get("toolchain") != versions:
        raise ContextEvaluationError("export toolchain differs from the pinned evaluator toolchain")
    torch.set_num_threads(1)
    graph = onnx.load(model_path, load_external_data=False)
    export_swipe_model._reject_external_data(graph, onnx)
    export_context_model._validate_tensor_abi(graph, spec, onnx)
    onnx.checker.check_model(graph, full_check=True)
    required = export_swipe_model.required_operators(graph, onnx)
    custom = export_swipe_model._canonical_operator_list(required)
    if {name for name in custom if "::" in name} != export_context_model.EXPECTED_CUSTOM_OPERATORS:
        raise ContextEvaluationError("unexpected context custom operators")
    custom_counts = {
        name: sum(f"{node.domain}::{node.op_type}" == name for node in graph.graph.node)
        for name in export_context_model.EXPECTED_CUSTOM_OPERATORS
    }
    if (custom_counts != {"com.microsoft::GatherBlockQuantized": 4, "com.microsoft::MatMulNBits": 56}
            or custom_counts != report.get("customOperatorCounts")):
        raise ContextEvaluationError("context custom operator counts differ from the export contract")
    del graph
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ContextEvaluationError("context diagnostic requires CPU-only inference")
    policy = train_context_model.score_context_teacher.load_policy()
    parser = {
        "vocabulary_size": spec.architecture["vocabularySize"],
        "maximum_prefix_tokens": policy.maximum_prefix_student_tokens,
        "maximum_candidate_tokens": spec.architecture["maximumCandidateTokens"],
        "maximum_candidates": policy.maximum_candidates,
    }
    details = manifest["outputs"][f"{args.split}.scored.jsonl"]
    counts = defaultdict(lambda: defaultdict(int))
    times = []
    prompt_counts = defaultdict(lambda: defaultdict(int))
    excluded_rows_seen = 0
    for index, record in enumerate(train_context_model._records(
        root / f"{args.split}.scored.jsonl", split=args.split, expected_records=details["records"],
        seed=spec.training["seed"], shuffle_buffer=1, parser_arguments=parser,
    )):
        if args.maximum_examples is not None and index >= args.maximum_examples:
            break
        inputs = train_context_model._model_inputs(
            record, tokenizer, spec, torch, torch.device("cpu"), len(record["candidates"]),
        )
        feed = {name: value.numpy() for name, value in zip(
            ("input_ids", "attention_mask", "candidate_mask", "field_class"), inputs[:4],
        )}
        start = time.perf_counter_ns()
        scores = session.run(["candidate_log_likelihood"], feed)[0]
        times.append((time.perf_counter_ns() - start) / 1_000_000)
        if scores.shape != (len(record["candidates"]),) or not numpy.isfinite(scores).all():
            raise ContextEvaluationError("context runtime returned invalid scores")
        metrics = ranking_metrics(scores.tolist(), [c["teacherMeanLogProbability"] for c in record["candidates"]])
        for group in ("overall", record["language"]):
            for key, value in metrics.items():
                counts[group][key] += value
                if prompt_audit is not None and record["id"] not in prompt_excluded:
                    prompt_counts[group][key] += value
        excluded_rows_seen += record["id"] in prompt_excluded
        if (index + 1) % 250 == 0:
            print(f"evaluated {index + 1}/{details['records']} context slates", file=sys.stderr, flush=True)
    if not times:
        raise ContextEvaluationError("context evaluation has no examples")
    metrics = metric_report(counts)
    result = {
        "schemaVersion": 1, "diagnosticOnly": True, "satisfiesPhase0": False,
        "development": args.development, "split": args.split,
        "maximumExamples": args.maximum_examples, "metrics": metrics,
        "latencyMs": {"hostInferenceOnly": True, **evaluate_swipe_ctc._percentiles(times)},
        "artifacts": {"exportReportSha256": report_hash, "modelSha256": report["model"]["sha256"],
                      "tokenizerSha256": tokenizer.sha256, "distillationManifestSha256": manifest_hash,
                      "evaluationDataSha256": details["sha256"],
                      "inputBuilderToolSha256": model_sources.file_sha256(pathlib.Path(train_context_model.__file__)),
                      "evaluationToolSha256": model_sources.file_sha256(pathlib.Path(__file__))},
        "toolchain": versions, "threads": args.threads,
        "limitations": ["Distillation slates are not human tap-error evidence.",
                        "Static, spatial, personal, commit policy and full fusion are not measured.",
                        "Host inference timing is not Android latency or memory evidence.",
                        "Top-rank ties are counted against the observed label."],
    }

    if prompt_audit is not None:
        result["promptDisjointDiagnostic"] = {
            "diagnosticOnly": True, "contextCorpusManifestSha256": prompt_audit["contextCorpusManifestSha256"],
            "auditToolSha256": prompt_audit["toolSha256"], "normalization": prompt_audit["normalization"],
            "excludedScoredIds": sorted(prompt_excluded), "excludedRowsSeen": excluded_rows_seen,
            "metrics": metric_report(prompt_counts),
            "limitations": prompt_audit["limitations"],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=pathlib.Path, default=export_context_model.DEFAULT_OUTPUT_ROOT / "export-report.json")
    parser.add_argument("--scored-root", type=pathlib.Path, default=train_context_model.DEFAULT_SCORED_ROOT)
    parser.add_argument("--distillation-manifest", type=pathlib.Path)
    parser.add_argument("--prompt-audit-corpus-manifest", type=pathlib.Path)
    parser.add_argument("--prompt-audit-data-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=1)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--maximum-examples", type=int)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args)
        evaluate_swipe_ctc._write_report(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ContextEvaluationError, train_context_model.ContextTrainingError,
            export_context_model.ContextExportError, export_swipe_model.SwipeExportError,
            evaluate_swipe_ctc.SwipeEvaluationError, context_model_contract.ContextModelContractError,
            context_tokenizer_contract.ContextTokenizerContractError, model_sources.ModelSourceError,
            OSError, ValueError) as failure:
        print(f"context evaluation failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
