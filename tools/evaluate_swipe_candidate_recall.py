#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Diagnostic candidate-recall experiments for the frozen swipe validation slate.

Fits OOV score calibration on training paths (targets are never returned), injects
calibrated reserved-slot n-best CTC spellings, applies stratum-adaptive CTC/geometry
budgets, and measures beam-256 recall. Production vocabulary, scoring, and safety
policy are not changed. Results are candidate-membership bounds, not quality wins.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import sys
import time
from typing import Any

import evaluate_swipe_ctc as evaluator
import model_sources
import swipe_model_contract


ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN_SLATES_SHA256 = "fbbbcd0ef004f651a2e6c698ce6d834d4d2ca8d5f8195a6c9996e5ff2c61776e"
DEFAULT_SLATES = ROOT / "build" / "reports" / "swipe-validation-context-base-6000.jsonl"
DEFAULT_LEXICON = ROOT / "build" / "device-evidence" / "static-swipe-lexicon.json"
DEFAULT_APK = ROOT / "build" / "device-evidence" / "static-vocabulary-origin.apk"
DEFAULT_OFFENSIVE = (
    ROOT / "build" / "device-evidence" / "full-static-vocabulary-v1" / "static-swipe-lexicon-full-diagnostic.json"
)
DEFAULT_EXPORT = ROOT / "build" / "model-export" / "swipe-latin-v1" / "export-report.json"
DEFAULT_DATA = ROOT / "build" / "model-data" / "swipe-latin-v1"


def sha256_file(path: pathlib.Path) -> str:
    return model_sources.file_sha256(path)


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_frozen_slates(path: pathlib.Path) -> bytes:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FROZEN_SLATES_SHA256:
        raise evaluator.SwipeEvaluationError(
            f"frozen slates hash {digest} does not match {FROZEN_SLATES_SHA256}"
        )
    return payload


def scored(items: list[dict[str, Any]]) -> list[evaluator.ScoredLexiconEntry]:
    return [
        evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry(item["word"], item["language"], (), item["frequency"]),
            float(item["spatial"]),
        )
        for item in items
    ]


def metric_tables(rows: list[dict[str, Any]], present: list[bool], ranked: list[list[str]]):
    names = ("overall",) + evaluator.REQUIRED_STRATA
    membership = {name: {"rows": 0, "targetPresent": 0} for name in names}
    ranking = {name: {"rows": 0, "top1": 0, "top3": 0} for name in names}
    for row, hit, words in zip(rows, present, ranked, strict=True):
        groups = ("overall", *row["strata"])
        for name in groups:
            membership[name]["rows"] += 1
            membership[name]["targetPresent"] += bool(hit)
            ranking[name]["rows"] += 1
            ranking[name]["top1"] += bool(words and words[0] == row["target"])
            ranking[name]["top3"] += row["target"] in words[:3]
    def decorate(table):
        decorated = {}
        for name, counts in table.items():
            rows_n = counts["rows"]
            extra = {}
            if "targetPresent" in counts:
                extra["recall"] = counts["targetPresent"] / rows_n if rows_n else 0.0
            else:
                extra["top1Accuracy"] = counts["top1"] / rows_n if rows_n else 0.0
                extra["top3Accuracy"] = counts["top3"] / rows_n if rows_n else 0.0
            decorated[name] = {**counts, **extra}
        return decorated
    return decorate(membership), decorate(ranking)


def open_session(export_report: pathlib.Path):
    export, model_path, export_hash = evaluator.load_export(export_report, development=False)
    numpy, onnxruntime, versions = evaluator._dependencies()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    return export, model_path, export_hash, numpy, versions, session


def fit_calibration(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    layout = evaluator.load_layout(data_root / "layout.json")
    paths = evaluator.load_train_paths(
        data_root / "train.jsonl", layout, maximum_rows=args.calibration_rows,
    )
    lexicon, _provenance = evaluator.load_dictionary_lexicon(args.dictionary_lexicon, args.dictionary_apk, layout)
    lexicon_words = {entry.word for entries in lexicon.values() for entry in entries}
    export, model_path, export_hash, numpy, versions, session = open_session(args.export_report)
    key_centers = numpy.asarray(layout["keyCenters"], dtype=numpy.float32).reshape(1, 64, 2)
    key_mask = numpy.asarray(layout["keyMask"], dtype=numpy.float32).reshape(1, 64)
    trie_cache: dict[tuple[str, int], evaluator.TrieNode] = {}
    pairs: list[tuple[float, float]] = []
    gaps: list[float] = []
    for index, (_identifier, language, path) in enumerate(paths, 1):
        output = session.run(["logits"], {
            "path_coordinates": numpy.asarray(path, dtype=numpy.float32).reshape(1, 64, 2),
            "key_centers": key_centers,
            "key_mask": key_mask,
        })[0]
        logits = output[0].tolist()
        greedy = evaluator.collapse_greedy(logits)
        approximate_length = len(greedy) or 1
        cache_key = (language, approximate_length)
        node = trie_cache.get(cache_key)
        if node is None:
            node = evaluator.build_trie(lexicon.get(language, ()), approximate_length)
            trie_cache[cache_key] = node
        decoded = evaluator.prefix_beam_decode_scored(logits, node)
        spelling = evaluator.emissions_to_spelling(greedy, layout)
        observed, gap = evaluator.collect_oov_calibration_observations(
            logits, decoded, greedy, spelling, lexicon_words,
        )
        pairs.extend(observed)
        if gap is not None:
            gaps.append(gap)
        if index % 64 == 0:
            print(f"calibration {index}/{len(paths)}", file=sys.stderr, flush=True)
    forwards = [pair[0] for pair in pairs]
    spatials = [pair[1] for pair in pairs]
    calibration = evaluator.fit_oov_score_calibration(forwards, spatials, gaps)
    report = evaluator.oov_score_calibration_report(calibration)
    report.update({
        "fitSplit": "train",
        "fitRows": len(paths),
        "fitPathCount": len(paths),
        "modelSha256": export["model"]["sha256"],
        "exportReportSha256": export_hash,
        "toolSha256": sha256_file(pathlib.Path(__file__)),
        "evaluatorSha256": sha256_file(pathlib.Path(evaluator.__file__)),
        "slatesSha256": FROZEN_SLATES_SHA256,
        "toolchain": versions,
    })
    write_json(args.output, report)
    return report


def ranking_1000(args: argparse.Namespace) -> dict[str, Any]:
    argv = [
        "--split", "validation",
        "--development",
        "--sample-count", "1000",
        "--minimum-per-stratum", "100",
        "--dictionary-lexicon", str(args.dictionary_lexicon),
        "--dictionary-apk", str(args.dictionary_apk),
        "--export-report", str(args.export_report),
        "--data-root", str(args.data_root),
        "--output", str(args.output),
        "--reserved-oov-nbest", str(args.reserved_oov_nbest),
        "--oov-calibration", str(args.oov_calibration),
        "--known-offensive-lexicon", str(args.known_offensive_lexicon),
        "--beam-width", str(args.beam_width),
    ]
    if args.stratum_adaptive_merge:
        argv.append("--stratum-adaptive-merge")
    parsed = evaluator.parse_args(argv)
    report = evaluator.evaluate(parsed)
    evaluator._write_report(args.output, report)
    return report


def recall_6000(args: argparse.Namespace) -> dict[str, Any]:
    slates_payload = verify_frozen_slates(args.slates)
    slates = [json.loads(line) for line in slates_payload.splitlines()]
    if len(slates) != 6000:
        raise evaluator.SwipeEvaluationError("frozen slate count is not 6,000")
    data_root = args.data_root.resolve()
    layout = evaluator.load_layout(data_root / "layout.json")
    rows = evaluator.select_rows(
        evaluator.load_test_rows(data_root / "validation.jsonl", layout, "validation"),
        sample_count=6000,
        minimum_per_stratum=600,
    )
    by_id = {row.identifier: row for row in rows}
    calibration = evaluator.load_oov_score_calibration(args.oov_calibration)
    known_offensive = evaluator.load_known_offensive_words(args.known_offensive_lexicon)
    lexicon, _provenance = evaluator.load_dictionary_lexicon(args.dictionary_lexicon, args.dictionary_apk, layout)
    lexicon_by_language = {
        language: {evaluator._normalize(entry.word): entry for entry in entries}
        for language, entries in lexicon.items()
    }
    export, model_path, export_hash, numpy, versions, session = open_session(args.export_report)
    key_centers = numpy.asarray(layout["keyCenters"], dtype=numpy.float32).reshape(1, 64, 2)
    key_mask = numpy.asarray(layout["keyMask"], dtype=numpy.float32).reshape(1, 64)
    present = []
    ranked_words = []
    baseline_present = []
    baseline_ranked = []
    added = 0
    rejected = 0
    lost_at_append = 0
    inference_ms = []
    decode_ms = []
    for index, slate in enumerate(slates, 1):
        row = by_id[slate["id"]]
        if slate["sessionId"] != row.session_id or slate["target"] != row.target:
            raise evaluator.SwipeEvaluationError("frozen slate identity does not match selected validation row")
        started = time.perf_counter_ns()
        output = session.run(["logits"], {
            "path_coordinates": numpy.asarray(row.path, dtype=numpy.float32).reshape(1, 64, 2),
            "key_centers": key_centers,
            "key_mask": key_mask,
        })[0]
        inferred = time.perf_counter_ns()
        logits = output[0].tolist()
        ctc_n, geo_n = (
            evaluator.decoder_slate_budgets(row.strata)
            if args.stratum_adaptive_merge else (32, 32)
        )
        merged = evaluator.merge_swipe_slates(scored(slate["ctc"][:ctc_n]), scored(slate["geometric"][:geo_n]))
        reserved, offensive = evaluator.reserved_oov_from_logits(
            logits,
            layout,
            calibration=calibration,
            n_best=args.reserved_oov_nbest,
            beam_width=args.oov_beam_width,
            existing=merged,
            known_offensive=known_offensive,
            lexicon_by_word=lexicon_by_language.get(row.language, {}),
            language=row.language,
            ctc_spatials=[item["spatial"] for item in slate["ctc"]],
            include_lexicon_neighbors=True,
        )
        rejected += offensive
        dropped = evaluator.reserved_from_decoder_slates(
            scored(slate["ctc"]), scored(slate["geometric"]), merged,
        )
        reserved = evaluator.competing_slate(reserved, dropped)
        added += len(reserved)
        competing = evaluator.competing_slate(merged, reserved)
        if row.target in {item.word for item in merged} and row.target not in {item.word for item in competing}:
            lost_at_append += 1
        finished = time.perf_counter_ns()
        inference_ms.append((inferred - started) / 1_000_000)
        decode_ms.append((finished - inferred) / 1_000_000)
        present.append(row.target in {item.word for item in competing})
        ranked_words.append([entry.word for entry in evaluator.rank_static_fusion_with_reserved(merged, reserved)])
        baseline_present.append(row.target in {item["word"] for item in slate["merged"]})
        baseline_ranked.append(
            [entry.word for entry in evaluator.rank_static_fusion(scored(slate["merged"]))]
        )
        if index % 250 == 0:
            print(f"recall {index}/6000", file=sys.stderr, flush=True)
    after = hashlib.sha256(args.slates.read_bytes()).hexdigest()
    if after != FROZEN_SLATES_SHA256:
        raise evaluator.SwipeEvaluationError("frozen slates hash changed during the diagnostic")
    membership, ranking = metric_tables(slates, present, ranked_words)
    baseline_membership, baseline_ranking = metric_tables(slates, baseline_present, baseline_ranked)
    report = {
        "schemaVersion": 1,
        "diagnosticOnly": True,
        "releaseEligible": False,
        "slatesSha256": FROZEN_SLATES_SHA256,
        "slatesSha256After": after,
        "modelSha256": export["model"]["sha256"],
        "exportReportSha256": export_hash,
        "toolSha256": sha256_file(pathlib.Path(__file__)),
        "evaluatorSha256": sha256_file(pathlib.Path(evaluator.__file__)),
        "knownOffensiveSourceSha256": sha256_file(args.known_offensive_lexicon),
        "oovCalibrationSha256": sha256_file(args.oov_calibration),
        "reservedOovNBest": args.reserved_oov_nbest,
        "oovBeamWidth": args.oov_beam_width,
        "stratumAdaptiveMerge": bool(args.stratum_adaptive_merge),
        "counts": {
            "rows": 6000,
            "baselineTargetPresent": baseline_membership["overall"]["targetPresent"],
            "proposedTargetPresent": membership["overall"]["targetPresent"],
            "newTargetsRecovered": membership["overall"]["targetPresent"] - baseline_membership["overall"]["targetPresent"],
            "targetsLostAtAppend": lost_at_append,
            "reservedCandidatesAdded": added,
            "knownOffensiveRejected": rejected,
            "recallCountsCompetingSlateOnly": True,
        },
        "candidateRecall": membership,
        "baselineCandidateRecall": baseline_membership,
        "ranking": ranking,
        "baselineRanking": baseline_ranking,
        "latencyMs": {
            "hostDiagnosticOnly": True,
            "inference": evaluator._percentiles(inference_ms),
            "reservedOov": evaluator._percentiles(decode_ms),
            "reservedOovP95BudgetMs": 91.0,
            "reservedOovP95WithinBudget": evaluator._percentiles(decode_ms)["p95"] < 91.0,
        },
        "toolchain": versions,
        "limitations": [
            "Candidate recall is membership in the ranking-competing slate (merged union reserved), not an unbounded bag.",
            "Every reserved candidate counted for recall is passed to rank_static_fusion_with_reserved.",
            "Frequency-free OOV compete at the median lexicon spatial; no fabricated frequency prior.",
            "Production vocabulary, scoring, and safety policy were not changed.",
            "Host timing is not Android device latency.",
        ],
    }
    write_json(args.output, report)
    return report


def beam256_recall(args: argparse.Namespace) -> dict[str, Any]:
    verify_frozen_slates(args.slates)
    data_root = args.data_root.resolve()
    layout = evaluator.load_layout(data_root / "layout.json")
    rows = evaluator.select_rows(
        evaluator.load_test_rows(data_root / "validation.jsonl", layout, "validation"),
        sample_count=6000,
        minimum_per_stratum=600,
    )
    focused = [row for row in rows if row.strata & {"long", "double_letter"}]
    lexicon, _provenance = evaluator.load_dictionary_lexicon(args.dictionary_lexicon, args.dictionary_apk, layout)
    export, model_path, export_hash, numpy, versions, session = open_session(args.export_report)
    key_centers = numpy.asarray(layout["keyCenters"], dtype=numpy.float32).reshape(1, 64, 2)
    key_mask = numpy.asarray(layout["keyMask"], dtype=numpy.float32).reshape(1, 64)
    trie_cache: dict[tuple[str, int, int], evaluator.TrieNode] = {}
    decode_ms = {64: [], 256: []}
    present = {64: collections.Counter(), 256: collections.Counter()}
    counts = collections.Counter()
    for index, row in enumerate(focused, 1):
        output = session.run(["logits"], {
            "path_coordinates": numpy.asarray(row.path, dtype=numpy.float32).reshape(1, 64, 2),
            "key_centers": key_centers,
            "key_mask": key_mask,
        })[0]
        logits = output[0].tolist()
        greedy = evaluator.collapse_greedy(logits)
        approximate_length = len(greedy) or evaluator.path_length_estimate(row.path, layout) or 1
        for beam in (64, 256):
            cache_key = (row.language, approximate_length, beam)
            node = trie_cache.get(cache_key)
            if node is None:
                node = evaluator.build_trie(lexicon.get(row.language, ()), approximate_length)
                trie_cache[cache_key] = node
            started = time.perf_counter_ns()
            decoded = evaluator.prefix_beam_decode_scored(logits, node, beam_width=beam)
            decode_ms[beam].append((time.perf_counter_ns() - started) / 1_000_000)
            hit = row.target in {candidate.word for candidate in decoded}
            for name in ("overall", *sorted(row.strata & {"long", "double_letter"})):
                present[beam][f"{name}:rows"] += 1
                present[beam][f"{name}:hit"] += int(hit)
        counts["rows"] += 1
        if index % 100 == 0:
            print(f"beam256 {index}/{len(focused)}", file=sys.stderr, flush=True)
    after = hashlib.sha256(args.slates.read_bytes()).hexdigest()
    p95_256 = evaluator._percentiles(decode_ms[256])["p95"]
    adopted = p95_256 < 91.0
    def table(beam):
        result = {}
        for name in ("overall", "long", "double_letter"):
            rows_n = present[beam][f"{name}:rows"]
            hits = present[beam][f"{name}:hit"]
            result[name] = {
                "rows": rows_n,
                "targetPresent": hits,
                "recall": hits / rows_n if rows_n else 0.0,
            }
        return result
    report = {
        "schemaVersion": 1,
        "diagnosticOnly": True,
        "releaseEligible": False,
        "slatesSha256": FROZEN_SLATES_SHA256,
        "slatesSha256After": after,
        "modelSha256": export["model"]["sha256"],
        "exportReportSha256": export_hash,
        "toolSha256": sha256_file(pathlib.Path(__file__)),
        "evaluatorSha256": sha256_file(pathlib.Path(evaluator.__file__)),
        "rows": counts["rows"],
        "candidateRecall": {"beam64": table(64), "beam256": table(256)},
        "latencyMs": {
            "hostDiagnosticOnly": True,
            "prefixBeam64": evaluator._percentiles(decode_ms[64]),
            "prefixBeam256": evaluator._percentiles(decode_ms[256]),
        },
        "beam256Adopted": False,
        "beam256RejectedBecauseP95Unaffordable": p95_256 >= 91.0 or not adopted,
        "toolchain": versions,
        "limitations": [
            "Beam-256 recall is a search/capacity diagnostic on long and double_letter only.",
            "Host prefix-decoding p95 near 91 ms is unaffordable against the 200 ms swipe gate.",
            "Production beam width remains 64.",
        ],
    }
    write_json(args.output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit-calibration", "ranking-1000", "recall-6000", "beam256"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--export-report", type=pathlib.Path, default=DEFAULT_EXPORT)
    parser.add_argument("--dictionary-lexicon", type=pathlib.Path, default=DEFAULT_LEXICON)
    parser.add_argument("--dictionary-apk", type=pathlib.Path, default=DEFAULT_APK)
    parser.add_argument("--known-offensive-lexicon", type=pathlib.Path, default=DEFAULT_OFFENSIVE)
    parser.add_argument("--oov-calibration", type=pathlib.Path)
    parser.add_argument("--slates", type=pathlib.Path, default=DEFAULT_SLATES)
    parser.add_argument("--calibration-rows", type=int, default=512)
    parser.add_argument("--reserved-oov-nbest", type=int, default=1)
    parser.add_argument("--oov-beam-width", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--stratum-adaptive-merge", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        print(f"output already exists: {args.output}", file=sys.stderr)
        return 2
    try:
        if args.command == "fit-calibration":
            report = fit_calibration(args)
        elif args.command == "ranking-1000":
            if args.oov_calibration is None:
                raise evaluator.SwipeEvaluationError("ranking-1000 requires --oov-calibration")
            report = ranking_1000(args)
        elif args.command == "recall-6000":
            if args.oov_calibration is None:
                raise evaluator.SwipeEvaluationError("recall-6000 requires --oov-calibration")
            report = recall_6000(args)
        else:
            report = beam256_recall(args)
        print(json.dumps({
            "command": args.command,
            "output": str(args.output),
            "overall": report.get("ctcGeometricFusionMetrics", report.get("candidateRecall", report.get("counts"))),
            "decoder": report.get("decoder"),
            "latencyMs": report.get("latencyMs"),
        }, indent=2, sort_keys=True, default=str))
        return 0
    except (evaluator.SwipeEvaluationError, swipe_model_contract.SwipeModelContractError) as failure:
        print(f"candidate-recall diagnostic failed: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
