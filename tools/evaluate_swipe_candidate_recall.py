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


def _gap_stats(values: list[float]) -> dict[str, int | float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "positive": 0, "within0.10": 0, "within0.25": 0, "within0.50": 0}
    ordered = sorted(values)

    def at(percentile: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
        return ordered[index]

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": at(0.5),
        "positive": sum(1 for value in values if value >= 0.0),
        "within0.10": sum(1 for value in values if -0.10 <= value < 0.0),
        "within0.25": sum(1 for value in values if -0.25 <= value < 0.0),
        "within0.50": sum(1 for value in values if -0.50 <= value < 0.0),
    }


def _conversion_outside_report(outside: dict[str, Any]) -> dict[str, Any]:
    """Summarize published-31 reserved recoveries that sit outside top-3. Labels only."""
    ols = _gap_stats(outside["olsGaps"])
    return {
        "count": outside["count"],
        "frequencyFree": outside["frequencyFree"],
        "inLexicon": outside["inLexicon"],
        "bySource": dict(outside["bySource"]),
        "rank": dict(outside["rank"]),
        "blendedGap": _gap_stats(outside["blendedGaps"]),
        "olsGap": ols,
        "inLexiconGap": _gap_stats(outside["inLexiconGaps"]),
        "olsWouldEnterTop3": ols["positive"],
        "limitations": [
            "olsGap inverts oovMapBlend on frequency-free rows; unique-best rows already at OLS are overstated.",
            "Gaps are reserved fusion minus the 3rd lexicon fusion. Construction does not read targets.",
        ],
    }


def _empty_unpublished_competing() -> dict[str, Any]:
    sources = {name: 0 for name in evaluator.RESERVED_SOURCE_ORDER}
    sources["adaptive_merged"] = 0
    return {
        "count": 0,
        "frozenLost": 0,
        "newUnpublished": 0,
        "inReserved": 0,
        "inAdaptiveMergedOnly": 0,
        "frequencyFree": 0,
        "inLexicon": 0,
        "firstSource": sources,
        "fillRank": {"le11": 0, "gt11": 0, "none": 0},
        "oovCtcRank": {"le7": 0, "gt7": 0, "none": 0},
        "inPublishedBudget": 0,
        "olsWouldEnterTop3": 0,
        "convertingFillLoss": 0,
        "frozenLostMergedRankLe23": 0,
        "blendedGaps": [],
        "olsGaps": [],
        "inLexiconGaps": [],
        "rows": [],
    }


def _record_unpublished_competing(
    unpublished: dict[str, Any],
    *,
    match: evaluator.ScoredLexiconEntry | None,
    fill_rank: int | None,
    oov_ctc_rank: int | None,
    in_frozen: bool,
    in_merged: bool,
    reserved_budget: int,
    extra_oov: int,
    lexicon: list[evaluator.ScoredLexiconEntry],
    blend: float,
    merged_rank: int | None = None,
) -> None:
    """Label a competing-but-unpublished row. Construction never calls this."""
    unpublished["count"] += 1
    unpublished["frozenLost"] += int(in_frozen)
    unpublished["newUnpublished"] += int(not in_frozen)
    if in_frozen and merged_rank is not None and merged_rank <= 23:
        unpublished["frozenLostMergedRankLe23"] += 1
    if match is None:
        unpublished["inAdaptiveMergedOnly"] += 1
        unpublished["firstSource"]["adaptive_merged"] += 1
        unpublished["fillRank"]["none"] += 1
        unpublished["oovCtcRank"]["none"] += 1
        unpublished["rows"].append({
            "firstSource": "adaptive_merged",
            "fillRank": None,
            "oovCtcRank": None,
            "mergedRank": merged_rank,
            "inFrozen": in_frozen,
            "inAdaptiveMerged": in_merged,
            "frequencyFree": False,
            "olsGap": None,
            "blendedGap": None,
        })
        return
    unpublished["inReserved"] += 1
    source_name = match.source if match.source else "greedy"
    if source_name in unpublished["firstSource"]:
        unpublished["firstSource"][source_name] += 1
    if fill_rank is None:
        unpublished["fillRank"]["none"] += 1
    elif fill_rank <= reserved_budget:
        unpublished["fillRank"]["le11"] += 1
        unpublished["inPublishedBudget"] += 1
    else:
        unpublished["fillRank"]["gt11"] += 1
    if oov_ctc_rank is None:
        unpublished["oovCtcRank"]["none"] += 1
    elif oov_ctc_rank <= extra_oov:
        unpublished["oovCtcRank"]["le7"] += 1
    else:
        unpublished["oovCtcRank"]["gt7"] += 1
    dest = [item.spatial for item in lexicon]
    blended_gap = evaluator.reserved_gap_to_lexicon_top3(match, lexicon)
    ols_gap = None
    if match.frequency_free:
        unpublished["frequencyFree"] += 1
        unpublished["blendedGaps"].append(blended_gap)
        ols_spatial = evaluator.unblend_oov_spatial(match.spatial, dest, blend)
        ols_candidate = evaluator.ScoredLexiconEntry(
            match.entry, ols_spatial, frequency_free=True, source=match.source,
        )
        ols_gap = evaluator.reserved_gap_to_lexicon_top3(ols_candidate, lexicon)
        unpublished["olsGaps"].append(ols_gap)
        unpublished["olsWouldEnterTop3"] += int(ols_gap >= 0.0)
        lost_on_fill = (fill_rank is not None and fill_rank > reserved_budget) or (
            oov_ctc_rank is not None and oov_ctc_rank > extra_oov
        )
        if (
            not in_frozen
            and lost_on_fill
            and ols_gap >= 0.0
            and source_name in {"greedy", "greedy_alts"}
        ):
            unpublished["convertingFillLoss"] += 1
    else:
        unpublished["inLexicon"] += 1
        unpublished["inLexiconGaps"].append(blended_gap)
    unpublished["rows"].append({
        "firstSource": source_name,
        "fillRank": fill_rank,
        "oovCtcRank": oov_ctc_rank,
        "mergedRank": merged_rank,
        "inFrozen": in_frozen,
        "inAdaptiveMerged": in_merged,
        "frequencyFree": match.frequency_free,
        "olsGap": ols_gap,
        "blendedGap": blended_gap,
    })


def _unpublished_competing_report(unpublished: dict[str, Any]) -> dict[str, Any]:
    """Summarize competing-slate hits that missed ranked[:31]. Labels only."""
    ols = _gap_stats(unpublished["olsGaps"])
    return {
        "count": unpublished["count"],
        "frozenLost": unpublished["frozenLost"],
        "newUnpublished": unpublished["newUnpublished"],
        "inReserved": unpublished["inReserved"],
        "inAdaptiveMergedOnly": unpublished["inAdaptiveMergedOnly"],
        "frequencyFree": unpublished["frequencyFree"],
        "inLexicon": unpublished["inLexicon"],
        "firstSource": dict(unpublished["firstSource"]),
        "fillRank": dict(unpublished["fillRank"]),
        "oovCtcRank": dict(unpublished["oovCtcRank"]),
        "inPublishedBudget": unpublished["inPublishedBudget"],
        "olsWouldEnterTop3": unpublished["olsWouldEnterTop3"],
        "convertingFillLoss": unpublished["convertingFillLoss"],
        "frozenLostMergedRankLe23": unpublished["frozenLostMergedRankLe23"],
        "blendedGap": _gap_stats(unpublished["blendedGaps"]),
        "olsGap": ols,
        "inLexiconGap": _gap_stats(unpublished["inLexiconGaps"]),
        "rows": list(unpublished["rows"]),
        "limitations": [
            "Rows are competing-slate membership minus ranked[:31]. Construction does not read targets.",
            "convertingFillLoss is new unpublished frequency-free greedy/alts that lose the 11-slot or extra_oov=7 window and whose unblended OLS fusion would clear lexicon top-3.",
            "A non-zero convertingFillLoss is the only justification for a bounded extra_oov reorder.",
        ],
    }


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
        "--reserved-budget", str(args.reserved_budget),
        "--oov-beam-width", str(args.oov_beam_width),
        "--oov-map-blend", str(args.oov_map_blend),
        "--oov-lift-max-rank", str(args.oov_lift_max_rank),
        "--park-extra-reserved-min-rank", str(args.park_extra_reserved_min_rank),
    ]
    allowed_sources = parse_reserved_sources(args.reserved_sources)
    if args.stratum_adaptive_merge:
        argv.append("--stratum-adaptive-merge")
    if args.include_lexicon_neighbors or "neighbors" in allowed_sources:
        argv.append("--include-lexicon-neighbors")
    if args.include_truncated_leftovers or any(name.startswith("truncated_") for name in allowed_sources):
        argv.append("--include-truncated-leftovers")
    if args.oov_conservative_spatial:
        argv.append("--oov-conservative-spatial")
    if args.converting_best_occupant:
        argv.append("--converting-best-occupant")
    if args.prefer_converting_greedy_alts:
        argv.append("--prefer-converting-greedy-alts")
    if args.ablation_extra_fill:
        argv.append("--ablation-extra-fill")
    if args.leftover_inlex_fill:
        argv.append("--leftover-inlex-fill")
    if args.converting_leftover_extras:
        argv.append("--converting-leftover-extras")
    if args.converting_alt_expand:
        argv.append("--converting-alt-expand")
    if args.ablation_first_source_fill:
        argv.append("--ablation-first-source-fill")
    if args.converting_fill_loss_append:
        argv.append("--converting-fill-loss-append")
    if args.leftover_greedy_alts_append:
        argv.append("--leftover-greedy-alts-append")
    if args.protect_frozen_ranks:
        argv.append("--protect-frozen-ranks")
    if args.length_changing_extra_oov:
        argv.append("--length-changing-extra-oov")
    if args.leftover_converting_after_extra_oov:
        argv.append("--leftover-converting-after-extra-oov")
    parsed = evaluator.parse_args(argv)
    report = evaluator.evaluate(parsed)
    report["diagnosticOnly"] = True
    report["releaseEligible"] = False
    report["qualityClaim"] = False
    report["published31Membership"] = int(args.reserved_budget) > 0
    evaluator._write_report(args.output, report)
    return report


ABLATION_PREFIXES = (
    ("greedyOnly", ("greedy",), None),
    ("greedyAlts", ("greedy", "greedy_alts"), None),
    ("beamLe2", ("greedy", "greedy_alts", "nbest"), 2),
    ("beamLe4", ("greedy", "greedy_alts", "nbest"), 4),
    ("beamLe8", ("greedy", "greedy_alts", "nbest"), 8),
    ("beamLe16", ("greedy", "greedy_alts", "nbest"), 16),
    ("beamLe32", ("greedy", "greedy_alts", "nbest"), 32),
    ("plusNeighbors", ("greedy", "greedy_alts", "nbest", "neighbors"), 32),
    ("plusDropped", evaluator.RESERVED_SOURCE_ORDER, 32),
)
PUBLISHED_BUDGETS = (1, 2, 4)


def parse_reserved_sources(value: str | None) -> tuple[str, ...]:
    if not value:
        return evaluator.RESERVED_SOURCE_ORDER
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [name for name in names if name not in evaluator.RESERVED_SOURCE_ORDER]
    if unknown:
        raise evaluator.SwipeEvaluationError(f"unknown reserved sources: {unknown}")
    return names


def empty_stratum_counts() -> dict[str, dict[str, int]]:
    names = ("overall",) + evaluator.REQUIRED_STRATA
    return {name: {"rows": 0, "targetPresent": 0} for name in names}


def add_present(table: dict[str, dict[str, int]], strata: list[str], hit: bool) -> None:
    for name in ("overall", *strata):
        table[name]["rows"] += 1
        table[name]["targetPresent"] += bool(hit)


def decorate_recall(table: dict[str, dict[str, int]]) -> dict[str, dict[str, int | float]]:
    return {
        name: {
            **counts,
            "recall": counts["targetPresent"] / counts["rows"] if counts["rows"] else 0.0,
        }
        for name, counts in table.items()
    }


def ablate_184(args: argparse.Namespace) -> dict[str, Any]:
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
    first_source_counts = {name: 0 for name in evaluator.RESERVED_SOURCE_ORDER}
    overlap_counts = {name: 0 for name in evaluator.RESERVED_SOURCE_ORDER}
    cumulative_hits = {
        "greedyOnly": 0,
        "greedyAlts": 0,
        "beamLe": {str(width): 0 for width in range(2, 33)},
        "plusNeighbors": 0,
        "plusDropped": 0,
    }
    recovered_rows: list[dict[str, Any]] = []
    competing_present = 0
    frozen_present = 0
    rejected = 0
    inference_ms = []
    decode_ms = []
    grid = {
        f"{prefix_name}:b{budget}": {
            "membership": empty_stratum_counts(),
            "ranking": {name: {"rows": 0, "top1": 0, "top3": 0} for name in ("overall",) + evaluator.REQUIRED_STRATA},
        }
        for prefix_name, _sources, _limit in ABLATION_PREFIXES
        for budget in PUBLISHED_BUDGETS
    }
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
        ctc_n, geo_n = evaluator.decoder_slate_budgets(row.strata)
        merged = evaluator.merge_swipe_slates(scored(slate["ctc"][:ctc_n]), scored(slate["geometric"][:geo_n]))
        frozen_merged = scored(slate["merged"])
        sources, offensive, _spellings = evaluator.collect_reserved_sources(
            logits,
            layout,
            n_best=32,
            beam_width=32,
            existing=merged,
            known_offensive=known_offensive,
            lexicon_by_word=lexicon_by_language.get(row.language, {}),
            language=row.language,
            ctc=scored(slate["ctc"]),
            geometric=scored(slate["geometric"]),
            baseline_merged=frozen_merged,
            include_lexicon_neighbors=True,
            include_truncated=True,
        )
        finished = time.perf_counter_ns()
        rejected += offensive
        inference_ms.append((inferred - started) / 1_000_000)
        decode_ms.append((finished - inferred) / 1_000_000)
        skip_keys = {(evaluator._normalize(item.word), item.entry.language) for item in merged}
        reserved = evaluator.score_reserved_sources(
            sources,
            calibration=calibration,
            ctc_spatials=[item["spatial"] for item in slate["ctc"]],
            oov_conservative=False,
            lexicon_spatials=[item.spatial for item in merged],
            skip_keys=skip_keys,
        )
        competing = evaluator.competing_slate(merged, reserved)
        in_frozen = row.target in {item.word for item in frozen_merged}
        in_competing = row.target in {item.word for item in competing}
        frozen_present += in_frozen
        competing_present += in_competing
        if not in_frozen and in_competing:
            label = evaluator.first_recovered_source(
                row.target, [item.word for item in frozen_merged], sources,
            )
            stage = evaluator.cumulative_recovery_stage(row.target, sources)
            if label:
                first_source_counts[label] += 1
            for source in stage["sourcesPresent"]:
                overlap_counts[source] += 1
            if stage["greedyOnly"]:
                cumulative_hits["greedyOnly"] += 1
            if stage["greedyAlts"]:
                cumulative_hits["greedyAlts"] += 1
            for width, hit in stage["beamLe"].items():
                cumulative_hits["beamLe"][width] += bool(hit)
            if stage["plusNeighbors"]:
                cumulative_hits["plusNeighbors"] += 1
            if stage["plusDropped"]:
                cumulative_hits["plusDropped"] += 1
            recovered_rows.append({
                "id": row.identifier,
                "strata": sorted(row.strata),
                "firstSource": label,
                "sourcesPresent": stage["sourcesPresent"],
                "nbestRank": stage["nbestRank"],
            })
        ctc_spatials = [item["spatial"] for item in slate["ctc"]]
        for prefix_name, allowed, nbest_limit in ABLATION_PREFIXES:
            scored_reserved = evaluator.score_reserved_sources(
                sources,
                calibration=calibration,
                ctc_spatials=ctc_spatials,
                oov_conservative=False,
                lexicon_spatials=[item.spatial for item in merged],
                allowed=allowed,
                nbest_rank_limit=nbest_limit,
                skip_keys=skip_keys,
            )
            for budget in PUBLISHED_BUDGETS:
                published = evaluator.publish_reserved_slots(
                    merged, scored_reserved, reserved_budget=budget,
                )
                ranked = [
                    entry.word
                    for entry in evaluator.published_ranking(
                        published, scored_reserved, lexicon_reference=merged,
                    )
                ]
                key = f"{prefix_name}:b{budget}"
                present = row.target in ranked
                add_present(grid[key]["membership"], slate["strata"], present)
                for name in ("overall", *slate["strata"]):
                    ranking = grid[key]["ranking"][name]
                    ranking["rows"] += 1
                    ranking["top1"] += bool(ranked and ranked[0] == row.target)
                    ranking["top3"] += row.target in ranked[:3]
        if index % 250 == 0:
            print(f"ablate {index}/6000", file=sys.stderr, flush=True)
    after = hashlib.sha256(args.slates.read_bytes()).hexdigest()
    if after != FROZEN_SLATES_SHA256:
        raise evaluator.SwipeEvaluationError("frozen slates hash changed during the diagnostic")
    recovered = competing_present - frozen_present
    grid_report = {}
    for key, value in grid.items():
        membership = decorate_recall(value["membership"])
        ranking = {}
        for name, counts in value["ranking"].items():
            rows_n = counts["rows"]
            ranking[name] = {
                **counts,
                "top1Accuracy": counts["top1"] / rows_n if rows_n else 0.0,
                "top3Accuracy": counts["top3"] / rows_n if rows_n else 0.0,
            }
        grid_report[key] = {
            "published31TargetPresent": membership["overall"]["targetPresent"],
            "published31Recall": membership["overall"]["recall"],
            "returnTripTargetPresent": membership["return_trip"]["targetPresent"],
            "top1Accuracy": ranking["overall"]["top1Accuracy"],
            "top3Accuracy": ranking["overall"]["top3Accuracy"],
            "returnTripTop3": ranking["return_trip"]["top3"],
            "candidateRecall": membership,
            "ranking": ranking,
        }
    report = {
        "schemaVersion": 1,
        "diagnosticOnly": True,
        "releaseEligible": False,
        "qualityClaim": False,
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
        "stratumAdaptiveMerge": True,
        "counts": {
            "rows": 6000,
            "frozenMerged32TargetPresent": frozen_present,
            "competingTargetPresent": competing_present,
            "newTargetsRecovered": recovered,
            "knownOffensiveRejected": rejected,
            "firstSourceLabeled": len(recovered_rows),
            "constructionReadsTargets": False,
        },
        "firstSourceRecovered": first_source_counts,
        "sourceOverlap": overlap_counts,
        "cumulative": {
            "greedyOnly": cumulative_hits["greedyOnly"],
            "greedyAlts": cumulative_hits["greedyAlts"],
            "beamLe": cumulative_hits["beamLe"],
            "plusNeighbors": cumulative_hits["plusNeighbors"],
            "plusDropped": cumulative_hits["plusDropped"],
        },
        "recoveredRows": recovered_rows,
        "published31Grid": grid_report,
        "latencyMs": {
            "hostDiagnosticOnly": True,
            "inference": evaluator._percentiles(inference_ms),
            "reservedDecodeIncludingNBest32": evaluator._percentiles(decode_ms),
            "limitations": [
                "Ablation decode p95 includes n-best 32 and is not the published-31 path budget.",
            ],
        },
        "toolchain": versions,
        "limitations": [
            "First-source labels apply only to rows with target absent from the frozen merged 32 and present in the competing bag.",
            "Construction helpers never read evaluation targets; targets are labels only.",
            "published31Grid membership is ranked[:31] after unparked scoring; it is not a quality claim.",
            "Production vocabulary, scoring, beam, and safety policy were not changed.",
        ],
    }
    write_json(args.output, report)
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
    allowed_sources = parse_reserved_sources(args.reserved_sources)
    reserved_budget = int(args.reserved_budget)
    if reserved_budget <= 0:
        raise evaluator.SwipeEvaluationError("recall-6000 requires a positive published reserved budget")
    include_neighbors = "neighbors" in allowed_sources
    include_truncated = any(name.startswith("truncated_") for name in allowed_sources)
    nbest_limit = args.reserved_oov_nbest if "nbest" in allowed_sources else 1
    present = []
    ranked_words = []
    baseline_present = []
    baseline_ranked = []
    added = 0
    rejected = 0
    lost_at_publish = 0
    over_bound = 0
    inference_ms = []
    decode_ms = []
    new_hits = {
        "frequencyFree": 0,
        "inLexicon": 0,
        "top3FrequencyFree": 0,
        "top3InLexicon": 0,
        "bySource": {
            name: {"present": 0, "top3": 0} for name in evaluator.RESERVED_SOURCE_ORDER
        },
    }
    outside_top3 = {
        "count": 0,
        "frequencyFree": 0,
        "inLexicon": 0,
        "bySource": {name: 0 for name in evaluator.RESERVED_SOURCE_ORDER},
        "rank": {"4-6": 0, "7-10": 0, "11-20": 0, "21-31": 0},
        "blendedGaps": [],
        "olsGaps": [],
        "inLexiconGaps": [],
    }
    unpublished = _empty_unpublished_competing()
    competing_present = 0
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
            n_best=nbest_limit,
            beam_width=min(args.oov_beam_width, max(nbest_limit, 4)),
            existing=merged,
            known_offensive=known_offensive,
            lexicon_by_word=lexicon_by_language.get(row.language, {}),
            language=row.language,
            ctc_spatials=[item["spatial"] for item in slate["ctc"]],
            include_lexicon_neighbors=include_neighbors,
            ctc=scored(slate["ctc"]),
            geometric=scored(slate["geometric"]),
            baseline_merged=scored(slate["merged"]),
            include_truncated=include_truncated,
            oov_conservative=bool(args.oov_conservative_spatial),
            oov_map_blend=float(args.oov_map_blend),
            allowed_sources=allowed_sources,
            nbest_rank_limit=nbest_limit if "nbest" in allowed_sources else None,
        )
        rejected += offensive
        published = evaluator.publish_reserved_slots(
            merged,
            reserved,
            reserved_budget=reserved_budget,
            converting_best=bool(args.converting_best_occupant),
            prefer_converting_greedy_alts=bool(args.prefer_converting_greedy_alts),
            ablation_extra_fill_sources=bool(args.ablation_extra_fill),
            leftover_inlex_fill=bool(args.leftover_inlex_fill),
            converting_leftover_extras_fill=bool(args.converting_leftover_extras),
            converting_alt_expand=bool(args.converting_alt_expand),
            ablation_first_source_fill_sources=bool(args.ablation_first_source_fill),
            converting_fill_loss_append=bool(args.converting_fill_loss_append),
            leftover_greedy_alts_append=bool(args.leftover_greedy_alts_append),
            protect_frozen_ranks=bool(args.protect_frozen_ranks),
            length_changing_extra_oov=bool(args.length_changing_extra_oov),
            leftover_converting_after_extra_oov_fill=bool(args.leftover_converting_after_extra_oov),
        )
        skip_keys = {
            (evaluator._normalize(item.word), item.entry.language) for item in merged
        }
        extra_park_keys = evaluator.extra_reserved_occupant_keys(
            reserved,
            merged,
            reserved_budget=reserved_budget,
            skip_keys=skip_keys,
            converting_best=bool(args.converting_best_occupant),
            prefer_converting_greedy_alts=bool(args.prefer_converting_greedy_alts),
            ablation_extra_fill_sources=bool(args.ablation_extra_fill),
            leftover_inlex_fill=bool(args.leftover_inlex_fill),
            converting_leftover_extras_fill=bool(args.converting_leftover_extras),
            converting_alt_expand=bool(args.converting_alt_expand),
            ablation_first_source_fill_sources=bool(args.ablation_first_source_fill),
            converting_fill_loss_append=bool(args.converting_fill_loss_append),
            leftover_greedy_alts_append=bool(args.leftover_greedy_alts_append),
            protect_frozen_ranks=bool(args.protect_frozen_ranks),
            length_changing_extra_oov=bool(args.length_changing_extra_oov),
            leftover_converting_after_extra_oov_fill=bool(args.leftover_converting_after_extra_oov),
        )
        park_min_rank = int(args.park_extra_reserved_min_rank)
        published = evaluator.lift_near_top3_frequency_free(
            published,
            reserved,
            merged,
            blend=float(args.oov_map_blend),
            optimism_offset=calibration.optimism_offset,
            max_rank=int(args.oov_lift_max_rank),
            extra_park_keys=extra_park_keys,
            park_min_rank=park_min_rank,
        )
        published = evaluator.unblend_greedy_alts_when_greedy_misses_top3(
            published, reserved, merged,
        )
        if len(published) > evaluator.PUBLISHED_SLATE_BOUND:
            over_bound += 1
        added += min(reserved_budget, len(published))
        ranked = evaluator.published_ranking(
            published,
            reserved,
            lexicon_reference=merged,
            extra_park_keys=extra_park_keys,
            park_min_rank=park_min_rank,
            protect_frozen_ranks=bool(args.protect_frozen_ranks),
        )
        ranked_list = [entry.word for entry in ranked]
        if row.target in {item.word for item in merged} and row.target not in ranked_list:
            lost_at_publish += 1
        finished = time.perf_counter_ns()
        inference_ms.append((inferred - started) / 1_000_000)
        decode_ms.append((finished - inferred) / 1_000_000)
        present.append(row.target in ranked_list)
        ranked_words.append(ranked_list)
        frozen_words = {item["word"] for item in slate["merged"]}
        baseline_present.append(row.target in frozen_words)
        competing = evaluator.competing_slate(merged, reserved)
        competing_words = {item.word for item in competing}
        merged_words = {item.word for item in merged}
        in_competing = row.target in competing_words
        competing_present += int(in_competing)
        if in_competing and row.target not in ranked_list:
            match = next((item for item in reserved if item.word == row.target), None)
            key = (evaluator._normalize(row.target), row.language)
            fill_ranks = evaluator.reserved_fill_ranks(reserved)
            oov_ranks = evaluator.reserved_oov_ctc_ranks(reserved)
            merged_rank = next(
                (index for index, item in enumerate(merged, 1) if item.word == row.target),
                None,
            )
            _record_unpublished_competing(
                unpublished,
                match=match,
                fill_rank=fill_ranks.get(key),
                oov_ctc_rank=oov_ranks.get(key),
                in_frozen=row.target in frozen_words,
                in_merged=row.target in merged_words,
                reserved_budget=reserved_budget,
                extra_oov=evaluator.EXTRA_OOV_FILL,
                lexicon=merged,
                blend=float(args.oov_map_blend),
                merged_rank=merged_rank,
            )
        if row.target in ranked_list and row.target not in frozen_words:
            match = next((item for item in published if item.word == row.target), None)
            in_top3 = row.target in ranked_list[:3]
            if match is not None and match.frequency_free:
                new_hits["frequencyFree"] += 1
                new_hits["top3FrequencyFree"] += int(in_top3)
            else:
                new_hits["inLexicon"] += 1
                new_hits["top3InLexicon"] += int(in_top3)
            source_name = match.source if match is not None and match.source else "greedy"
            if source_name in new_hits["bySource"]:
                new_hits["bySource"][source_name]["present"] += 1
                new_hits["bySource"][source_name]["top3"] += int(in_top3)
            if match is not None and not in_top3:
                rank = ranked_list.index(row.target) + 1
                gap = evaluator.reserved_gap_to_lexicon_top3(match, merged)
                dest = [item.spatial for item in merged]
                outside_top3["count"] += 1
                if match.frequency_free:
                    outside_top3["frequencyFree"] += 1
                    outside_top3["blendedGaps"].append(gap)
                    ols_spatial = evaluator.unblend_oov_spatial(
                        match.spatial, dest, float(args.oov_map_blend),
                    )
                    ols_candidate = evaluator.ScoredLexiconEntry(
                        match.entry, ols_spatial, frequency_free=True, source=match.source,
                    )
                    outside_top3["olsGaps"].append(
                        evaluator.reserved_gap_to_lexicon_top3(ols_candidate, merged)
                    )
                else:
                    outside_top3["inLexicon"] += 1
                    outside_top3["inLexiconGaps"].append(gap)
                if source_name in outside_top3["bySource"]:
                    outside_top3["bySource"][source_name] += 1
                if rank <= 6:
                    outside_top3["rank"]["4-6"] += 1
                elif rank <= 10:
                    outside_top3["rank"]["7-10"] += 1
                elif rank <= 20:
                    outside_top3["rank"]["11-20"] += 1
                else:
                    outside_top3["rank"]["21-31"] += 1
        baseline_ranked.append(
            [entry.word for entry in evaluator.rank_static_fusion(scored(slate["merged"]))]
        )
        if index % 250 == 0:
            print(f"recall {index}/6000", file=sys.stderr, flush=True)
    after = hashlib.sha256(args.slates.read_bytes()).hexdigest()
    if after != FROZEN_SLATES_SHA256:
        raise evaluator.SwipeEvaluationError("frozen slates hash changed during the diagnostic")
    if over_bound:
        raise evaluator.SwipeEvaluationError("published slate grew past the 32-slot bound")
    membership, ranking = metric_tables(slates, present, ranked_words)
    baseline_membership, baseline_ranking = metric_tables(slates, baseline_present, baseline_ranked)
    p95 = evaluator._percentiles(decode_ms)["p95"]
    report = {
        "schemaVersion": 1,
        "diagnosticOnly": True,
        "releaseEligible": False,
        "qualityClaim": False,
        "slatesSha256": FROZEN_SLATES_SHA256,
        "slatesSha256After": after,
        "modelSha256": export["model"]["sha256"],
        "exportReportSha256": export_hash,
        "toolSha256": sha256_file(pathlib.Path(__file__)),
        "evaluatorSha256": sha256_file(pathlib.Path(evaluator.__file__)),
        "knownOffensiveSourceSha256": sha256_file(args.known_offensive_lexicon),
        "oovCalibrationSha256": sha256_file(args.oov_calibration),
        "reservedOovNBest": nbest_limit,
        "oovBeamWidth": min(args.oov_beam_width, max(nbest_limit, 4)),
        "reservedBudget": reserved_budget,
        "reservedSources": list(allowed_sources),
        "stratumAdaptiveMerge": bool(args.stratum_adaptive_merge),
        "oovConservativeSpatial": bool(args.oov_conservative_spatial),
        "oovMapBlend": float(args.oov_map_blend),
        "parkExtraReservedMinRank": int(args.park_extra_reserved_min_rank),
        "convertingBestOccupant": bool(args.converting_best_occupant),
        "preferConvertingGreedyAlts": bool(args.prefer_converting_greedy_alts),
        "ablationExtraFill": bool(args.ablation_extra_fill),
        "leftoverInlexFill": bool(args.leftover_inlex_fill),
        "convertingLeftoverExtras": bool(args.converting_leftover_extras),
        "convertingAltExpand": bool(args.converting_alt_expand),
        "ablationFirstSourceFill": bool(args.ablation_first_source_fill),
        "convertingFillLossAppend": bool(args.converting_fill_loss_append),
        "leftoverGreedyAltsAppend": bool(args.leftover_greedy_alts_append),
        "protectFrozenRanks": bool(args.protect_frozen_ranks),
        "lengthChangingExtraOov": bool(args.length_changing_extra_oov),
        "leftoverConvertingAfterExtraOov": bool(args.leftover_converting_after_extra_oov),
        "published31Membership": True,
        "newHits": new_hits,
        "conversionOutsideTop3": _conversion_outside_report(outside_top3),
        "unpublishedCompeting": _unpublished_competing_report(unpublished),
        "counts": {
            "rows": 6000,
            "baselineTargetPresent": baseline_membership["overall"]["targetPresent"],
            "competingTargetPresent": competing_present,
            "published31TargetPresent": membership["overall"]["targetPresent"],
            "newTargetsRecovered": membership["overall"]["targetPresent"] - baseline_membership["overall"]["targetPresent"],
            "targetsLostAtPublish": lost_at_publish,
            "reservedBudget": reserved_budget,
            "knownOffensiveRejected": rejected,
            "recallCountsPublished31Only": True,
            "publishedSlateBound": evaluator.PUBLISHED_SLATE_BOUND,
            "publishedRankingBound": evaluator.PUBLISHED_RANKING_BOUND,
        },
        "candidateRecall": membership,
        "baselineCandidateRecall": baseline_membership,
        "ranking": ranking,
        "baselineRanking": baseline_ranking,
        "conversionMath": {
            "phase0Top3Gate": 5700,
            "historicalLexiconConversion": 0.979,
            "published31NeededAtHistoricalConversion": 5822,
            "measuredPublished31TargetPresent": membership["overall"]["targetPresent"],
            "measuredTop3": ranking["overall"]["top3"],
            "measuredConversionOfPublished31": (
                ranking["overall"]["top3"] / membership["overall"]["targetPresent"]
                if membership["overall"]["targetPresent"] else 0.0
            ),
            "newMembershipHits": membership["overall"]["targetPresent"] - baseline_membership["overall"]["targetPresent"],
            "newTop3Hits": ranking["overall"]["top3"] - baseline_ranking["overall"]["top3"],
            "returnTripPublished31NeededAtHistoricalConversion": 1572,
            "returnTripPublished31TargetPresent": membership["return_trip"]["targetPresent"],
            "returnTripTop3Gate": 1539,
            "returnTripMeasuredTop3": ranking["return_trip"]["top3"],
            "note": "Historical 97.9% conversion applies to frozen lexicon membership, not new reserved recoveries.",
        },
        "latencyMs": {
            "hostDiagnosticOnly": True,
            "inference": evaluator._percentiles(inference_ms),
            "reservedOov": evaluator._percentiles(decode_ms),
            "reservedOovP95BudgetMs": 91.0,
            "reservedOovP95WithinBudget": p95 < 91.0,
        },
        "toolchain": versions,
        "limitations": [
            "Candidate recall is membership in rank_static_fusion_with_reserved(...)[:31], the published list.",
            "Reserved merge displaces into the 32-slot bound and does not grow a bag ranking then truncates.",
            "True OOV stay frequency-free and use the train-fit spatial map; in-lexicon reserved use real frequency.",
            "This is published-31 membership, not a Phase 0 quality or top-3 win.",
            "Production vocabulary, scoring, beam, and safety policy were not changed.",
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
    parser.add_argument(
        "command",
        choices=("fit-calibration", "ranking-1000", "recall-6000", "beam256", "ablate-184"),
    )
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
    parser.add_argument("--oov-beam-width", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--stratum-adaptive-merge", action="store_true")
    parser.add_argument("--reserved-budget", type=int, default=1)
    parser.add_argument(
        "--reserved-sources",
        type=str,
        default="greedy,greedy_alts",
    )
    parser.add_argument("--include-lexicon-neighbors", action="store_true")
    parser.add_argument("--include-truncated-leftovers", action="store_true")
    parser.add_argument("--oov-conservative-spatial", action="store_true")
    parser.add_argument("--oov-map-blend", type=float, default=0.0)
    parser.add_argument("--oov-lift-max-rank", type=int, default=6)
    parser.add_argument(
        "--park-extra-reserved-min-rank",
        type=int,
        default=0,
        help="Demote extra reserved occupants after greedy-first to this rank or worse (0 = off)",
    )
    parser.add_argument(
        "--converting-best-occupant",
        action="store_true",
        help="First reserved seat is the highest-OLS converting OOV, not greedy-first",
    )
    parser.add_argument(
        "--prefer-converting-greedy-alts",
        action="store_true",
        help="Extra reserved seats prefer OLS-converting greedy_alts over non-converting n-best/alts",
    )
    parser.add_argument(
        "--ablation-extra-fill",
        action="store_true",
        help="Extra reserved seats round-robin greedy_alts, neighbors, n-best, truncated",
    )
    parser.add_argument(
        "--leftover-inlex-fill",
        action="store_true",
        help="Leftover extra_oov seats after converting OOV go to neighbors/truncated",
    )
    parser.add_argument(
        "--converting-leftover-extras",
        action="store_true",
        help="Leftover extra seats only OLS-converting greedy_alts/nbest",
    )
    parser.add_argument(
        "--converting-alt-expand",
        action="store_true",
        help="Expand extra converting greedy_alt seats when they overflow extra_oov",
    )
    parser.add_argument(
        "--ablation-first-source-fill",
        action="store_true",
        help="Extra seats follow ablation first-source (length-changing greedy_alts, neighbors, n-best last)",
    )
    parser.add_argument(
        "--converting-fill-loss-append",
        action="store_true",
        help="Append leftover converting greedy_alts after spatial occupants; keep unique n-best seats",
    )
    parser.add_argument(
        "--leftover-greedy-alts-append",
        action="store_true",
        help="Append leftover greedy_alts after spatial occupants; do not park extras",
    )
    parser.add_argument(
        "--protect-frozen-ranks",
        action="store_true",
        help="32-slot fill keeping lexicon fusion ranks 1-23; leftover converting alts occupy 24-32; count ranked[:31]",
    )
    parser.add_argument(
        "--length-changing-extra-oov",
        action="store_true",
        help="extra_oov leftover seats prefer length-changing converting greedy_alts; hold spatial n-best",
    )
    parser.add_argument(
        "--leftover-converting-after-extra-oov",
        action="store_true",
        help="leftover seats after spatial extra_oov prefer leftover converting greedy_alts; hold extra_oov/n-best",
    )
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
        elif args.command == "ablate-184":
            if args.oov_calibration is None:
                raise evaluator.SwipeEvaluationError("ablate-184 requires --oov-calibration")
            report = ablate_184(args)
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
