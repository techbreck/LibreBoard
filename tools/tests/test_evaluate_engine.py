# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
import zipfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluate_engine import (  # noqa: E402
    EvaluationError,
    SwipeVocabulary,
    accuracy,
    auto_correction_rate,
    evaluate,
    false_correction_rate,
    load_swipe_vocabulary,
    parse_example,
    read_jsonl,
    validate_swipe_strata,
)


def example(
    number: int,
    category: str,
    target: str,
    raw: str,
    predictions: dict[str, list[str]],
    *,
    split: str = "test",
    session: str | None = None,
    strata: list[str] | None = None,
    should_correct: bool | None = None,
    environment_kind: str = "stock_android_hardware",
    test_run_id: str = "stock-run",
    latency_overrides: dict[str, float] | None = None,
    commits: dict[str, dict[str, object]] | None = None,
):
    latency = {system: 20.0 for system in predictions}
    latency.update(latency_overrides or {})
    value = {
        "schemaVersion": 4,
        "id": f"example-{number}",
        "sessionId": session or f"session-{number}",
        "split": split,
        "category": category,
        "target": target,
        "raw": raw,
        "predictions": predictions,
        "latencyMs": latency,
        "strata": strata or [],
    }
    if split == "test":
        value["environmentKind"] = environment_kind
        value["testRunId"] = test_run_id
    if category == "lexical":
        value["lexicalKind"] = "contraction"
    if should_correct is not None:
        value["shouldCorrect"] = should_correct
    if split == "test" and category != "swipe":
        # Default to committing each system's own head candidate, which is what a decision-free
        # fixture means; tests that care about keep-versus-correct pass commits explicitly.
        value["commits"] = commits or {
            system: {"committed": slate[0], "willAutoCorrect": slate[0] != raw}
            for system, slate in predictions.items()
        }
    return parse_example(value, number)


def vocabulary(*words: str, apk: str = "b" * 64) -> SwipeVocabulary:
    """A production vocabulary bound to the metadata() APK unless told otherwise."""
    return SwipeVocabulary(frozenset(words), {"measuredApkSha256": apk, "decodableWords": len(words)})


def metadata():
    return {
        "schemaVersion": 4,
        "appCommit": "a" * 40,
        "coreApkSha256": "b" * 64,
        "swipeModelSha256": "c" * 64,
        "contextModelSha256": "d" * 64,
        "artifactPin": {
            "candidateId": "context-shared-session-v2",
            "contextModelSha256": "d" * 64,
            "swipeModelSha256": "c" * 64,
        },
        "environments": [
            {
                "kind": "stock_android_hardware",
                "deviceModel": "Pixel 8a",
                "buildFingerprint": "stock/fingerprint",
                "testRunId": "stock-run",
                "apiLevel": 36,
                "physicalDevice": True,
                "peakAddedNeuralMemoryMiB": 40,
            },
            {
                "kind": "grapheneos_hardware",
                "deviceModel": "Pixel 8a",
                "buildFingerprint": "graphene/fingerprint",
                "testRunId": "graphene-run",
                "apiLevel": 36,
                "physicalDevice": True,
                "grapheneOsBuildNumber": "2026090100",
                "sandboxedGooglePlayInstalled": False,
                "peakAddedNeuralMemoryMiB": 42,
            },
            {
                "kind": "low_ram_emulator",
                "deviceModel": "AOSP low RAM",
                "buildFingerprint": "aosp/fingerprint",
                "testRunId": "low-ram-run",
                "apiLevel": 36,
                "physicalDevice": False,
                "isLowRamDevice": True,
                "memoryMiB": 1_024,
                "peakAddedNeuralMemoryMiB": 18,
            },
        ],
    }


class EvaluateEngineTest(unittest.TestCase):
    def test_lexical_kind_is_required_and_category_scoped(self):
        row = {"schemaVersion": 4, "id": "lexical", "sessionId": "session", "split": "train",
               "category": "lexical", "raw": "dont", "target": "don't"}
        for invalid in (None, "unknown", True):
            with self.subTest(kind=invalid), self.assertRaisesRegex(EvaluationError, "valid lexicalKind"):
                parse_example({**row, "lexicalKind": invalid}, 1)
        with self.assertRaisesRegex(EvaluationError, "valid lexicalKind"):
            parse_example(row, 1)
        self.assertEqual("contraction", parse_example({**row, "lexicalKind": "contraction"}, 1).lexical_kind)
        with self.assertRaisesRegex(EvaluationError, "only for lexical"):
            parse_example({**row, "category": "tap_error", "lexicalKind": "contraction"}, 1)
        with self.assertRaisesRegex(EvaluationError, "unsupported schemaVersion"):
            parse_example({**row, "schemaVersion": 2, "lexicalKind": "contraction"}, 1)

    def test_valid_word_labels_cannot_reclassify_keeps_as_corrections(self):
        row = {"schemaVersion": 4, "id": "valid", "sessionId": "session", "split": "train",
               "category": "valid_word", "raw": "their", "target": "there"}
        with self.assertRaisesRegex(EvaluationError, "contradicts"):
            parse_example({**row, "shouldCorrect": False}, 1)
        with self.assertRaisesRegex(EvaluationError, "contradicts"):
            parse_example({**row, "target": "THEIR", "shouldCorrect": True}, 1)
        self.assertFalse(parse_example({**row, "target": "THEIR", "shouldCorrect": False}, 1).should_correct)

    def test_typed_word_first_slate_is_scored_by_the_commit_not_the_strip_head(self):
        """The schema-3 harness defect: production seats the typed word ahead of its correction.

        Scoring rank one of that slate reported the typed text back to itself, so every system
        showed 0% top-1 on tap errors regardless of whether it actually corrected anything.
        """
        slate = ["targte", "target"]
        corrected = example(
            1, "tap_error", "target", "targte",
            {system: list(slate) for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            commits={system: {"committed": "target", "willAutoCorrect": True} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        kept = example(
            2, "tap_error", "target", "targte",
            {system: list(slate) for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            commits={system: {"committed": "targte", "willAutoCorrect": False} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        self.assertEqual(1.0, accuracy([corrected], "fused_neural"))
        self.assertEqual(0.0, accuracy([kept], "fused_neural"))
        # Both rows expose the correction one tap away, so top-3 cannot distinguish them.
        self.assertEqual(1.0, accuracy([corrected], "fused_neural", 3))
        self.assertEqual(1.0, accuracy([kept], "fused_neural", 3))
        self.assertEqual(1.0, auto_correction_rate([corrected], "fused_neural"))
        self.assertEqual(0.0, auto_correction_rate([kept], "fused_neural"))

    def test_top_three_keeps_the_committed_word_reachable(self):
        row = example(
            1, "tap_error", "target", "targte",
            {system: ["targte", "other", "another", "target"] for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            commits={system: {"committed": "target", "willAutoCorrect": True} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        # "target" sits at rank four on the strip, but it is what the editor receives.
        self.assertEqual(1.0, accuracy([row], "fused", 3))
        self.assertEqual(1.0, accuracy([row], "fused"))

    def test_false_corrections_count_commits_rather_than_strip_order(self):
        ranked_but_kept = example(
            1, "valid_word", "their", "their",
            {system: ["their", "there"] for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            should_correct=False,
            commits={system: {"committed": "their", "willAutoCorrect": False} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        replaced = example(
            2, "valid_word", "their", "their",
            {system: ["their", "there"] for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            should_correct=False,
            commits={system: {"committed": "there", "willAutoCorrect": True} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        self.assertEqual(0.0, false_correction_rate([ranked_but_kept], "fused_neural"))
        self.assertEqual(1.0, false_correction_rate([replaced], "fused_neural"))

    def test_auto_capitalization_is_not_a_false_correction(self):
        # Committing "Thursday" for a typed "thursday" is the same word, and accuracy scores it as
        # a hit because accuracy is normalized. The false-correction metric must agree, or the
        # keyboard is penalized for behaviour the accuracy gate rewards.
        capitalized = example(
            1, "valid_word", "thursday", "thursday",
            {system: ["thursday"] for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            should_correct=False,
            commits={system: {"committed": "Thursday", "willAutoCorrect": True} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        self.assertEqual(0.0, false_correction_rate([capitalized], "fused_neural"))
        self.assertEqual(1.0, accuracy([capitalized], "fused_neural"))
        # Replacing it with a genuinely different word still counts.
        replaced = example(
            2, "valid_word", "thursday", "thursday",
            {system: ["thursday", "Tuesday"] for system in
             ("heliboard", "fused", "fused_personal", "fused_neural")},
            should_correct=False,
            commits={system: {"committed": "Tuesday", "willAutoCorrect": True} for system in
                     ("heliboard", "fused", "fused_personal", "fused_neural")},
        )
        self.assertEqual(1.0, false_correction_rate([replaced], "fused_neural"))

    def test_commit_records_must_be_present_and_internally_consistent(self):
        def row(**overrides):
            value = {
                "schemaVersion": 4, "id": "tap", "sessionId": "session", "split": "test",
                "environmentKind": "stock_android_hardware", "testRunId": "stock-run",
                "category": "tap_error", "target": "target", "raw": "targte",
                "predictions": {system: ["targte", "target"] for system in
                                ("heliboard", "fused", "fused_personal", "fused_neural")},
                "latencyMs": {system: 20.0 for system in
                              ("heliboard", "fused", "fused_personal", "fused_neural")},
                "commits": {system: {"committed": "target", "willAutoCorrect": True} for system in
                            ("heliboard", "fused", "fused_personal", "fused_neural")},
            }
            value.update(overrides)
            return value

        self.assertEqual(("target", True), parse_example(row(), 1).commits["fused"])

        missing = row()
        del missing["commits"]
        with self.assertRaisesRegex(EvaluationError, "require commits"):
            parse_example(missing, 1)

        partial = row()
        partial["commits"] = {"fused": {"committed": "target", "willAutoCorrect": True}}
        with self.assertRaisesRegex(EvaluationError, "require commits"):
            parse_example(partial, 1)

        keep_mismatch = row()
        keep_mismatch["commits"]["fused"] = {"committed": "target", "willAutoCorrect": False}
        with self.assertRaisesRegex(EvaluationError, "keeps the typed word"):
            parse_example(keep_mismatch, 1)

        correction_mismatch = row()
        correction_mismatch["commits"]["fused"] = {"committed": "targte", "willAutoCorrect": True}
        with self.assertRaisesRegex(EvaluationError, "equals the raw surface"):
            parse_example(correction_mismatch, 1)

        off_slate = row()
        off_slate["commits"]["fused"] = {"committed": "elsewhere", "willAutoCorrect": True}
        with self.assertRaisesRegex(EvaluationError, "absent from its own slate"):
            parse_example(off_slate, 1)

        malformed = row()
        malformed["commits"]["fused"] = {"committed": "target"}
        with self.assertRaisesRegex(EvaluationError, "invalid commit record"):
            parse_example(malformed, 1)

        non_boolean = row()
        non_boolean["commits"]["fused"] = {"committed": "target", "willAutoCorrect": "yes"}
        with self.assertRaisesRegex(EvaluationError, "willAutoCorrect must be a boolean"):
            parse_example(non_boolean, 1)

    def test_commits_belong_only_to_measured_tap_rows(self):
        swipe = {
            "schemaVersion": 4, "id": "swipe", "sessionId": "session", "split": "test",
            "environmentKind": "stock_android_hardware", "testRunId": "stock-run",
            "category": "swipe", "target": "target", "raw": "", "strata": ["short"],
            "predictions": {system: ["target"] for system in ("geometric", "ctc", "fused_swipe")},
            "latencyMs": {system: 20.0 for system in ("geometric", "ctc", "fused_swipe")},
            "commits": {"fused_swipe": {"committed": "target", "willAutoCorrect": True}},
        }
        with self.assertRaisesRegex(EvaluationError, "only to measured tap rows"):
            parse_example(swipe, 1)

        training = {
            "schemaVersion": 4, "id": "tap", "sessionId": "session", "split": "train",
            "category": "tap_error", "target": "target", "raw": "targte",
            "commits": {system: {"committed": "target", "willAutoCorrect": True} for system in
                        ("heliboard", "fused", "fused_personal", "fused_neural")},
        }
        with self.assertRaisesRegex(EvaluationError, "only to measured tap rows"):
            parse_example(training, 1)

    def test_metadata_must_name_the_candidate_it_qualifies(self):
        rows = [example(1, "tap_error", "target", "targte", {
            system: ["targte", "target"] for system in
            ("heliboard", "fused", "fused_personal", "fused_neural")})]

        def run(pin):
            data = metadata()
            if pin is None:
                del data["artifactPin"]
            else:
                data["artifactPin"] = pin
            return evaluate(rows, data, measurement_sha256="e" * 64, enforce_minimum_counts=False)

        with self.assertRaisesRegex(EvaluationError, "requires an artifactPin"):
            run(None)
        with self.assertRaisesRegex(EvaluationError, "requires a candidateId"):
            run({"candidateId": "", "contextModelSha256": "d" * 64, "swipeModelSha256": "c" * 64})
        # The superseded-model failure this exists to catch: the pin names one candidate while the
        # measurement was taken against another.
        with self.assertRaisesRegex(EvaluationError, "does not qualify the pinned candidate"):
            run({"candidateId": "context-shared-session-v2",
                 "contextModelSha256": "f" * 64, "swipeModelSha256": "c" * 64})
        with self.assertRaisesRegex(EvaluationError, "requires an artifactPin"):
            run({"candidateId": "context-shared-session-v2", "contextModelSha256": "d" * 64})

    def test_passing_measurements_satisfy_every_gate(self):
        tap_systems = {
            "heliboard": ["wrong"],
            "fused": ["target", "targte", "inthe", "dont"],
            "fused_personal": ["target", "targte", "inthe", "dont"],
            "fused_neural": ["target", "targte", "inthe", "dont"],
        }
        rows = [example(1, "tap_error", "target", "targte", tap_systems)]
        rows += [example(2, "spacing", "in the", "inthe", tap_systems)]
        rows += [example(3, "lexical", "don't", "dont", tap_systems)]
        rows += [example(4, "valid_word", "there", "their", {
            "heliboard": ["their"], "fused": ["their"], "fused_personal": ["their"],
            "fused_neural": ["there", "their"],
        }, should_correct=True)]
        rows += [example(5, "valid_word", "their", "their", {
            "heliboard": ["their"], "fused": ["their"], "fused_personal": ["their"],
            "fused_neural": ["their"],
        }, should_correct=False)]
        swipe_systems = {
            "geometric": ["wrong"], "ctc": ["target"], "fused_swipe": ["target"],
        }
        rows += [example(6, "swipe", "target", "", swipe_systems, strata=["short"])]
        rows += [example(7, "swipe", "target", "", swipe_systems, strata=["return_trip"])]
        rows += [example(8, "tap_error", "target", "targte", tap_systems,
                         environment_kind="grapheneos_hardware", test_run_id="graphene-run")]
        rows += [example(9, "swipe", "target", "", swipe_systems, strata=["medium"],
                         environment_kind="grapheneos_hardware", test_run_id="graphene-run")]
        rows += [example(10, "tap_error", "target", "targte", tap_systems,
                         environment_kind="low_ram_emulator", test_run_id="low-ram-run")]
        rows += [example(11, "swipe", "target", "", swipe_systems, strata=["long"],
                         environment_kind="low_ram_emulator", test_run_id="low-ram-run")]

        result = evaluate(
            rows,
            metadata(),
            measurement_sha256="e" * 64,
            enforce_minimum_counts=False,
            swipe_vocabulary=vocabulary("target"),
        )
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual({}, result["notApplicable"])
        self.assertEqual("b" * 64, result["evidence"]["coreApkSha256"])
        self.assertEqual("context-shared-session-v2", result["evidence"]["artifactPin"]["candidateId"])
        self.assertEqual(1, result["swipeStrataCounts"]["short"])
        self.assertEqual(1.0, result["gates"]["neuralContextRelativeErrorReduction"])
        self.assertEqual({"correct": 1, "keep": 1}, result["validWordCounts"])

    def test_swipe_gates_score_in_lexicon_targets_and_fail_closed_without_a_vocabulary(self):
        rows = self._minimum_rows()
        miss = {system: ["venue"] for system in ("geometric", "ctc", "fused_swipe")}
        rows.append(example(19, "swipe", "venango", "", miss, strata=["short", "return_trip"]))

        unbound = evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False)
        self.assertIsNone(unbound["swipeVocabulary"])
        for name in ("swipe_lexicon_coverage", "swipe_in_lexicon_top1", "swipe_in_lexicon_top3",
                     "swipe_in_lexicon_short_top3", "swipe_in_lexicon_return_trip_top3"):
            self.assertFalse(unbound["checks"][name], name)

        # The place name is outside the vocabulary: it no longer counts against accuracy, and it
        # still counts against coverage, so a vocabulary cannot shrink its way past the gates.
        bound = evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False,
                         swipe_vocabulary=vocabulary("target"))
        self.assertEqual(0.75, bound["systems"]["fused_swipe"]["top1"])
        self.assertEqual(1.0, bound["systems"]["fused_swipe"]["inLexicon"]["top1"])
        self.assertEqual(0.5, bound["gates"]["swipeReturnTripTop3"])
        self.assertTrue(bound["checks"]["swipe_in_lexicon_top1"])
        self.assertTrue(bound["checks"]["swipe_in_lexicon_return_trip_top3"])
        self.assertEqual(0.75, bound["gates"]["swipeLexiconCoverage"])
        self.assertFalse(bound["checks"]["swipe_lexicon_coverage"])

        with self.assertRaisesRegex(EvaluationError, "not bound to the measured APK"):
            evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False,
                     swipe_vocabulary=vocabulary("target", apk="f" * 64))

    def test_unavailable_context_model_reports_neural_gates_and_scores_the_shipped_path(self):
        rows = self._minimum_rows()
        tap = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        rows[3] = example(14, "tap_error", "target", "raw", tap, environment_kind="grapheneos_hardware",
                          test_run_id="graphene-run", latency_overrides={"fused_personal": 81.0})

        def run(mode):
            return evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False,
                            swipe_vocabulary=vocabulary("target"), context_model=mode)

        qualifying, unavailable = run("qualifying"), run("unavailable")
        self.assertIn("neural_valid_word_absolute_gain", qualifying["checks"])
        self.assertTrue(qualifying["checks"]["tap_p95_latency"])
        for name in ("neural_valid_word_relative_error_reduction", "neural_valid_word_absolute_gain",
                     "false_correction_ceiling"):
            self.assertNotIn(name, unavailable["checks"])
            self.assertIn(name, unavailable["notApplicable"])
        self.assertIn("neuralValidWordAbsoluteGain", unavailable["gates"])
        self.assertEqual("fused_personal", unavailable["contextModel"]["shippedTapSystem"])
        self.assertFalse(unavailable["checks"]["tap_p95_latency"])
        with self.assertRaisesRegex(EvaluationError, "context model mode"):
            run("disabled")

    def test_shipped_tap_path_may_not_add_false_corrections_over_classic(self):
        rows = self._minimum_rows()
        rows.append(example(20, "valid_word", "form", "form", {
            "heliboard": ["form"], "fused": ["form"],
            "fused_personal": ["from", "form"], "fused_neural": ["form"],
        }, should_correct=False))
        result = evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False,
                          swipe_vocabulary=vocabulary("target"), context_model="unavailable")
        self.assertEqual(0.5, result["gates"]["shippedFalseCorrectionIncreaseOverClassic"])
        self.assertFalse(result["checks"]["shipped_false_correction_ceiling"])

    def test_swipe_vocabulary_is_bound_to_the_dictionary_the_measured_apk_ships(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            apk = root / "measured.apk"
            with zipfile.ZipFile(apk, "w") as archive:
                archive.writestr("assets/dicts/main_en-US.dict", b"dictionary bytes")
            words = [
                {"word": "Target", "languageTag": "en-US", "frequency": 200, "possiblyOffensive": False},
                {"word": "rude", "languageTag": "en-US", "frequency": 90, "possiblyOffensive": True},
            ]

            def export(dictionary_sha256, maximum=100_000):
                path = root / "export.json"
                path.write_text(json.dumps({
                    "schemaVersion": 1, "source": "bundled-static-dictionary",
                    "dictionaryAsset": "dicts/main_en-US.dict", "dictionarySha256": dictionary_sha256,
                    "apkSha256": "0" * 64, "visited": 2, "maximumWords": maximum, "words": words,
                }), encoding="utf-8")
                return path

            shipped = hashlib.sha256(b"dictionary bytes").hexdigest()
            loaded = load_swipe_vocabulary(export(shipped), apk)
            self.assertEqual(frozenset({"target"}), loaded.words)
            self.assertEqual(hashlib.sha256(apk.read_bytes()).hexdigest(), loaded.provenance["measuredApkSha256"])
            with self.assertRaisesRegex(EvaluationError, "different dictionary"):
                load_swipe_vocabulary(export("1" * 64), apk)
            with self.assertRaisesRegex(EvaluationError, "bound differs from production"):
                load_swipe_vocabulary(export(shipped, maximum=200_000), apk)

    def test_session_crossing_splits_is_rejected(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        rows = [
            example(1, "tap_error", "target", "raw", systems, split="train", session="same"),
            example(2, "tap_error", "target", "raw", systems, session="same"),
        ]
        with self.assertRaisesRegex(EvaluationError, "sessions cross"):
            evaluate(
                rows,
                metadata(),
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_missing_valid_word_label_is_rejected(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        with self.assertRaisesRegex(EvaluationError, "shouldCorrect"):
            example(1, "valid_word", "target", "raw", systems)

    def test_measured_tap_must_preserve_exact_raw_candidate(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "Raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        with self.assertRaisesRegex(EvaluationError, "fused does not preserve the exact raw"):
            example(1, "tap_error", "target", "raw", systems)

    def test_measurement_rows_reject_cross_path_systems_and_duplicate_candidates(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
            "ctc": ["target"],
        }
        with self.assertRaisesRegex(EvaluationError, "exactly the applicable systems"):
            example(1, "tap_error", "target", "raw", systems)

        systems.pop("ctc")
        systems["fused"] = ["target", "raw", "RAW"]
        with self.assertRaisesRegex(EvaluationError, "normalization-distinct"):
            example(2, "tap_error", "target", "raw", systems)

    def test_should_correct_is_rejected_outside_valid_word_rows(self):
        systems = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        with self.assertRaisesRegex(EvaluationError, "only for valid_word"):
            example(1, "tap_error", "target", "raw", systems, should_correct=True)

    def test_jsonl_reader_rejects_unbounded_or_invalid_utf8_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "measurements.jsonl"
            path.write_bytes(b"x" * (1024 * 1024 + 1) + b"\n")
            with self.assertRaisesRegex(EvaluationError, "byte limit"):
                read_jsonl(path)

            path.write_bytes(b"\xff\n")
            with self.assertRaisesRegex(EvaluationError, "invalid UTF-8"):
                read_jsonl(path)

    def test_release_swipe_strata_require_substantial_coverage(self):
        swipe = {system: ["target"] for system in ("geometric", "ctc", "fused_swipe")}
        rows = [
            example(100 + index, "swipe", "target", "", swipe, strata=["short", "clean"])
            for index in range(500)
        ]
        with self.assertRaisesRegex(EvaluationError, "medium=0<500"):
            validate_swipe_strata(rows, {
                "short": 500,
                "medium": 500,
                "clean": 500,
            })

    def test_missing_reference_environment_is_rejected(self):
        value = metadata()
        value["environments"] = value["environments"][:-1]
        with self.assertRaisesRegex(EvaluationError, "three reference environments"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_grapheneos_run_with_google_play_is_rejected(self):
        value = metadata()
        value["environments"][1]["sandboxedGooglePlayInstalled"] = True
        with self.assertRaisesRegex(EvaluationError, "without sandboxed Google Play"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_unbound_artifact_hash_is_rejected(self):
        value = metadata()
        value["coreApkSha256"] = "not-a-hash"
        with self.assertRaisesRegex(EvaluationError, "coreApkSha256"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_emulator_must_be_reported_as_low_ram(self):
        value = metadata()
        value["environments"][2]["isLowRamDevice"] = False
        with self.assertRaisesRegex(EvaluationError, "low-RAM emulator"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_measurement_run_must_match_declared_environment(self):
        rows = self._minimum_rows()
        rows[0] = example(
            11,
            "tap_error",
            "target",
            "raw",
            {system: list(values) for system, values in rows[0].predictions.items()},
            environment_kind="grapheneos_hardware",
            test_run_id="stock-run",
        )
        with self.assertRaisesRegex(EvaluationError, "environmentKind disagrees"):
            evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False)

    def test_release_evidence_requires_substantial_measurements_from_each_environment(self):
        with self.assertRaisesRegex(EvaluationError, "measurement coverage for stock_android_hardware"):
            evaluate(
                self._minimum_rows(),
                metadata(),
                measurement_sha256="e" * 64,
                enforce_minimum_counts=True,
            )

    def test_metadata_rejects_duplicate_run_ids(self):
        value = metadata()
        value["environments"][1]["testRunId"] = "stock-run"
        with self.assertRaisesRegex(EvaluationError, "duplicate testRunId"):
            evaluate(
                self._minimum_rows(),
                value,
                measurement_sha256="e" * 64,
                enforce_minimum_counts=False,
            )

    def test_peak_memory_is_derived_from_the_slowest_environment(self):
        value = metadata()
        value["environments"][1]["peakAddedNeuralMemoryMiB"] = 65
        result = evaluate(
            self._minimum_rows(),
            value,
            measurement_sha256="e" * 64,
            enforce_minimum_counts=False,
        )
        self.assertEqual(65.0, result["evidence"]["peakAddedNeuralMemoryMiB"])
        self.assertFalse(result["checks"]["peak_neural_memory"])

    def test_each_environment_controls_its_own_latency_gate(self):
        rows = self._minimum_rows()
        tap = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        rows.extend(
            example(100 + index, "tap_error", "target", "raw", tap,
                    latency_overrides={"fused_neural": 1.0})
            for index in range(30)
        )
        rows[3] = example(
            14,
            "tap_error",
            "target",
            "raw",
            tap,
            environment_kind="grapheneos_hardware",
            test_run_id="graphene-run",
            latency_overrides={"fused_neural": 81.0},
        )
        result = evaluate(rows, metadata(), measurement_sha256="e" * 64, enforce_minimum_counts=False)
        self.assertLessEqual(result["systems"]["fused_neural"]["latencyMs"]["p95"], 80.0)
        self.assertEqual(81.0, result["environmentLatencyMs"]["grapheneos_hardware"]["tap"]["p95"])
        self.assertFalse(result["checks"]["tap_p95_latency"])

    @staticmethod
    def _minimum_rows():
        tap = {
            "heliboard": ["target"],
            "fused": ["target", "raw"],
            "fused_personal": ["target", "raw"],
            "fused_neural": ["target", "raw"],
        }
        swipe = {system: ["target"] for system in ("geometric", "ctc", "fused_swipe")}
        return [
            example(11, "tap_error", "target", "raw", tap),
            example(12, "valid_word", "target", "target", tap, should_correct=False),
            example(13, "swipe", "target", "", swipe, strata=["short", "return_trip"]),
            example(14, "tap_error", "target", "raw", tap,
                    environment_kind="grapheneos_hardware", test_run_id="graphene-run"),
            example(15, "swipe", "target", "", swipe, strata=["medium"],
                    environment_kind="grapheneos_hardware", test_run_id="graphene-run"),
            example(16, "tap_error", "target", "raw", tap,
                    environment_kind="low_ram_emulator", test_run_id="low-ram-run"),
            example(17, "swipe", "target", "", swipe, strata=["long"],
                    environment_kind="low_ram_emulator", test_run_id="low-ram-run"),
            example(18, "valid_word", "there", "their", {
                "heliboard": ["their"],
                "fused": ["their"],
                "fused_personal": ["their"],
                "fused_neural": ["there", "their"],
            }, should_correct=True),
        ]


if __name__ == "__main__":
    unittest.main()
