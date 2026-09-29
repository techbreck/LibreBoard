# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_engine  # noqa: E402
from run_phase0_measurement import (  # noqa: E402
    MeasurementRunError,
    compile_package,
    load_artifact_pin,
    parse_instrumentation_result,
    pinned_hash,
    verify_measurement_rows,
)

REPOSITORY_PIN = pathlib.Path(__file__).resolve().parents[2] / "docs" / "phase-0-artifact-pin.json"


def instrumentation(*status_codes: int, result: int = -1) -> str:
    lines = []
    for code in status_codes:
        lines.append("INSTRUMENTATION_STATUS: class=Phase0MeasurementInstrumentedTest")
        lines.append(f"INSTRUMENTATION_STATUS_CODE: {code}")
    lines.append(f"INSTRUMENTATION_CODE: {result}")
    return "\n".join(lines) + "\n"


def row(**overrides) -> str:
    value = {
        "schemaVersion": evaluate_engine.SCHEMA_VERSION,
        "id": "example-1",
        "category": "tap_error",
        "testRunId": "graphene-run",
        "environmentKind": "grapheneos_hardware",
    }
    value.update(overrides)
    return json.dumps(value, sort_keys=True)


class ParseInstrumentationResultTest(unittest.TestCase):
    def test_a_clean_run_reports_its_counts(self):
        counts = parse_instrumentation_result(instrumentation(1, 0, 1, 0), 2)
        self.assertEqual(
            {"passed": 2, "failed": 0, "skipped": 0, "started": 2, "runnerResultCode": -1}, counts)

    def test_the_process_being_reaped_after_the_tests_pass_is_not_a_crash(self):
        # The system kills the app process "due to finished inst" once the runner completes, and
        # `am` reports that as shortMsg=Process crashed with code 0 even though every test passed
        # and its rows were written. The per-test stream is the authority.
        counts = parse_instrumentation_result(instrumentation(1, 0, 1, -4, result=0), 1)
        self.assertEqual(1, counts["passed"])
        self.assertEqual(0, counts["runnerResultCode"])

    def test_a_requested_test_that_skips_is_not_a_measurement_run(self):
        # The gate arguments failing to reach the runner is the quiet failure this catches: the
        # test skips itself by assumption, adb still exits zero, and a previous output file stays
        # on the device.
        for code in (-3, -4):
            with self.subTest(code=code), self.assertRaisesRegex(MeasurementRunError, "were skipped"):
                parse_instrumentation_result(instrumentation(1, code), 1)

    def test_the_unrequested_replay_test_may_skip(self):
        # Requesting only the tap corpus still runs the class, and the swipe replay correctly
        # skips itself by assumption. That is a complete run, not a failed one.
        counts = parse_instrumentation_result(instrumentation(1, 0, 1, -4), 1)
        self.assertEqual(
            {"passed": 1, "failed": 0, "skipped": 1, "started": 2, "runnerResultCode": -1}, counts)

    def test_test_failures_are_rejected(self):
        for code in (-1, -2):
            with self.subTest(code=code), self.assertRaisesRegex(MeasurementRunError, "failed"):
                parse_instrumentation_result(instrumentation(1, code), 1)

    def test_a_crash_before_the_tests_finish_is_rejected(self):
        # No test ever reported OK, so the run genuinely died mid-flight.
        with self.assertRaisesRegex(MeasurementRunError, "did not finish"):
            parse_instrumentation_result("INSTRUMENTATION_STATUS_CODE: 1\n", 1)
        with self.assertRaisesRegex(MeasurementRunError, "exited with code 0"):
            parse_instrumentation_result(instrumentation(1, result=0), 1)

    def test_a_partial_run_cannot_pass_for_a_complete_one(self):
        # Both corpora were requested but only the tap test ran.
        with self.assertRaisesRegex(MeasurementRunError, "expected 2 instrumented"):
            parse_instrumentation_result(instrumentation(1, 0), 2)


class CompilePackageTest(unittest.TestCase):
    def test_unsafe_filters_are_rejected_before_adb(self):
        with self.assertRaisesRegex(MeasurementRunError, "unsafe compile filter"):
            compile_package("adb", None, "speed; reboot")


class VerifyMeasurementRowsTest(unittest.TestCase):
    def test_matching_rows_are_counted_by_category(self):
        payload = (row() + "\n" + row(id="example-2", category="swipe") + "\n").encode()
        self.assertEqual(
            {"tap": 1, "swipe": 1},
            verify_measurement_rows(payload, "graphene-run", "grapheneos_hardware"),
        )

    def test_another_runs_output_is_rejected(self):
        payload = (row(testRunId="stale-run") + "\n").encode()
        with self.assertRaisesRegex(MeasurementRunError, "another run's output"):
            verify_measurement_rows(payload, "graphene-run", "grapheneos_hardware")

    def test_environment_mismatch_is_rejected(self):
        payload = (row(environmentKind="stock_android_hardware") + "\n").encode()
        with self.assertRaisesRegex(MeasurementRunError, "environmentKind"):
            verify_measurement_rows(payload, "graphene-run", "grapheneos_hardware")

    def test_an_older_schema_is_rejected(self):
        payload = (row(schemaVersion=evaluate_engine.SCHEMA_VERSION - 1) + "\n").encode()
        with self.assertRaisesRegex(MeasurementRunError, "expected schemaVersion"):
            verify_measurement_rows(payload, "graphene-run", "grapheneos_hardware")

    def test_truncated_output_is_rejected(self):
        payload = (row() + "\n" + row(id="example-2")[:40]).encode()
        with self.assertRaisesRegex(MeasurementRunError, "invalid JSON"):
            verify_measurement_rows(payload, "graphene-run", "grapheneos_hardware")

    def test_blank_lines_are_tolerated(self):
        payload = ("\n" + row() + "\n\n").encode()
        self.assertEqual(
            {"tap": 1, "swipe": 0},
            verify_measurement_rows(payload, "graphene-run", "grapheneos_hardware"),
        )


class ArtifactPinTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="libreboard-pin-")
        self.addCleanup(self.temporary.cleanup)
        self.path = pathlib.Path(self.temporary.name) / "pin.json"

    def write(self, **overrides) -> pathlib.Path:
        pin = {
            "schemaVersion": 1,
            "candidateId": "context-shared-session-v2",
            "contextModelSha256": "8" * 64,
            "contextTokenizerSha256": "e" * 64,
            "swipeModelSha256": "1" * 64,
            "rejectedContextModelSha256": {"b" * 64: "superseded independent-split export"},
        }
        pin.update(overrides)
        self.path.write_text(json.dumps(pin), encoding="utf-8")
        return self.path

    def test_the_repository_pin_loads(self):
        pin = load_artifact_pin(REPOSITORY_PIN)
        self.assertEqual("context-shared-session-v2", pin["candidateId"])
        # The artifact the superseded GrapheneOS run actually measured must stay named as rejected.
        self.assertIn(
            "b6dde70259d75790d8685884a50b9939f88befd55405fa3e78f6d8edebc033df",
            pin["rejectedContextModelSha256"],
        )
        self.assertNotIn(pin["contextModelSha256"], pin["rejectedContextModelSha256"])

    def test_the_pin_overrides_nothing_and_contradicts_nothing(self):
        pin = load_artifact_pin(self.write())
        self.assertEqual("8" * 64, pinned_hash(pin, "contextModelSha256", None, "context-model-sha256"))
        self.assertEqual("8" * 64, pinned_hash(pin, "contextModelSha256", "8" * 64, "context-model-sha256"))
        with self.assertRaisesRegex(MeasurementRunError, "artifact pin requires"):
            pinned_hash(pin, "contextModelSha256", "c" * 64, "context-model-sha256")
        # Without a pin the caller's own value stands; that path is diagnostic-only by design.
        self.assertEqual("c" * 64, pinned_hash(None, "contextModelSha256", "c" * 64, "x"))

    def test_a_pin_that_rejects_its_own_candidate_is_refused(self):
        path = self.write(rejectedContextModelSha256={"8" * 64: "contradiction"})
        with self.assertRaisesRegex(MeasurementRunError, "pins a context model it also rejects"):
            load_artifact_pin(path)

    def test_malformed_pins_are_refused(self):
        with self.assertRaisesRegex(MeasurementRunError, "schema-1 artifact pin"):
            load_artifact_pin(self.write(schemaVersion=2))
        with self.assertRaisesRegex(MeasurementRunError, "schema-1 artifact pin"):
            load_artifact_pin(self.write(unexpectedKey="value"))
        with self.assertRaisesRegex(MeasurementRunError, "lowercase contextModelSha256"):
            load_artifact_pin(self.write(contextModelSha256="NOTAHASH"))
        with self.assertRaisesRegex(MeasurementRunError, "malformed rejected"):
            load_artifact_pin(self.write(rejectedContextModelSha256={"b" * 64: 7}))
        missing = pathlib.Path(self.temporary.name) / "absent.json"
        with self.assertRaisesRegex(MeasurementRunError, "cannot read artifact pin"):
            load_artifact_pin(missing)


if __name__ == "__main__":
    unittest.main()
