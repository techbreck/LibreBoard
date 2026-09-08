# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import audit_joint_model_splits as audit


class JointSplitAuditTest(unittest.TestCase):
    def test_training_and_validation_exposure_are_both_excluded(self):
        rows = [{"sessionId": split, "strata": ["short"]} for split in audit.CONTEXT_SPLITS]
        result = audit.count_overlap(rows, {split: split for split in audit.CONTEXT_SPLITS})
        self.assertEqual(2, result["contextUnseenRows"])
        self.assertEqual(2, result["contextUnseenStrata"]["short"])
        self.assertEqual(1, result["contextSplitRows"]["train"])
        self.assertEqual(1, result["contextSplitRows"]["validation"])

    def test_unmapped_session_is_not_treated_as_absent(self):
        with self.assertRaises(audit.JointSplitAuditError):
            audit.count_overlap([{"sessionId": "unknown", "strata": []}], {})

    def test_unknown_context_split_is_rejected(self):
        with self.assertRaises(audit.JointSplitAuditError):
            audit.count_overlap([{"sessionId": "session", "strata": []}], {"session": "other"})

    def test_altered_prepared_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "test.jsonl"
            payload = (json.dumps({"split": "test"}) + "\n").encode()
            path.write_bytes(payload)
            details = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "records": 1}
            self.assertEqual(1, len(list(audit.verified_rows(path, details, "test"))))
            path.write_bytes(payload.replace(b': ', b' :'))
            with self.assertRaises(audit.JointSplitAuditError):
                list(audit.verified_rows(path, details, "test"))


if __name__ == "__main__":
    unittest.main()
