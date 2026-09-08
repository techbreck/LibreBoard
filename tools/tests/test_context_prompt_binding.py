# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_context_model as evaluator


def scored(source_id, policy):
    binding = {'sourceRecordId': source_id, 'prefixTokenIds': [1, 2],
               'candidates': ['first', 'second'], 'policySha256': policy}
    identifier = hashlib.sha256(evaluator.train_context_model.score_context_teacher._canonical_line(binding)).hexdigest()
    return {'id': identifier, 'prefixTokenIds': [1, 2], 'candidates': [{'normalized': 'first'}, {'normalized': 'second'}]}


class PromptBindingTest(unittest.TestCase):
    def test_exact_binding_maps_only_requested_source_records(self):
        rows = [scored('source-a', 'policy'), scored('source-b', 'policy')]
        self.assertEqual({'source-b': rows[1]['id']}, evaluator.bind_prompt_exclusions(rows, ['source-b'], 'policy'))

    def test_changed_policy_and_missing_record_fail_closed(self):
        for policy, ids in [('changed', ['source-a']), ('policy', ['source-a', 'missing'])]:
            with self.subTest(policy=policy, ids=ids):
                with self.assertRaises(evaluator.ContextEvaluationError):
                    evaluator.bind_prompt_exclusions([scored('source-a', 'policy')], ids, policy)

    def test_empty_exclusions_still_consume_verified_stream(self):
        def rows():
            yield scored('source-a', 'policy')
            raise ValueError('stream hash mismatch at end')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            evaluator.bind_prompt_exclusions(rows(), [], 'policy')


if __name__ == '__main__':
    unittest.main()
