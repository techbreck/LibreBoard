# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_swipe_ctc as evaluator


class SlateEvidenceTest(unittest.TestCase):
    def args(self, directory):
        root = pathlib.Path(directory)
        return types.SimpleNamespace(slates_output=root / 'slates.jsonl', output=root / 'report.json')

    def test_complete_output_is_hash_bound_and_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            def evaluate(_, stream):
                stream.write(b'{"id":"one"}\n')
                return {'sample': {'rows': 1}}
            with mock.patch.object(evaluator, '_evaluate', side_effect=evaluate) as run:
                report = evaluator.evaluate(args)
                self.assertEqual(hashlib.sha256(args.slates_output.read_bytes()).hexdigest(), report['candidateSlates']['sha256'])
                self.assertEqual(1, report['candidateSlates']['records'])
                with self.assertRaises(evaluator.SwipeEvaluationError):
                    evaluator.evaluate(args)
                self.assertEqual(1, run.call_count)

    def test_failure_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            def fail(_, stream):
                stream.write(b'partial')
                raise evaluator.SwipeEvaluationError('inference failed')
            with mock.patch.object(evaluator, '_evaluate', side_effect=fail):
                with self.assertRaises(evaluator.SwipeEvaluationError):
                    evaluator.evaluate(args)
            self.assertEqual([], list(pathlib.Path(directory).iterdir()))

    def test_report_cannot_overwrite_slate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory)
            args.output = args.slates_output
            with self.assertRaises(evaluator.SwipeEvaluationError):
                evaluator.evaluate(args)

    def test_serialization_preserves_identity_score_and_order(self):
        row = evaluator.EvaluationRow('id', 'session', 'en', 'word', (), (), frozenset({'short', 'clean'}))
        slate = [evaluator.ScoredLexiconEntry(evaluator.LexiconEntry('word', 'en', (1,), 7), -2.5),
                 evaluator.ScoredLexiconEntry(evaluator.LexiconEntry('ward', 'en', (2,), 3), -3.25)]
        stream = io.BytesIO()
        evaluator._write_candidate_slate(stream, row, slate, [], slate)
        result = json.loads(stream.getvalue())
        self.assertEqual('session', result['sessionId'])
        self.assertEqual(['word', 'ward'], [item['word'] for item in result['merged']])
        self.assertEqual([-2.5, -3.25], [item['spatial'] for item in result['merged']])
        self.assertEqual([7, 3], [item['frequency'] for item in result['merged']])
        invalid = [evaluator.ScoredLexiconEntry(slate[0].entry, float('nan'))]
        with self.assertRaises(ValueError):
            evaluator._write_candidate_slate(io.BytesIO(), row, invalid, [], [])

    def test_reserved_slots_are_serialized_without_dropping_merged_candidates(self):
        row = evaluator.EvaluationRow('id', 'session', 'en', 'word', (), (), frozenset({'short'}))
        merged = [
            evaluator.ScoredLexiconEntry(evaluator.LexiconEntry(f'w{index}', 'en', (1,), 5), -0.1 * index)
            for index in range(32)
        ]
        reserved = [evaluator.ScoredLexiconEntry(
            evaluator.LexiconEntry('oovword', 'en', (2,), 0), 0.4, frequency_free=True,
        )]
        stream = io.BytesIO()
        evaluator._write_candidate_slate(stream, row, merged[:2], [], merged, reserved)
        result = json.loads(stream.getvalue())
        self.assertEqual(32, len(result['merged']))
        self.assertEqual([item.word for item in merged], [item['word'] for item in result['merged']])
        self.assertEqual(['oovword'], [item['word'] for item in result['reserved']])
        self.assertTrue(result['reserved'][0]['frequencyFree'])


if __name__ == '__main__':
    unittest.main()
