# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import evaluate_swipe_context as diagnostic
import evaluate_swipe_ctc as swipe


class SwipeContextTest(unittest.TestCase):
    def test_zero_context_replays_static_fusion_and_missing_scores_are_neutral(self):
        candidates = [{'word': word, 'language': 'en', 'frequency': frequency, 'spatial': spatial}
                      for word, frequency, spatial in [('cat', 12, -.1), ('car', 80, -.7), ('can', 60, -.5)]]
        entries = [swipe.ScoredLexiconEntry(swipe.LexiconEntry(c['word'], c['language'], (), c['frequency']), c['spatial'])
                   for c in candidates]
        expected = [c.word for c in swipe.rank_static_fusion(entries)]
        for scores, coefficient in [([8., 1., -3.], 0.), ([None] * 3, 1.2)]:
            ranked = [candidates[i]['word'] for i in diagnostic.rank(candidates, scores, coefficient)]
            self.assertEqual(expected, ranked)
        self.assertEqual([0., 1., 0.], diagnostic.normalized_optional([None, 4., None]))

    def test_prefix_join_rejects_wrong_session_even_when_target_matches(self):
        row = {'id': 'a', 'sessionId': 'one', 'target': 'cat', 'language': 'en', 'strata': ['short']}
        prefix = {'id': 'a', 'sessionId': 'two', 'target': 'cat', 'languageTag': 'en-US',
                  'strata': ['short'], 'contextOrigin': 'publisher_reference_prefix'}
        with self.assertRaisesRegex(ValueError, 'identity'):
            list(diagnostic.joined_rows([row], {'a': prefix}))
        prefix['sessionId'] = 'one'
        self.assertEqual([(row, prefix)], list(diagnostic.joined_rows([row], {'a': prefix})))
        self.assertEqual([], list(diagnostic.joined_rows([row], {})))

    def test_slate_hash_and_unique_identity_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / 'slates.jsonl'
            row = {'id': 'a', 'merged': [{'word': 'cat', 'language': 'en', 'frequency': 10, 'spatial': -1.}]}
            payload = (json.dumps(row) + '\n').encode()
            path.write_bytes(payload)
            details = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'records': 1}
            self.assertEqual([row], diagnostic.checked_slates(path, details))
            path.write_bytes(payload.replace(b'cat', b'car'))
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                diagnostic.checked_slates(path, details)
            path.write_bytes(payload * 2)
            details.update(bytes=len(payload) * 2, sha256=hashlib.sha256(payload * 2).hexdigest(), records=2)
            with self.assertRaisesRegex(ValueError, 'identities'):
                diagnostic.checked_slates(path, details)
