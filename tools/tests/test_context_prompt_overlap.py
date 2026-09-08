# SPDX-License-Identifier: GPL-3.0-only
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from audit_context_prompt_overlap import compare_prompts


def row(identifier, text, language='en-US'):
    return {'id': identifier, 'text': text, 'language': language, 'sourceId': 'source'}


class PromptOverlapTest(unittest.TestCase):
    def test_normalized_match_does_not_require_same_session(self):
        counts, excluded = compare_prompts({'train': [row('a', 'ＦＯＯ  bar')],
            'validation': [row('b', 'foo\tBAR'), row('c', 'foo bar', 'de')], 'test': []})
        self.assertEqual(['b'], excluded['validation'])
        self.assertEqual(1, counts['validation']['source']['rowsWithPromptInTraining'])

    def test_test_excludes_union_without_double_counting(self):
        _, excluded = compare_prompts({'train': [row('a', 'shared')],
            'validation': [row('b', 'shared'), row('c', 'validation only')],
            'test': [row('d', 'shared'), row('e', 'validation only'), row('f', 'new')]})
        self.assertEqual(['b'], excluded['validation'])
        self.assertEqual(['d', 'e'], excluded['test'])

    def test_partial_prefix_is_not_claimed_as_exact_overlap(self):
        _, excluded = compare_prompts({'train': [row('a', 'this prefix ends here')],
            'validation': [row('b', 'this prefix ends elsewhere')], 'test': []})
        self.assertEqual([], excluded['validation'])


if __name__ == '__main__':
    unittest.main()
