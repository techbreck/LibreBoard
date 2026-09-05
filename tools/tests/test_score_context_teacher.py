# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import context_tokenizer_contract  # noqa: E402
import score_context_teacher as scorer  # noqa: E402


class FakeTeacherTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    vocab_size = 32_000

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [10 + sum(map(ord, word)) % 100 for word in text.split()]


def student_tokenizer() -> context_tokenizer_contract.ContextTokenizer:
    tokens = ["<pad>", "<bos>", "<unk>", "<lang:en>", "<lang:de>", "▁"]
    tokens.extend("abcdefghijklmnopqrstuvwxyzäöüß'-")
    vocabulary = {token: index for index, token in enumerate(tokens)}
    return context_tokenizer_contract.ContextTokenizer(
        path=pathlib.Path("fixture-tokenizer.json"),
        sha256="a" * 64,
        raw={
            "schemaVersion": 1,
            "normalization": "NFKC_LOWER",
            "vocabulary": vocabulary,
            "merges": [],
            "specialTokens": {
                "padding": "<pad>",
                "beginningOfSequence": "<bos>",
                "unknown": "<unk>",
                "languages": {"en": "<lang:en>", "de": "<lang:de>"},
            },
        },
        merge_ranks={},
        unknown_token_id=vocabulary["<unk>"],
    )


def record(text: str, language: str = "en-US") -> dict:
    return {
        "schemaVersion": 1,
        "id": "1" * 64,
        "sessionId": "2" * 64,
        "split": "train",
        "language": language,
        "fieldClass": "plain",
        "text": text,
        "sourceId": "fixture-source",
    }


class ScoreContextTeacherTest(unittest.TestCase):
    def test_policy_rejects_tokenizer_and_candidate_limit_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "policy.json"
            value = json.loads(scorer.DEFAULT_POLICY.read_text())
            value["tokenizerSha256"] = "not-a-hash"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(scorer.ContextTeacherError, "tokenizer hash"):
                scorer.load_policy(path)

            value = json.loads(scorer.DEFAULT_POLICY.read_text())
            value["minimumCandidates"] = 9
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(scorer.ContextTeacherError, "inverted"):
                scorer.load_policy(path)

    def test_candidate_slate_is_deterministic_bounded_and_language_scoped(self):
        policy = scorer.load_policy()
        confusion = {"wahr": ("war",)}
        arguments = {
            "lexicon": ["war", "Wert", "Wort", "sehr", "noch", "dort", "hier", "auch"],
            "confusion_map": confusion,
            "aliases": {},
            "student_tokenizer": student_tokenizer(),
            "teacher_tokenizer": FakeTeacherTokenizer(),
            "policy": policy,
        }
        first, rejection = scorer._prepare_example(record("es wahr", "de"), **arguments)
        second, _rejection = scorer._prepare_example(record("es wahr", "de"), **arguments)
        self.assertIsNone(rejection)
        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual("de", first.language)
        self.assertEqual("observed", first.candidates[0].sources[0])
        self.assertIn("war", [candidate.normalized for candidate in first.candidates])
        self.assertGreaterEqual(len(first.candidates), policy.minimum_candidates)
        self.assertLessEqual(len(first.candidates), policy.maximum_candidates)
        self.assertTrue(all(
            len(candidate.student_token_ids) <= policy.maximum_candidate_student_tokens
            for candidate in first.candidates
        ))

    def test_output_document_contains_no_teacher_prefix_or_source_session(self):
        policy = scorer.load_policy()
        example, rejection = scorer._prepare_example(
            record("we test words"),
            lexicon=["word", "work", "test", "text", "best", "west", "rest", "nest"],
            confusion_map={},
            aliases={},
            student_tokenizer=student_tokenizer(),
            teacher_tokenizer=FakeTeacherTokenizer(),
            policy=policy,
        )
        self.assertIsNone(rejection)
        assert example is not None
        output = example.output_document([-1.0] * len(example.candidates))
        self.assertEqual(scorer.OUTPUT_KEYS, set(output))
        self.assertNotIn("teacherPrefixTokenIds", output)
        self.assertNotIn("sessionId", output)
        self.assertTrue(all(set(candidate) == scorer.CANDIDATE_KEYS for candidate in output["candidates"]))

    def test_default_environment_has_actionable_optional_dependency_error(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("the optional context teacher toolchain is active")
        with self.assertRaisesRegex(scorer.ContextTeacherError, "requirements-context"):
            scorer._dependencies()

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "optional model toolchain is not installed")
    def test_teacher_score_batch_is_finite_and_preserves_slate_shape(self):
        import torch

        class FakeModel:
            def __call__(self, input_ids, attention_mask, use_cache):
                del attention_mask, use_cache
                vocabulary = 128
                logits = torch.arange(vocabulary, dtype=torch.float32).reshape(1, 1, -1)
                logits = logits.expand(input_ids.shape[0], input_ids.shape[1], -1).clone()
                return type("Output", (), {"logits": logits})()

        candidate = scorer.PreparedCandidate("one", "one", ("observed",), (5,), (20,))
        example = scorer.PreparedExample(
            identifier="3" * 64,
            split="train",
            language="en-US",
            field_class_id=0,
            source_id="fixture-source",
            prefix_student_token_ids=(5,),
            prefix_teacher_token_ids=(11, 12),
            candidates=(candidate, dataclasses.replace(candidate, surface="two", normalized="two", teacher_token_ids=(21,))),
        )
        scores = scorer._score_rows([example], FakeModel(), FakeTeacherTokenizer(), torch)
        self.assertEqual((1, 2), (len(scores), len(scores[0])))
        self.assertTrue(all(map(lambda value: value < 0, scores[0])))
        self.assertGreater(scores[0][1], scores[0][0])


if __name__ == "__main__":
    unittest.main()
