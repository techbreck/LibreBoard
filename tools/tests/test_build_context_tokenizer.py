# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path[:0] = [str(TOOLS), str(ROOT)]
import build_context_tokenizer as builder  # noqa: E402
import context_tokenizer_contract  # noqa: E402
from tools.tests.test_prepare_context_dataset import PreparedFixture, canonical  # noqa: E402


class BuildContextTokenizerTest(unittest.TestCase):
    def test_default_environment_has_actionable_optional_dependency_error(self):
        if importlib.util.find_spec("tokenizers") is not None:
            self.skipTest("the optional context tokenizer toolchain is active")
        with self.assertRaisesRegex(builder.ContextTokenizerBuildError, "requirements-context"):
            builder._dependencies()

    def test_policy_rejects_special_token_and_vocabulary_drift(self):
        fixture = PreparedFixture()
        try:
            value = json.loads(builder.DEFAULT_POLICY.read_text())
            value["specialTokens"] = list(reversed(value["specialTokens"]))
            path = fixture.root / "tokenizer-policy.json"
            path.write_bytes(canonical(value))
            with self.assertRaisesRegex(builder.ContextTokenizerBuildError, "special-token order"):
                builder.load_policy(path)

            value = json.loads(builder.DEFAULT_POLICY.read_text())
            value["vocabularySize"] = 16_385
            path.write_bytes(canonical(value))
            with self.assertRaisesRegex(builder.ContextTokenizerBuildError, "vocabularySize"):
                builder.load_policy(path)
        finally:
            fixture.close()

    @unittest.skipUnless(importlib.util.find_spec("tokenizers") is not None, "optional tokenizers is not installed")
    def test_fixture_build_is_deterministic_bounded_and_android_compatible(self):
        fixture = PreparedFixture()
        try:
            fixture.prepare()
            value = json.loads(builder.DEFAULT_POLICY.read_text())
            value.update({"vocabularySize": 64, "minimumFrequency": 1, "alphabetLimit": 128})
            policy_path = fixture.root / "tokenizer-policy.json"
            policy_path.write_bytes(canonical(value))

            first = builder.build(
                data_root=fixture.output_root,
                policy_path=policy_path,
                development=True,
            )
            tokenizer_path = fixture.output_root / "tokenizer-development.json"
            first_bytes = tokenizer_path.read_bytes()
            validated = context_tokenizer_contract.load_tokenizer(
                tokenizer_path,
                expected_vocabulary_size=64,
            )
            self.assertFalse(first["releaseEligible"])
            self.assertEqual(validated.sha256, first["tokenizer"]["sha256"])
            self.assertLessEqual(first["tokenizer"]["bytes"], 2 * 1024 * 1024)
            self.assertGreaterEqual(first["runtimeParityProbes"], 4)

            second = builder.build(
                data_root=fixture.output_root,
                policy_path=policy_path,
                development=True,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_bytes, tokenizer_path.read_bytes())
        finally:
            fixture.close()

    @unittest.skipUnless(importlib.util.find_spec("tokenizers") is not None, "optional tokenizers is not installed")
    def test_release_build_requires_the_exact_pinned_corpus(self):
        fixture = PreparedFixture()
        try:
            fixture.prepare()
            value = json.loads(builder.DEFAULT_POLICY.read_text())
            value.update({"vocabularySize": 64, "minimumFrequency": 1, "alphabetLimit": 128})
            policy_path = fixture.root / "tokenizer-policy.json"
            policy_path.write_bytes(canonical(value))
            wrong_pinned = fixture.root / "wrong-pinned.json"
            wrong_pinned.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(builder.prepare_context_dataset.ContextDataError, "does not match"):
                builder.build(
                    data_root=fixture.output_root,
                    policy_path=policy_path,
                    development=False,
                    pinned_manifest_path=wrong_pinned,
                )
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
