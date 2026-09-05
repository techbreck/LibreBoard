# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


TOOLS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import context_tokenizer_contract as contract  # noqa: E402


def tokenizer_document() -> dict:
    return {
        "schemaVersion": 1,
        "normalization": "NFKC_LOWER",
        "vocabulary": {
            "<pad>": 0,
            "<bos>": 1,
            "<unk>": 2,
            "<lang:en>": 3,
            "<lang:de>": 4,
            "▁": 5,
            "h": 6,
            "▁h": 7,
        },
        "merges": [["▁", "h"]],
        "specialTokens": {
            "padding": "<pad>",
            "beginningOfSequence": "<bos>",
            "unknown": "<unk>",
            "languages": {"en": "<lang:en>", "de": "<lang:de>"},
        },
    }


class ContextTokenizerContractTest(unittest.TestCase):
    def write(self, root: pathlib.Path, value: object) -> pathlib.Path:
        path = root / "tokenizer.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_accepts_android_compatible_bilingual_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = contract.load_tokenizer(
                self.write(pathlib.Path(temporary), tokenizer_document()),
                expected_vocabulary_size=8,
            )
            self.assertEqual(8, len(tokenizer.raw["vocabulary"]))
            self.assertEqual(64, len(tokenizer.sha256))

    def test_encoder_normalizes_merges_and_uses_unknown_tokens_like_android(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = contract.load_tokenizer(
                self.write(pathlib.Path(temporary), tokenizer_document()),
                expected_vocabulary_size=8,
            )
            self.assertEqual([7, 5, 2], contract.encode_text(tokenizer, " Ｈ x "))

    def test_rejects_sparse_ids_language_drift_and_missing_merge_outputs(self):
        mutations = []
        sparse = tokenizer_document()
        sparse["vocabulary"]["h"] = 99
        mutations.append((sparse, "unique and dense"))
        language = tokenizer_document()
        language["specialTokens"]["languages"] = {"en": "<lang:en>"}
        mutations.append((language, "exactly English and German"))
        merge = tokenizer_document()
        merge["merges"] = [["h", "h"]]
        mutations.append((merge, "merge output is absent"))

        for value, message in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(contract.ContextTokenizerContractError, message):
                    contract.load_tokenizer(self.write(pathlib.Path(temporary), value))

    def test_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "tokenizer.json"
            path.write_text('{"schemaVersion":1,"schemaVersion":1}', encoding="utf-8")
            with self.assertRaisesRegex(contract.ContextTokenizerContractError, "duplicate key"):
                contract.load_tokenizer(path)


if __name__ == "__main__":
    unittest.main()
