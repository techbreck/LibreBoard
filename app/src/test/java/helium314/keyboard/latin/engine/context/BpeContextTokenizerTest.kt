// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.context

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.test.assertFailsWith

class BpeContextTokenizerTest {
    @Test
    fun appliesRankedMergesNfkcAndLanguageFallback() {
        val tokenizer = BpeContextTokenizer.fromJson(document())

        assertArrayEquals(intArrayOf(11), tokenizer.encode("ＨELLO", 8, TokenTruncation.KEEP_START))
        assertEquals(3, tokenizer.languageTokenId("en-US"))
        assertEquals(4, tokenizer.languageTokenId("de-DE"))
        assertEquals(0, tokenizer.paddingTokenId)
        assertEquals(1, tokenizer.beginningOfSequenceTokenId)
    }

    @Test
    fun truncationDirectionIsExplicitAndUnknownCharactersStayBounded() {
        val tokenizer = BpeContextTokenizer.fromJson(document())

        assertArrayEquals(
            intArrayOf(11, 11),
            tokenizer.encode("hello hello hello", 2, TokenTruncation.KEEP_START),
        )
        assertArrayEquals(
            intArrayOf(11, 11),
            tokenizer.encode("x hello hello", 2, TokenTruncation.KEEP_END),
        )
        assertArrayEquals(intArrayOf(5, 2), tokenizer.encode("x", 8, TokenTruncation.KEEP_START))
    }

    @Test
    fun rejectsSchemaDriftSparseIdsAndInvalidMerges() {
        val valid = Json.parseToJsonElement(document().decodeToString()).jsonObject
        val unknownField = valid.toMutableMap().apply { put("unexpected", Json.parseToJsonElement("true")) }
        assertFailsWith<Exception> {
            BpeContextTokenizer.fromJson(Json.encodeToString(unknownField).encodeToByteArray())
        }

        val sparse = document().decodeToString().replace("\"▁hello\": 11", "\"▁hello\": 99")
        assertFailsWith<IllegalArgumentException> {
            BpeContextTokenizer.fromJson(sparse.encodeToByteArray())
        }

        val missingMergeOutput = document().decodeToString()
            .replace("[\"▁hell\", \"o\"]", "[\"▁hello\", \"o\"]")
        assertFailsWith<IllegalArgumentException> {
            BpeContextTokenizer.fromJson(missingMergeOutput.encodeToByteArray())
        }
    }

    private fun document(): ByteArray = """
        {
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
            "e": 7,
            "l": 8,
            "o": 9,
            "▁h": 10,
            "▁hello": 11,
            "▁he": 12,
            "▁hel": 13,
            "▁hell": 14
          },
          "merges": [
            ["▁", "h"],
            ["▁h", "e"],
            ["▁he", "l"],
            ["▁hel", "l"],
            ["▁hell", "o"]
          ],
          "specialTokens": {
            "padding": "<pad>",
            "beginningOfSequence": "<bos>",
            "unknown": "<unk>",
            "languages": {
              "en": "<lang:en>",
              "de": "<lang:de>"
            }
          }
        }
    """.trimIndent().encodeToByteArray()
}
