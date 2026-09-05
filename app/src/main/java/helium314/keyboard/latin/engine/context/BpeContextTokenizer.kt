// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.context

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.text.Normalizer
import java.util.Locale

@Serializable
private data class BpeTokenizerDocument(
    val schemaVersion: Int,
    val normalization: String,
    val vocabulary: Map<String, Int>,
    val merges: List<List<String>>,
    val specialTokens: BpeSpecialTokens,
)

@Serializable
private data class BpeSpecialTokens(
    val padding: String,
    val beginningOfSequence: String,
    val unknown: String,
    val languages: Map<String, String>,
)

/** Deterministic, dependency-free BPE tokenizer for the context-en-de-v1 sidecar. */
class BpeContextTokenizer private constructor(
    private val vocabulary: Map<String, Int>,
    private val mergeRanks: Map<Pair<String, String>, Int>,
    override val paddingTokenId: Int,
    override val beginningOfSequenceTokenId: Int,
    private val unknownTokenId: Int,
    private val languageTokenIds: Map<String, Int>,
) : ContextTokenizer {
    override fun languageTokenId(languageTag: String): Int? =
        languageTokenIds[languageTag] ?: languageTokenIds[languageTag.substringBefore('-')]

    override fun encode(text: String, maximumTokens: Int, truncation: TokenTruncation): IntArray {
        require(maximumTokens >= 0)
        if (maximumTokens == 0 || text.isBlank()) return IntArray(0)
        require(text.codePointCount(0, text.length) <= MAX_INPUT_CODE_POINTS) {
            "tokenizer input is too long"
        }
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(Locale.ROOT)
        val tokenIds = ArrayList<Int>()
        normalized.split(WHITESPACE).filter(String::isNotEmpty).forEach { word ->
            val symbols = ArrayList<String>()
            symbols += WORD_START
            var index = 0
            while (index < word.length) {
                val codePoint = word.codePointAt(index)
                symbols += String(Character.toChars(codePoint))
                index += Character.charCount(codePoint)
            }
            applyMerges(symbols)
            symbols.forEach { symbol -> tokenIds += vocabulary[symbol] ?: unknownTokenId }
        }
        val selected = when (truncation) {
            TokenTruncation.KEEP_START -> tokenIds.take(maximumTokens)
            TokenTruncation.KEEP_END -> tokenIds.takeLast(maximumTokens)
        }
        return selected.toIntArray()
    }

    private fun applyMerges(symbols: MutableList<String>) {
        while (symbols.size > 1) {
            var selectedIndex = -1
            var selectedRank = Int.MAX_VALUE
            for (index in 0 until symbols.lastIndex) {
                val rank = mergeRanks[symbols[index] to symbols[index + 1]] ?: continue
                if (rank < selectedRank) {
                    selectedRank = rank
                    selectedIndex = index
                }
            }
            if (selectedIndex < 0) return
            symbols[selectedIndex] = symbols[selectedIndex] + symbols[selectedIndex + 1]
            symbols.removeAt(selectedIndex + 1)
        }
    }

    companion object {
        const val MAX_TOKENIZER_BYTES = 2 * 1024 * 1024
        const val MAX_VOCABULARY_SIZE = 16_384
        const val MAX_MERGES = 65_536
        const val MAX_INPUT_CODE_POINTS = 256
        private const val NORMALIZATION = "NFKC_LOWER"
        private const val WORD_START = "▁"
        private val WHITESPACE = Regex("\\s+")
        private val json = Json { ignoreUnknownKeys = false }

        fun fromJson(bytes: ByteArray): BpeContextTokenizer {
            require(bytes.size in 1..MAX_TOKENIZER_BYTES) { "tokenizer file size is invalid" }
            val document = json.decodeFromString<BpeTokenizerDocument>(
                bytes.decodeToString(throwOnInvalidSequence = true),
            )
            require(document.schemaVersion == 1) { "unsupported tokenizer schema" }
            require(document.normalization == NORMALIZATION) { "unsupported tokenizer normalization" }
            require(document.vocabulary.size in 4..MAX_VOCABULARY_SIZE) { "invalid tokenizer vocabulary size" }
            require(document.vocabulary.keys.all { it.isNotEmpty() && it.length <= MAX_TOKEN_LENGTH }) {
                "invalid tokenizer token"
            }
            val ids = document.vocabulary.values
            require(ids.distinct().size == ids.size && ids.sorted() == ids.indices.toList()) {
                "tokenizer IDs must be unique and dense"
            }
            require(WORD_START in document.vocabulary) { "tokenizer is missing the word-start token" }
            require(document.merges.size <= MAX_MERGES) { "tokenizer has too many merges" }
            val mergeRanks = LinkedHashMap<Pair<String, String>, Int>()
            document.merges.forEachIndexed { rank, merge ->
                require(merge.size == 2 && merge.all(String::isNotEmpty)) { "invalid BPE merge" }
                val pair = merge[0] to merge[1]
                require(mergeRanks.put(pair, rank) == null) { "duplicate BPE merge" }
                require(merge[0] + merge[1] in document.vocabulary) {
                    "BPE merge output is absent from the vocabulary"
                }
            }
            fun specialId(token: String, label: String): Int = requireNotNull(document.vocabulary[token]) {
                "tokenizer is missing $label"
            }
            require(document.specialTokens.languages.isNotEmpty()) { "tokenizer has no language tokens" }
            require(document.specialTokens.languages.keys.all {
                it.isNotBlank() && it.length <= MAX_LANGUAGE_TAG_LENGTH
            }) { "invalid tokenizer language tag" }
            val languageIds = document.specialTokens.languages.mapValues { (_, token) ->
                specialId(token, "language token")
            }
            val reservedIds = listOf(
                specialId(document.specialTokens.padding, "padding token"),
                specialId(document.specialTokens.beginningOfSequence, "beginning-of-sequence token"),
                specialId(document.specialTokens.unknown, "unknown token"),
            ) + languageIds.values
            require(reservedIds.distinct().size == reservedIds.size) {
                "tokenizer special tokens must use distinct IDs"
            }
            return BpeContextTokenizer(
                vocabulary = document.vocabulary.toMap(),
                mergeRanks = mergeRanks,
                paddingTokenId = reservedIds[0],
                beginningOfSequenceTokenId = reservedIds[1],
                unknownTokenId = reservedIds[2],
                languageTokenIds = languageIds,
            )
        }

        private const val MAX_TOKEN_LENGTH = 128
        private const val MAX_LANGUAGE_TAG_LENGTH = 64
    }
}
