// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.Locale

/** Immutable length/language index built off the UI thread from data-only dictionary entries. */
class SwipeLexiconIndex private constructor(
    private val buckets: Map<String, Map<Int, List<LexiconWord>>>,
) {
    val size: Int = buckets.values.sumOf { byLength -> byLength.values.sumOf { it.size } }

    fun words(
        languageTags: List<String>,
        approximateLength: Int,
        maximumWords: Int,
        blockPossiblyOffensive: Boolean,
    ): List<LexiconWord> {
        if (maximumWords <= 0) return emptyList()
        val lengths = if (approximateLength <= 0) {
            1..MAX_EMISSION_LENGTH
        } else {
            (approximateLength - LENGTH_TOLERANCE_BELOW).coerceAtLeast(1)..
                (approximateLength + LENGTH_TOLERANCE_ABOVE).coerceAtMost(MAX_EMISSION_LENGTH)
        }
        return languageTags.asSequence()
            .map(::canonicalLanguageTag)
            .distinct()
            .flatMap { language ->
                val byLength = buckets[language].orEmpty()
                lengths.asSequence().flatMap { length -> byLength[length].orEmpty().asSequence() }
            }
            .filterNot { blockPossiblyOffensive && it.possiblyOffensive }
            .sortedWith(compareByDescending<LexiconWord> { it.personal }.thenByDescending { it.frequency })
            // from() already deduplicates normalized word/language keys into disjoint length
            // buckets, and requested languages are distinct above. Re-normalizing every word
            // here repeats expensive Unicode/Locale work on each gesture without changing output.
            .take(maximumWords)
            .toList()
    }

    fun without(word: String): SwipeLexiconIndex {
        val normalized = normalizeCandidate(word)
        if (normalized.isBlank()) return this
        return from(buckets.values.asSequence().flatMap { byLength ->
            byLength.values.asSequence().flatMap { it.asSequence() }
        }.filterNot { normalizeCandidate(it.word) == normalized }.toList())
    }

    companion object {
        val EMPTY = SwipeLexiconIndex(emptyMap())

        fun from(words: Collection<LexiconWord>): SwipeLexiconIndex {
            if (words.isEmpty()) return EMPTY
            val deduplicated = linkedMapOf<Pair<String, String>, LexiconWord>()
            words.forEach { word ->
                val language = canonicalLanguageTag(word.languageTag)
                val normalized = normalizeCandidate(word.word)
                val length = emissionLength(normalized)
                if (language.isBlank() || normalized.isBlank() || length !in 1..MAX_EMISSION_LENGTH) return@forEach
                val normalizedWord = word.copy(languageTag = language)
                val key = normalized to language
                val previous = deduplicated[key]
                if (previous == null || normalizedWord.personal && !previous.personal ||
                    normalizedWord.personal == previous.personal && normalizedWord.frequency > previous.frequency
                ) {
                    deduplicated[key] = normalizedWord
                }
            }
            val buckets = deduplicated.values
                .groupBy { canonicalLanguageTag(it.languageTag) }
                .mapValues { (_, languageWords) ->
                    languageWords.groupBy { emissionLength(normalizeCandidate(it.word)) }
                        .mapValues { (_, sameLength) ->
                            sameLength.sortedWith(
                                compareByDescending<LexiconWord> { it.personal }
                                    .thenByDescending { it.frequency }
                                    .thenBy { it.word },
                            )
                        }
                }
            return SwipeLexiconIndex(buckets)
        }

        private fun emissionLength(word: String): Int {
            var count = 0
            var index = 0
            while (index < word.length) {
                val codePoint = word.codePointAt(index)
                if (codePoint != '\''.code && codePoint != 0x2019 && codePoint != '-'.code) count++
                index += Character.charCount(codePoint)
            }
            return count
        }

        private fun canonicalLanguageTag(tag: String): String =
            Locale.forLanguageTag(tag).takeUnless { it == Locale.ROOT }?.toLanguageTag().orEmpty()

        // The live-geometry path estimate is deliberately approximate. This bounded window covers
        // 99%+ of the pinned validation corpus while still excluding most unrelated word lengths.
        private const val LENGTH_TOLERANCE_BELOW = 3
        private const val LENGTH_TOLERANCE_ABOVE = 4
        private const val MAX_EMISSION_LENGTH = 64
    }
}
