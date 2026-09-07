// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.Locale
import java.util.TreeSet

/** Keeps a bounded frequency-ranked vocabulary independent of native trie traversal order. */
internal class StaticSwipeLexiconCollector(private val maximumWords: Int) {
    init { require(maximumWords > 0) }

    private val order = compareBy<LexiconWord> { it.frequency }
        .thenByDescending { normalizeCandidate(it.word) }
        .thenByDescending { it.languageTag }
        .thenByDescending { it.word }
    private val ranked = TreeSet(order)
    private val byIdentity = HashMap<Pair<String, String>, LexiconWord>()

    fun add(word: LexiconWord) {
        val normalized = normalizeCandidate(word.word)
        val language = Locale.forLanguageTag(word.languageTag).toLanguageTag()
        val length = normalized.count { it != '\'' && it != '\u2019' && it != '-' }
        if (word.frequency < 0 || language == "und" || length !in 1..64 ||
            !word.word.all { it.isLetter() || it == '\'' || it == '\u2019' || it == '-' }) return
        val candidate = word.copy(languageTag = language)
        val identity = normalized to language
        val previous = byIdentity[identity]
        if (previous != null) {
            if (order.compare(candidate, previous) <= 0) return
            ranked.remove(previous)
        } else if (ranked.size == maximumWords) {
            if (order.compare(candidate, ranked.first()) <= 0) return
            val removed = ranked.pollFirst()!!
            byIdentity.remove(normalizeCandidate(removed.word) to removed.languageTag)
        }
        ranked.add(candidate)
        byIdentity[identity] = candidate
    }

    fun words(): List<LexiconWord> = ranked.descendingSet().toList()
}
