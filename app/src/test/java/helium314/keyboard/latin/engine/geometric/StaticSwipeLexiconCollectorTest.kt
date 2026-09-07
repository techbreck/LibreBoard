// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import org.junit.Test
import kotlin.test.assertEquals

class StaticSwipeLexiconCollectorTest {
    @Test fun frequentWordsAfterTraversalCutoffReplaceEarlyRareWords() {
        val collector = StaticSwipeLexiconCollector(2)
        listOf("aardvark" to 1, "abacus" to 2, "the" to 222, "with" to 190).forEach { (word, frequency) ->
            collector.add(LexiconWord(word, "en-US", frequency))
        }
        assertEquals(listOf("the", "with"), collector.words().map { it.word })
    }

    @Test fun inputOrderCannotChangeBoundedVocabularyOrTies() {
        val words = listOf("the" to 222, "a" to 100, "b" to 100, "c" to 100, "with" to 190)
            .map { (word, frequency) -> LexiconWord(word, "en-US", frequency) }
        fun collect(values: List<LexiconWord>) = StaticSwipeLexiconCollector(3).apply {
            values.forEach(::add)
        }.words()
        assertEquals(collect(words), collect(words.reversed()))
        assertEquals(listOf("the", "with", "a"), collect(words).map { it.word })
    }

    @Test fun normalizationDuplicatesKeepHighestFrequencyWithoutConsumingSlots() {
        val collector = StaticSwipeLexiconCollector(2)
        collector.add(LexiconWord("Word", "en-us", 10))
        collector.add(LexiconWord("word", "en-US", 20))
        collector.add(LexiconWord("other", "en-US", 15))
        assertEquals(listOf("word", "other"), collector.words().map { it.word })
        assertEquals(listOf(20, 15), collector.words().map { it.frequency })
    }

    @Test fun invalidEntriesCannotDisplaceUsableWords() {
        val collector = StaticSwipeLexiconCollector(1)
        collector.add(LexiconWord("the", "en-US", 100))
        for (word in listOf("", "---", "not a word", "a".repeat(65))) {
            collector.add(LexiconWord(word, "en-US", 255))
        }
        assertEquals("the", collector.words().single().word)
    }
}
