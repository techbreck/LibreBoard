// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SwipeLexiconIndexTest {
    @Test
    fun filtersByCanonicalLanguageAndBoundedLengthWindow() {
        val index = SwipeLexiconIndex.from(
            listOf(
                LexiconWord("cat", "en-us", 80),
                LexiconWord("catalogue", "en-US", 100),
                LexiconWord("Katze", "de-DE", 90),
            ),
        )

        val words = index.words(listOf("en-US"), approximateLength = 3, maximumWords = 20, blockPossiblyOffensive = false)

        assertEquals(listOf("cat"), words.map(LexiconWord::word))
        assertEquals("en-US", words.single().languageTag)
    }

    @Test
    fun personalDuplicateWinsAndOffensiveFilteringIsLateBound() {
        val index = SwipeLexiconIndex.from(
            listOf(
                LexiconWord("LibreBoard", "en-US", 220),
                LexiconWord("libreboard", "en-US", 12, personal = true),
                LexiconWord("private", "en-US", 255, possiblyOffensive = true),
            ),
        )

        val unfiltered = index.words(listOf("en-US"), 9, 20, blockPossiblyOffensive = false)
        val filtered = index.words(listOf("en-US"), 9, 20, blockPossiblyOffensive = true)

        assertTrue(unfiltered.first { it.word.equals("libreboard", ignoreCase = true) }.personal)
        assertTrue(unfiltered.any { it.word == "private" })
        assertFalse(filtered.any { it.word == "private" })
    }

    @Test
    fun apostrophesAndHyphensDoNotInflateCtcEmissionLength() {
        val index = SwipeLexiconIndex.from(
            listOf(
                LexiconWord("don't", "en-US", 100),
                LexiconWord("co-op", "en-US", 90),
            ),
        )

        assertTrue(index.words(listOf("en-US"), 4, 20, false).any { it.word == "don't" })
        assertTrue(index.words(listOf("en-US"), 4, 20, false).any { it.word == "co-op" })
    }

    @Test
    fun removalDropsEveryCaseVariantWithoutMutatingOriginalIndex() {
        val index = SwipeLexiconIndex.from(
            listOf(
                LexiconWord("LibreBoard", "en-US", 100),
                LexiconWord("LIBREBOARD", "de-DE", 80),
                LexiconWord("keyboard", "en-US", 70),
            ),
        )

        val removed = index.without("libreboard")

        assertTrue(index.words(listOf("en-US"), 10, 20, false).any { it.word == "LibreBoard" })
        assertFalse(removed.words(listOf("en-US", "de-DE"), 10, 20, false).any {
            it.word.equals("libreboard", ignoreCase = true)
        })
        assertEquals(listOf("keyboard"), removed.words(listOf("en-US"), 8, 20, false).map(LexiconWord::word))
    }
}
