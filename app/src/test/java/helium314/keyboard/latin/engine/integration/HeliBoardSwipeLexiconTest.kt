// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.latin.DictionaryFacilitator
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.geometric.LexiconWord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.mockito.Mockito
import java.util.concurrent.atomic.AtomicInteger

class HeliBoardSwipeLexiconTest {
    @Test
    fun cachesStaticSnapshotByRevisionAndKeepsPersonalSurfaceFirst() {
        val facilitator = Mockito.mock(DictionaryFacilitator::class.java)
        Mockito.`when`(facilitator.swipeLexiconRevision).thenReturn(7L, 7L, 8L)
        Mockito.`when`(facilitator.getSwipeLexiconWords(listOf("en-US"), 10, 100_000, true))
            .thenReturn(listOf(LexiconWord("libreboard", "en-US", 200)))
        val personalCalls = AtomicInteger()
        val personalRevision = AtomicInteger()
        val lexicon = HeliBoardSwipeLexicon(
            facilitator,
            policyProvider = { SwipeLexiconRuntimePolicy(true, true, FieldPolicy.NORMAL, false) },
            personalProvider = { _, _, _, _ ->
                personalCalls.incrementAndGet()
                listOf(LexiconWord("LibreBoard", "en-us", 3, personal = true))
            },
            personalRevisionProvider = { personalRevision.get().toLong() },
        )

        repeat(2) {
            val words = lexicon.words(listOf("en-us"), 10).toList()
            assertEquals("LibreBoard", words.single().word)
            assertEquals("en-US", words.single().languageTag)
            assertTrue(words.single().personal)
        }
        personalRevision.incrementAndGet()
        val refreshed = lexicon.words(listOf("en-us"), 10).toList()
        assertEquals("LibreBoard", refreshed.single().word)

        Mockito.verify(facilitator, Mockito.times(2))
            .getSwipeLexiconWords(listOf("en-US"), 10, 100_000, true)
        assertEquals(2, personalCalls.get())
    }

    @Test
    fun disabledPersonalizationDoesNotOpenPersonalProvider() {
        val facilitator = Mockito.mock(DictionaryFacilitator::class.java)
        Mockito.`when`(facilitator.getSwipeLexiconWords(listOf("de-DE"), 5, 100_000, false))
            .thenReturn(listOf(LexiconWord("Katze", "de-DE", 100)))
        var personalCalled = false
        val lexicon = HeliBoardSwipeLexicon(
            facilitator,
            policyProvider = { SwipeLexiconRuntimePolicy(false, false, FieldPolicy.SENSITIVE, true) },
            personalProvider = { _, _, _, _ -> personalCalled = true; emptyList() },
        )

        assertEquals(listOf("Katze"), lexicon.words(listOf("de-DE"), 5).map(LexiconWord::word).toList())
        assertFalse(personalCalled)
    }
}
