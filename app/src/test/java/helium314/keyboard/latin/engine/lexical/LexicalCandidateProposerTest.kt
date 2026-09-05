// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.lexical

import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EditOperation
import helium314.keyboard.latin.engine.WordLock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LexicalCandidateProposerTest {
    @Test
    fun englishContractionIsCanonicalAndLanguageScoped() {
        val proposer = proposer(mapOf("en-US" to setOf("don't"), "de" to setOf("dont")))

        val result = proposer.propose("dont", listOf("en-US", "de"), WordLock.Unlocked, deadline())

        assertEquals(listOf("don't"), result.map { it.surface })
        assertEquals("en-US", result.single().languageTag)
        assertTrue(CandidateSource.CONTRACTION in result.single().sources)
        assertEquals(listOf(EditOperation.APOSTROPHE), result.single().edits)
    }

    @Test
    fun contractionPreservesUserCasing() {
        val proposer = proposer(mapOf("en-US" to setOf("don't")))

        assertEquals("Don't", proposer.propose("Dont", listOf("en-US"), WordLock.Unlocked, deadline()).single().surface)
        assertEquals("DON'T", proposer.propose("DONT", listOf("en-US"), WordLock.Unlocked, deadline()).single().surface)
    }

    @Test
    fun oneMissedSpaceHypothesisRequiresBothWordsInSameLanguage() {
        val proposer = proposer(mapOf("en-US" to setOf("in", "the"), "de" to setOf("in")))

        val result = proposer.propose("inthe", listOf("en-US", "de"), WordLock.Unlocked, deadline())

        assertEquals(listOf("in the"), result.map { it.surface })
        assertEquals(listOf(EditOperation.SPLIT), result.single().edits)
    }

    @Test
    fun crossLanguageFragmentsAreNeverCombined() {
        val proposer = proposer(mapOf("en-US" to setOf("in"), "de" to setOf("the")))

        assertTrue(proposer.propose("inthe", listOf("en-US", "de"), WordLock.Unlocked, deadline()).isEmpty())
    }

    @Test
    fun joinRequiresWholeWordInSameLanguage() {
        val proposer = proposer(mapOf("en-US" to setOf("notebook"), "de" to emptySet()))

        val result = proposer.propose("note book", listOf("en-US", "de"), WordLock.Unlocked, deadline())

        assertEquals(listOf("notebook"), result.map { it.surface })
        assertEquals(listOf(EditOperation.JOIN), result.single().edits)
    }

    @Test
    fun germanConstituentsReinforceUnsplitSurfaceWithoutProposingSplit() {
        val proposer = proposer(mapOf("de" to setOf("daten", "schutz")))

        val result = proposer.propose("datenschutz", listOf("de"), WordLock.Unlocked, deadline())

        assertEquals(listOf("datenschutz"), result.map { it.surface })
        assertEquals(setOf(CandidateSource.COMPOUND), result.single().sources)
        assertTrue(EditOperation.SPLIT !in result.single().edits)
    }

    @Test
    fun manualLanguageLockPreventsOtherLanguageProposal() {
        val proposer = proposer(mapOf("en-US" to setOf("don't"), "de" to setOf("daten", "schutz")))

        val result = proposer.propose("dont", listOf("en-US", "de"), WordLock.Manual("de"), deadline())

        assertTrue(result.isEmpty())
    }

    @Test
    fun expiredDeadlineReturnsWithoutDictionaryReads() {
        var reads = 0
        val proposer = LexicalCandidateProposer { _, _ -> reads += 1; true }

        val result = proposer.propose("inthe", listOf("en-US"), WordLock.Unlocked, Deadline.afterMillis(0))

        assertTrue(result.isEmpty())
        assertEquals(0, reads)
    }

    private fun proposer(wordsByLanguage: Map<String, Set<String>>) = LexicalCandidateProposer { word, language ->
        wordsByLanguage[language].orEmpty().any { it.equals(word, ignoreCase = true) }
    }

    private fun deadline() = Deadline.afterMillis(1_000)
}
