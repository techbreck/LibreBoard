// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Test

class LanguageLockTest {
    @Test
    fun locksOnStrongEvidenceDuringFirstTwoCharacters() {
        val controller = LanguageLockController(lockMargin = 0.25)
        controller.beginWord()
        assertEquals(
            WordLock.Automatic("de"),
            controller.observe(LanguageEvidence(mapOf("de" to 0.8, "en-US" to 0.2), 2)),
        )
    }

    @Test
    fun locksNoLaterThanThirdCharacter() {
        val controller = LanguageLockController(lockMargin = 0.25)
        assertEquals(
            WordLock.Unlocked,
            controller.observe(LanguageEvidence(mapOf("de" to 0.55, "en-US" to 0.45), 2)),
        )
        assertEquals(
            WordLock.Automatic("de"),
            controller.observe(LanguageEvidence(mapOf("de" to 0.55, "en-US" to 0.45), 3)),
        )
    }

    @Test
    fun manualSelectionSurvivesWordBoundariesUntilReleased() {
        val controller = LanguageLockController()
        controller.selectManually("de")
        controller.endWord()
        assertEquals(WordLock.Manual("de"), controller.beginWord())
        controller.releaseManualSelection()
        assertEquals(WordLock.Unlocked, controller.beginWord())
    }
}
