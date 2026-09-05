// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class FusedCandidateScorerTest {
    private val scorer = FusedCandidateScorer()

    @Test
    fun rawCandidateIsAlwaysFirstWhileCorrectionCanWin() {
        val raw = candidate("thsi", spatial = -3.0, context = -4.0)
        val correction = candidate("this", spatial = 3.0, context = 4.0, edits = listOf(EditOperation.TRANSPOSE))
        val result = scorer.rank("thsi", listOf(correction, raw), WordLock.Automatic("en-US"), 50,
            AutoCorrectionAggressiveness.BALANCED)

        assertEquals("thsi", result.candidates.first().surface)
        assertEquals("this", result.autoCorrection?.surface)
        assertTrue(result.autoCorrection!!.calibratedProbability >= 0.88)
    }

    @Test
    fun exactPersonalWordCannotBeCorrectedAway() {
        val name = candidate("Breck", spatial = -2.0, context = -2.0).copy(exactPersonalMatch = true)
        val replacement = candidate("break", spatial = 5.0, context = 5.0)
        val result = scorer.rank("Breck", listOf(name, replacement), WordLock.Automatic("en-US"), 100,
            AutoCorrectionAggressiveness.AGGRESSIVE)
        assertNull(result.autoCorrection)
    }

    @Test
    fun languageLockDropsCrossLanguageCandidates() {
        val english = candidate("gift", language = "en-US", spatial = 1.0, context = 1.0)
        val german = candidate("Gift", language = "de", spatial = 5.0, context = 5.0)
        val result = scorer.rank("gift", listOf(english, german), WordLock.Automatic("en-US"), 100,
            AutoCorrectionAggressiveness.AGGRESSIVE)
        assertTrue(result.candidates.all { it.languageTag == "en-US" })
    }

    @Test
    fun neuralSliderHasLockedCalibrationPoints() {
        assertEquals(0.0, scorer.neuralCoefficient(0), 0.0001)
        assertEquals(0.7, scorer.neuralCoefficient(50), 0.0001)
        assertEquals(1.2, scorer.neuralCoefficient(100), 0.0001)
    }

    @Test
    fun rejectionPenaltyVetoesAutomaticReplacement() {
        val raw = candidate("teh", spatial = -4.0, context = -4.0)
        val rejected = candidate("the", spatial = 4.0, context = 4.0).copy(rejectionPenalty = 2.0)
        val result = scorer.rank("teh", listOf(raw, rejected), WordLock.Automatic("en-US"), 100,
            AutoCorrectionAggressiveness.AGGRESSIVE)
        assertNull(result.autoCorrection)
    }

    @Test
    fun singlePersonalSignalIsNotDiscardedByNormalization() {
        val personal = Candidate(
            surface = "LibreBoard",
            languageTag = "en-US",
            sources = setOf(CandidateSource.PERSONAL),
            components = ScoreComponents(personal = 2.0),
        )
        val result = scorer.rank("libre", listOf(personal), WordLock.Automatic("en-US"), 0,
            AutoCorrectionAggressiveness.BALANCED)

        assertTrue(result.candidates.single { it.surface == "LibreBoard" }.totalScore > 0.0)
    }

    @Test
    fun mergedCandidatePreservesPersonalCasing() {
        val static = candidate("libreboard", spatial = 1.0, context = 0.0)
        val personal = static.copy(
            surface = "LibreBoard",
            sources = setOf(CandidateSource.PERSONAL),
            exactPersonalMatch = true,
            components = ScoreComponents(personal = 2.0),
        )
        val result = scorer.rank("libre", listOf(static, personal), WordLock.Automatic("en-US"), 0,
            AutoCorrectionAggressiveness.BALANCED)

        assertEquals("LibreBoard", result.candidates.single { it.normalized == "libreboard" }.surface)
    }

    private fun candidate(
        word: String,
        language: String = "en-US",
        spatial: Double,
        context: Double,
        edits: List<EditOperation> = emptyList(),
    ) = Candidate(
        surface = word,
        languageTag = language,
        sources = setOf(CandidateSource.STATIC_DICTIONARY, CandidateSource.SPATIAL),
        components = ScoreComponents(spatial = spatial, staticFrequency = spatial, context = context, language = 1.0),
        edits = edits,
    )
}
