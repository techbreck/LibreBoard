// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.Locale

@RunWith(RobolectricTestRunner::class)
class LegacySuggestionFusionTest {
    private val english = Locale.forLanguageTag("en-US")
    private val german = Locale.GERMAN

    @Test
    fun classicRankingIsPreservedWhileLiteralRawIsOmittedForSuggestToInject() {
        val fusion = LegacySuggestionFusion()
        val result = fusion.fuse(
            rawText = "thsi",
            classicSuggestions = listOf(
                suggestion("this", 900_000, english),
                suggestion("thus", 700_000, english),
            ),
            supplementalCandidates = emptyList(),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )

        assertEquals(listOf("this", "thus"), result.suggestions.map { it.mWord })
    }

    @Test
    fun uniquePersonalCompletionParticipatesInLiveRanking() {
        val fusion = LegacySuggestionFusion()
        val result = fusion.fuse(
            rawText = "libre",
            classicSuggestions = listOf(
                suggestion("libretto", 900_000, english),
                suggestion("liberty", 800_000, english),
            ),
            supplementalCandidates = listOf(personal("LibreBoard", exact = false)),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )

        assertEquals(listOf("libretto", "LibreBoard", "liberty"), result.suggestions.map { it.mWord })
    }

    @Test
    fun exactPersonalWordVetoIsRetainedAlongsideCaseDistinctSurface() {
        val fusion = LegacySuggestionFusion()
        val result = fusion.fuse(
            rawText = "libreboard",
            classicSuggestions = listOf(suggestion("whiteboard", 1_500_000, english)),
            supplementalCandidates = listOf(personal("LibreBoard", exact = true)),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )

        assertTrue(result.exactPersonalMatch)
        assertTrue(result.suggestions.any { it.mWord == "LibreBoard" })
    }

    @Test
    fun thirdCharacterLocksSlateToOneLanguage() {
        val fusion = LegacySuggestionFusion()
        val enabled = listOf("en-US", "de")
        fun fuse(raw: String, englishScore: Int, germanScore: Int) = fusion.fuse(
            rawText = raw,
            classicSuggestions = listOf(
                suggestion("gift", englishScore, english),
                suggestion("Gift", germanScore, german),
            ),
            supplementalCandidates = emptyList(),
            enabledLanguageTags = enabled,
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )

        assertEquals(WordLock.Unlocked, fuse("g", 1_000, 1_000).wordLock)
        assertEquals(WordLock.Unlocked, fuse("gi", 1_000, 1_000).wordLock)
        val locked = fuse("gif", 100, 1_000)
        assertEquals(WordLock.Automatic("de"), locked.wordLock)
        assertEquals(listOf("Gift"), locked.suggestions.map { it.mWord })
    }

    @Test
    fun generatedCorrectionUsesDistinctNonPersonalProvenance() {
        val fusion = LegacySuggestionFusion()
        val contraction = Candidate(
            surface = "don't",
            languageTag = "en-US",
            sources = setOf(CandidateSource.CONTRACTION),
            components = ScoreComponents(staticFrequency = 1.0),
        )

        val result = fusion.fuse(
            rawText = "dont",
            classicSuggestions = emptyList(),
            supplementalCandidates = listOf(contraction),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )

        assertEquals(listOf("don't"), result.suggestions.map { it.mWord })
        assertEquals(Dictionary.DICTIONARY_ENGINE_GENERATED, result.suggestions.single().mSourceDict)
        assertTrue(!result.suggestions.single().isAppropriateForAutoCorrection)
    }

    @Test
    fun germanCompoundEvidenceVetoesReplacementWithoutCreatingSplitSuggestion() {
        val fusion = LegacySuggestionFusion()
        val compound = Candidate(
            surface = "datenschutz",
            languageTag = "de",
            sources = setOf(CandidateSource.COMPOUND),
            components = ScoreComponents(staticFrequency = 1.0),
        )

        val result = fusion.fuse(
            rawText = "datenschutz",
            classicSuggestions = listOf(suggestion("Datenschatz", 1_000_000, german)),
            supplementalCandidates = listOf(compound),
            enabledLanguageTags = listOf("de"),
            defaultLocale = german,
            inputStyle = InputStyle.TAP,
        )

        assertTrue(result.rawReplacementVeto)
        assertTrue(result.suggestions.none { ' ' in it.mWord })
    }

    private fun personal(surface: String, exact: Boolean) = Candidate(
        surface = surface,
        languageTag = "en-US",
        sources = setOf(CandidateSource.PERSONAL),
        components = ScoreComponents(personal = 2.0),
        exactPersonalMatch = exact,
    )

    private fun suggestion(word: String, score: Int, locale: Locale) = SuggestedWordInfo(
        word,
        "",
        score,
        SuggestedWordInfo.KIND_CORRECTION or SuggestedWordInfo.KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION,
        TestDictionary(locale),
        SuggestedWordInfo.NOT_AN_INDEX,
        SuggestedWordInfo.NOT_A_CONFIDENCE,
    )

    private class TestDictionary(locale: Locale) : Dictionary(TYPE_MAIN, locale) {
        override fun getSuggestions(
            composedData: ComposedData?,
            ngramContext: NgramContext?,
            proximityInfoHandle: Long,
            settingsValuesForSuggestion: SettingsValuesForSuggestion?,
            sessionId: Int,
            weightForLocale: Float,
            inOutWeightOfLangModelVsSpatialModel: FloatArray?,
        ): ArrayList<SuggestedWordInfo> = arrayListOf()

        override fun isInDictionary(word: String?): Boolean = false
    }
}
