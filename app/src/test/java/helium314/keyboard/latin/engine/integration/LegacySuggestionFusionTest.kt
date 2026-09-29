// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.AutoCorrectionAggressiveness
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.NeuralRescorer
import helium314.keyboard.latin.engine.NeuralScoreResult
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.key
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
    fun manualLanguageSelectionSurvivesWordBoundariesUntilExplicitlyReleased() {
        val fusion = LegacySuggestionFusion()
        val enabled = listOf("en-US", "de")
        fun fuse(raw: String) = fusion.fuse(
            rawText = raw,
            classicSuggestions = listOf(
                suggestion("gift", 2_000, english),
                suggestion("Gift", 100, german),
            ),
            supplementalCandidates = emptyList(),
            enabledLanguageTags = enabled,
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )

        fusion.selectLanguageManually("de")
        assertEquals(WordLock.Manual("de"), fuse("g").wordLock)
        assertEquals(listOf("Gift"), fuse("g").suggestions.map { it.mWord })
        fusion.resetWord()
        assertEquals(WordLock.Manual("de"), fuse("g").wordLock)

        fusion.releaseManualLanguageSelection()
        assertEquals(WordLock.Automatic("en-US"), fuse("g").wordLock)
        assertEquals(listOf("gift"), fuse("g").suggestions.map { it.mWord })
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

    @Test
    fun availableNeuralScoresCanReorderTheLiveSlateAndSelectCalibratedCorrection() {
        var captured: TypingRequest? = null
        val fusion = LegacySuggestionFusion(neuralRescorer = NeuralRescorer { request, candidates, _ ->
            captured = request
            NeuralScoreResult(
                EngineAvailability.AVAILABLE,
                candidates.associate { candidate ->
                    candidate.key to if (candidate.normalized == "this") 10.0 else -10.0
                },
            )
        })
        val classics = buildList {
            add(suggestion("this", 100_000, english))
            listOf("thus", "thin", "then", "than", "tish", "wish", "fish", "dish", "his").forEach {
                add(suggestion(it, 100_000, english))
            }
        }

        val result = fusion.fuse(
            rawText = "thsi",
            classicSuggestions = classics,
            supplementalCandidates = emptyList(),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
            typingRequest = request("thsi", listOf("en-US")),
            neuralStrength = 100,
            aggressiveness = AutoCorrectionAggressiveness.AGGRESSIVE,
            neuralDeadline = Deadline.afterMillis(100),
        )

        assertEquals(EngineAvailability.AVAILABLE, result.neuralAvailability)
        assertEquals("this", result.suggestions.first().mWord)
        assertEquals("this", result.engineAutoCorrectionNormalized)
        assertEquals("before", captured?.precedingContext)
        assertEquals(result.wordLock, captured?.wordLock)
    }

    @Test
    fun neuralTimeoutPreservesClassicFallbackRanking() {
        val classics = listOf(
            suggestion("this", 900_000, english),
            suggestion("thus", 700_000, english),
        )
        val baseline = LegacySuggestionFusion().fuse(
            rawText = "thsi",
            classicSuggestions = classics,
            supplementalCandidates = emptyList(),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
        )
        val timeout = LegacySuggestionFusion(neuralRescorer = NeuralRescorer { _, _, _ ->
            NeuralScoreResult(EngineAvailability.TIMEOUT)
        }).fuse(
            rawText = "thsi",
            classicSuggestions = classics,
            supplementalCandidates = emptyList(),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
            typingRequest = request("thsi", listOf("en-US")),
            neuralStrength = 100,
            neuralDeadline = Deadline.afterMillis(100),
        )

        assertEquals(EngineAvailability.TIMEOUT, timeout.neuralAvailability)
        assertEquals(baseline.suggestions.map { it.mWord }, timeout.suggestions.map { it.mWord })
        assertEquals(null, timeout.engineAutoCorrectionNormalized)
    }

    @Test
    fun lockedLanguageIsCopiedIntoTheImmutableNeuralRequest() {
        var captured: TypingRequest? = null
        val fusion = LegacySuggestionFusion(neuralRescorer = NeuralRescorer { request, _, _ ->
            captured = request
            NeuralScoreResult(EngineAvailability.AVAILABLE)
        })
        val result = fusion.fuse(
            rawText = "gif",
            classicSuggestions = listOf(
                suggestion("gift", 100, english),
                suggestion("Gift", 1_000, german),
            ),
            supplementalCandidates = emptyList(),
            enabledLanguageTags = listOf("en-US", "de"),
            defaultLocale = english,
            inputStyle = InputStyle.TAP,
            typingRequest = request("gif", listOf("en-US", "de")),
            neuralStrength = 50,
            neuralDeadline = Deadline.afterMillis(100),
        )

        assertEquals(WordLock.Automatic("de"), result.wordLock)
        assertEquals(result.wordLock, captured?.wordLock)
        assertTrue(result.suggestions.all { it.mSourceDict.mLocale == german })
    }

    @Test
    fun swipeFusionPublishesCtcRankingAheadOfNativeMatcherScores() {
        val fusion = LegacySuggestionFusion()
        val result = fusion.fuse(
            rawText = "",
            classicSuggestions = listOf(
                suggestion("system", 2_000_000, english),
                suggestion("design", 1_500_000, english),
            ),
            supplementalCandidates = listOf(
                Candidate(
                    "division",
                    languageTag = "en-US",
                    sources = setOf(CandidateSource.CTC_SWIPE),
                    components = ScoreComponents(spatial = -0.4, staticFrequency = 1.0),
                ),
                Candidate(
                    "divisor",
                    languageTag = "en-US",
                    sources = setOf(CandidateSource.CTC_SWIPE),
                    components = ScoreComponents(spatial = -0.8, staticFrequency = 0.4),
                ),
            ),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.SWIPE,
            typingRequest = TypingRequest.bounded(
                rawText = "",
                geometry = KeyGeometry(1f, 1f, emptyList()),
                enabledLanguages = listOf("en-US"),
                fieldPolicy = FieldPolicy.NORMAL,
                inputStyle = InputStyle.SWIPE,
                sequenceId = 19,
            ),
        )

        assertEquals(listOf("division", "divisor"), result.suggestions.map { it.mWord })
    }

    @Test
    fun swipeFusionFallsBackToGeometricWhenCtcIsAbsent() {
        val fusion = LegacySuggestionFusion()
        val result = fusion.fuse(
            rawText = "",
            classicSuggestions = emptyList(),
            supplementalCandidates = listOf(
                Candidate(
                    "highway",
                    languageTag = "en-US",
                    sources = setOf(CandidateSource.GEOMETRIC_SWIPE),
                    components = ScoreComponents(spatial = 1.0, staticFrequency = 1.0),
                ),
            ),
            enabledLanguageTags = listOf("en-US"),
            defaultLocale = english,
            inputStyle = InputStyle.SWIPE,
            typingRequest = TypingRequest.bounded(
                rawText = "",
                geometry = KeyGeometry(1f, 1f, emptyList()),
                enabledLanguages = listOf("en-US"),
                fieldPolicy = FieldPolicy.NORMAL,
                inputStyle = InputStyle.SWIPE,
                sequenceId = 20,
            ),
        )

        assertEquals(listOf("highway"), result.suggestions.map { it.mWord })
    }

    @Test
    fun completeSwipeLocksToStrongestLanguageEvenWhenCtcScoresAreNegative() {
        val fusion = LegacySuggestionFusion()
        val result = fusion.fuse(
            rawText = "",
            classicSuggestions = emptyList(),
            supplementalCandidates = listOf(
                Candidate(
                    "gift",
                    languageTag = "en-US",
                    sources = setOf(CandidateSource.CTC_SWIPE),
                    components = ScoreComponents(spatial = -5.0, staticFrequency = 1.0),
                ),
                Candidate(
                    "Gift",
                    languageTag = "de",
                    sources = setOf(CandidateSource.CTC_SWIPE),
                    components = ScoreComponents(spatial = -0.1, staticFrequency = 1.0),
                ),
            ),
            enabledLanguageTags = listOf("en-US", "de"),
            defaultLocale = english,
            inputStyle = InputStyle.SWIPE,
            typingRequest = TypingRequest.bounded(
                rawText = "",
                geometry = KeyGeometry(1f, 1f, emptyList()),
                enabledLanguages = listOf("en-US", "de"),
                fieldPolicy = FieldPolicy.NORMAL,
                inputStyle = InputStyle.SWIPE,
                sequenceId = 18,
            ),
        )

        assertEquals(WordLock.Automatic("de"), result.wordLock)
        assertEquals(listOf("Gift"), result.suggestions.map { it.mWord })
    }

    private fun personal(surface: String, exact: Boolean) = Candidate(
        surface = surface,
        languageTag = "en-US",
        sources = setOf(CandidateSource.PERSONAL),
        components = ScoreComponents(personal = 2.0),
        exactPersonalMatch = exact,
    )

    private fun request(rawText: String, languages: List<String>) = TypingRequest.bounded(
        rawText = rawText,
        precedingContext = "before",
        geometry = KeyGeometry(1f, 1f, emptyList()),
        enabledLanguages = languages,
        fieldPolicy = FieldPolicy.NORMAL,
        inputStyle = InputStyle.TAP,
        sequenceId = 17,
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
