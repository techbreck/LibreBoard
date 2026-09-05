// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.engine.AutoCorrectionAggressiveness
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.FusedCandidateScorer
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.LanguageEvidence
import helium314.keyboard.latin.engine.LanguageLockController
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln1p
import kotlin.math.max

internal data class LegacyFusionResult(
    val suggestions: ArrayList<SuggestedWordInfo>,
    val exactPersonalMatch: Boolean,
    val rawReplacementVeto: Boolean,
    val wordLock: WordLock,
)

/**
 * Translates retained AOSP dictionary results into LibreBoard candidates, ranks the union with
 * supplemental engine candidates, and translates the bounded slate back to the existing IME UI
 * contract.
 *
 * The legacy dictionary score already combines spatial and static evidence, so it is represented
 * once rather than double-counted. Neural autocorrection is deliberately not enabled here until a
 * measured model and calibrated live probability mapping are available.
 */
internal class LegacySuggestionFusion(
    private val scorer: FusedCandidateScorer = FusedCandidateScorer(),
    private val languageLock: LanguageLockController = LanguageLockController(),
) {
    private var wordActive = false
    private var previousRaw = ""

    fun resetWord() {
        wordActive = false
        previousRaw = ""
        languageLock.endWord()
    }

    fun fuse(
        rawText: String,
        classicSuggestions: List<SuggestedWordInfo>,
        supplementalCandidates: List<Candidate>,
        enabledLanguageTags: List<String>,
        defaultLocale: Locale,
        inputStyle: InputStyle,
    ): LegacyFusionResult {
        val defaultLanguage = defaultLocale.toLanguageTag()
        val languageProbabilities = languageProbabilities(
            classicSuggestions,
            supplementalCandidates,
            enabledLanguageTags.ifEmpty { listOf(defaultLanguage) },
            defaultLanguage,
        )
        val currentLock = updateWordLock(rawText, languageProbabilities)
        val classicCandidates = classicSuggestions.take(SuggestedWords.MAX_SUGGESTIONS).map { info ->
            val language = languageTag(info, defaultLanguage)
            val userSpecific = info.mSourceDict.isUserSpecific
            val sources = buildSet {
                add(if (userSpecific) CandidateSource.PERSONAL else CandidateSource.STATIC_DICTIONARY)
                when (inputStyle) {
                    InputStyle.TAP -> add(CandidateSource.SPATIAL)
                    InputStyle.SWIPE -> add(CandidateSource.GEOMETRIC_SWIPE)
                    InputStyle.PREDICTION -> add(CandidateSource.NEXT_WORD)
                }
            }
            Candidate(
                surface = info.mWord,
                languageTag = language,
                sources = sources,
                components = ScoreComponents(
                    spatial = info.mScore.toDouble().takeIf { inputStyle != InputStyle.PREDICTION && !userSpecific },
                    staticFrequency = info.mScore.toDouble().takeIf { inputStyle == InputStyle.PREDICTION && !userSpecific },
                    personal = info.mScore.toDouble().takeIf { userSpecific },
                    language = languageProbabilities[language],
                ),
                exactPersonalMatch = userSpecific && normalizeCandidate(info.mWord) == normalizeCandidate(rawText),
            )
        }
        val supplementalWithLanguage = supplementalCandidates.map { candidate ->
            candidate.copy(
                components = candidate.components.copy(
                    language = candidate.components.language ?: languageProbabilities[candidate.languageTag],
                ),
            )
        }
        val ranked = scorer.rank(
            raw = rawText,
            candidates = classicCandidates + supplementalWithLanguage,
            wordLock = currentLock,
            neuralStrength = 0,
            aggressiveness = AutoCorrectionAggressiveness.BALANCED,
        )
        val rawNormalized = normalizeCandidate(rawText)
        val exactPersonalMatch = ranked.candidates.any {
            it.normalized == rawNormalized && it.exactPersonalMatch
        }
        val rawReplacementVeto = ranked.candidates.any {
            it.normalized == rawNormalized &&
                (it.exactPersonalMatch || CandidateSource.COMPOUND in it.sources)
        }

        val classicByKey = classicSuggestions.groupBy { info ->
            normalizeCandidate(info.mWord) to languageTag(info, defaultLanguage)
        }
        val supplementalByKey = supplementalWithLanguage.groupBy { it.normalized to it.languageTag }
        val seenWords = hashSetOf<String>()
        val translated = ArrayList<SuggestedWordInfo>()
        for (candidate in ranked.candidates) {
            if (candidate.surface.isBlank()) continue
            // Keep a case-distinct personal or dictionary surface (for example LibreBoard beside
            // raw "libreboard"). The literal raw surface itself is injected by Suggest.
            if (rawText.isNotEmpty() && candidate.surface == rawText) continue
            val key = candidate.normalized to candidate.languageTag
            val supplemental = supplementalByKey[key].orEmpty()
            val personal = supplemental.firstOrNull {
                it.exactPersonalMatch || CandidateSource.PERSONAL in it.sources || CandidateSource.PERSONAL_PHRASE in it.sources
            }
            val classic = classicByKey[key]?.maxByOrNull { it.mScore }
            val candidateIsPersonal = CandidateSource.PERSONAL in candidate.sources ||
                CandidateSource.PERSONAL_PHRASE in candidate.sources
            val info = if (personal != null && (classic == null || candidateIsPersonal)) {
                personalInfo(candidate, inputStyle)
            } else {
                classic?.withSurface(candidate.surface) ?: generatedInfo(candidate, inputStyle)
            }
            if (seenWords.add(info.mWord)) translated += info
            if (translated.size == SuggestedWords.MAX_SUGGESTIONS) break
        }
        return LegacyFusionResult(translated, exactPersonalMatch, rawReplacementVeto, currentLock)
    }

    private fun updateWordLock(rawText: String, probabilities: Map<String, Double>): WordLock {
        if (rawText.isEmpty()) {
            resetWord()
            return languageLock.current()
        }
        if (!wordActive || (rawText.codePointCount(0, rawText.length) == 1 && previousRaw.length > 1)) {
            languageLock.beginWord()
            wordActive = true
        }
        previousRaw = rawText
        return languageLock.observe(LanguageEvidence(
            probabilities = probabilities,
            typedCharacterCount = rawText.codePointCount(0, rawText.length),
        ))
    }

    private fun languageProbabilities(
        classic: List<SuggestedWordInfo>,
        personal: List<Candidate>,
        enabledLanguages: List<String>,
        defaultLanguage: String,
    ): Map<String, Double> {
        val scores = enabledLanguages.distinct().associateWithTo(linkedMapOf()) { 0.0 }
        classic.forEach { info ->
            val language = languageTag(info, defaultLanguage)
            scores[language] = max(scores[language] ?: 0.0, info.mScore.toDouble())
        }
        personal.forEach { candidate ->
            val score = ln1p(max(0.0, candidate.components.personal ?: 0.0))
            scores[candidate.languageTag] = max(scores[candidate.languageTag] ?: 0.0, score)
        }
        if (scores.isEmpty()) return mapOf(defaultLanguage to 1.0)
        val maximum = scores.values.max()
        val temperature = max(1.0, scores.values.maxOf(::abs) * LANGUAGE_SCORE_TEMPERATURE_RATIO)
        val exponentials = scores.mapValues { (_, score) -> exp(((score - maximum) / temperature).coerceIn(-50.0, 0.0)) }
        val total = exponentials.values.sum()
        return exponentials.mapValues { (_, value) -> value / total }
    }

    private fun personalInfo(candidate: Candidate, inputStyle: InputStyle): SuggestedWordInfo {
        val personalScore = candidate.components.personal ?: 0.0
        return SuggestedWordInfo(
            candidate.surface,
            "",
            max(1, (personalScore * PERSONAL_SCORE_SCALE).toInt()),
            if (inputStyle == InputStyle.PREDICTION) SuggestedWordInfo.KIND_PREDICTION else SuggestedWordInfo.KIND_COMPLETION,
            Dictionary.DICTIONARY_USER_TYPED,
            SuggestedWordInfo.NOT_AN_INDEX,
            SuggestedWordInfo.NOT_A_CONFIDENCE,
        )
    }

    private fun generatedInfo(candidate: Candidate, inputStyle: InputStyle): SuggestedWordInfo = SuggestedWordInfo(
        candidate.surface,
        "",
        max(1, (candidate.totalScore.coerceIn(0.0, MAX_GENERATED_SCORE) * GENERATED_SCORE_SCALE).toInt()),
        if (inputStyle == InputStyle.PREDICTION) SuggestedWordInfo.KIND_PREDICTION else SuggestedWordInfo.KIND_CORRECTION,
        Dictionary.DICTIONARY_ENGINE_GENERATED,
        SuggestedWordInfo.NOT_AN_INDEX,
        SuggestedWordInfo.NOT_A_CONFIDENCE,
    )

    private fun SuggestedWordInfo.withSurface(surface: String): SuggestedWordInfo =
        if (surface == mWord) this else SuggestedWordInfo(
            surface,
            mPrevWordsContext,
            mScore,
            mKindAndFlags,
            mSourceDict,
            mIndexOfTouchPointOfSecondWord,
            mAutoCommitFirstWordConfidence,
        )

    private fun languageTag(info: SuggestedWordInfo, defaultLanguage: String): String =
        info.mSourceDict.mLocale?.toLanguageTag()?.takeIf(String::isNotBlank) ?: defaultLanguage

    private companion object {
        const val LANGUAGE_SCORE_TEMPERATURE_RATIO = 0.25
        const val PERSONAL_SCORE_SCALE = 100.0
        const val GENERATED_SCORE_SCALE = 100_000.0
        const val MAX_GENERATED_SCORE = 10.0
    }
}
