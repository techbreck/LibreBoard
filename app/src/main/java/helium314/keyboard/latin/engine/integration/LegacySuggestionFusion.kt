// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.engine.AutoCorrectionAggressiveness
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FusedCandidateScorer
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.LanguageEvidence
import helium314.keyboard.latin.engine.LanguageLockController
import helium314.keyboard.latin.engine.MAX_CANDIDATES
import helium314.keyboard.latin.engine.NeuralRescorer
import helium314.keyboard.latin.engine.NeuralScoreResult
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.key
import helium314.keyboard.latin.engine.normalizeCandidate
import helium314.keyboard.latin.engine.runtime.LiveTypingEngine
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
    val neuralAvailability: EngineAvailability,
    val engineAutoCorrectionNormalized: String?,
)

/**
 * Translates retained AOSP dictionary results into LibreBoard candidates, ranks the union with
 * supplemental engine candidates, and translates the bounded slate back to the existing IME UI
 * contract.
 *
 * The legacy dictionary score already combines spatial and static evidence, so it is represented
 * once rather than double-counted. A verified context model may add one bounded score component;
 * unavailable, disabled, incompatible, or late inference leaves the classic ranking intact.
 */
internal class LegacySuggestionFusion(
    private val scorer: FusedCandidateScorer = FusedCandidateScorer(),
    private val languageLock: LanguageLockController = LanguageLockController(),
    private val neuralRescorer: NeuralRescorer = LiveTypingEngine.contextRescorer,
) {
    private var wordActive = false
    private var previousRaw = ""

    fun resetWord() {
        wordActive = false
        previousRaw = ""
        languageLock.endWord()
    }

    /** A user-selected subtype is a hard language choice, not merely a new prior. */
    fun selectLanguageManually(languageTag: String) {
        require(languageTag.isNotBlank())
        wordActive = false
        previousRaw = ""
        languageLock.selectManually(languageTag)
    }

    /** Restore boundary-only automatic language detection after an explicit user action. */
    fun releaseManualLanguageSelection() {
        wordActive = false
        previousRaw = ""
        languageLock.releaseManualSelection()
    }

    fun fuse(
        rawText: String,
        classicSuggestions: List<SuggestedWordInfo>,
        supplementalCandidates: List<Candidate>,
        enabledLanguageTags: List<String>,
        defaultLocale: Locale,
        inputStyle: InputStyle,
        typingRequest: TypingRequest? = null,
        neuralStrength: Int = 0,
        aggressiveness: AutoCorrectionAggressiveness = AutoCorrectionAggressiveness.BALANCED,
        neuralDeadline: Deadline? = null,
    ): LegacyFusionResult {
        require(neuralStrength in 0..100)
        val defaultLanguage = defaultLocale.toLanguageTag()
        val languageProbabilities = languageProbabilities(
            classicSuggestions,
            supplementalCandidates,
            enabledLanguageTags.ifEmpty { listOf(defaultLanguage) },
            defaultLanguage,
        )
        val currentLock = if (inputStyle == InputStyle.SWIPE) {
            resetWord()
            when (val requestedLock = typingRequest?.wordLock) {
                is WordLock.Manual -> requestedLock.takeIf {
                    it.languageTag in enabledLanguageTags
                } ?: WordLock.Automatic(
                    languageProbabilities.maxByOrNull { it.value }?.key ?: defaultLanguage,
                )
                else -> WordLock.Automatic(
                    languageProbabilities.maxByOrNull { it.value }?.key ?: defaultLanguage,
                )
            }
        } else {
            updateWordLock(rawText, languageProbabilities)
        }
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
        val unboundedUnion = classicCandidates + supplementalWithLanguage
        val rawNormalized = normalizeCandidate(rawText)
        val candidateUnion = if (rawText.isNotEmpty() && unboundedUnion.none { it.normalized == rawNormalized }) {
            val rawLanguage = when (currentLock) {
                is WordLock.Automatic -> currentLock.languageTag
                is WordLock.Manual -> currentLock.languageTag
                WordLock.Unlocked -> defaultLanguage
            }
            listOf(Candidate(rawText, languageTag = rawLanguage, sources = setOf(CandidateSource.RAW))) +
                unboundedUnion.take(MAX_CANDIDATES - 1)
        } else {
            unboundedUnion.take(MAX_CANDIDATES)
        }
        val neuralResult = when {
            neuralStrength == 0 -> NeuralScoreResult(EngineAvailability.DISABLED)
            typingRequest == null || neuralDeadline == null -> NeuralScoreResult(EngineAvailability.UNAVAILABLE)
            neuralDeadline.expired -> NeuralScoreResult(EngineAvailability.TIMEOUT)
            else -> runCatching {
                neuralRescorer.score(
                    typingRequest.copy(
                        rawText = rawText,
                        enabledLanguages = enabledLanguageTags.ifEmpty { listOf(defaultLanguage) }.distinct(),
                        wordLock = currentLock,
                        inputStyle = inputStyle,
                    ),
                    candidateUnion,
                    neuralDeadline,
                )
            }.getOrElse { NeuralScoreResult(EngineAvailability.UNAVAILABLE) }
        }
        val rescoredCandidates = if (neuralResult.availability == EngineAvailability.AVAILABLE) {
            candidateUnion.map { candidate ->
                val contextScore = neuralResult.scoresByCandidate[candidate.key]
                if (contextScore == null) candidate else candidate.copy(
                    components = candidate.components.copy(context = contextScore),
                )
            }
        } else {
            candidateUnion
        }
        val ranked = scorer.rank(
            raw = rawText,
            candidates = rescoredCandidates,
            wordLock = currentLock,
            neuralStrength = neuralStrength.takeIf {
                neuralResult.availability == EngineAvailability.AVAILABLE
            } ?: 0,
            aggressiveness = aggressiveness,
        )
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
        val engineAutoCorrectionNormalized = ranked.autoCorrection?.normalized?.takeIf { normalized ->
            translated.firstOrNull()?.mWord?.let(::normalizeCandidate) == normalized
        }
        return LegacyFusionResult(
            translated,
            exactPersonalMatch,
            rawReplacementVeto,
            currentLock,
            neuralResult.availability,
            engineAutoCorrectionNormalized,
        )
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
        val scores = enabledLanguages.distinct().associateWithTo(linkedMapOf()) { Double.NEGATIVE_INFINITY }
        classic.forEach { info ->
            val language = languageTag(info, defaultLanguage)
            if (language !in scores) return@forEach
            scores[language] = max(scores[language] ?: Double.NEGATIVE_INFINITY, info.mScore.toDouble())
        }
        personal.forEach { candidate ->
            if (candidate.languageTag !in scores) return@forEach
            val score = candidate.components.language ?: (
                (candidate.components.spatial ?: 0.0) +
                    (candidate.components.staticFrequency ?: 0.0) +
                    ln1p(max(0.0, candidate.components.personal ?: 0.0))
                )
            scores[candidate.languageTag] = max(
                scores[candidate.languageTag] ?: Double.NEGATIVE_INFINITY,
                score,
            )
        }
        if (scores.isEmpty()) return mapOf(defaultLanguage to 1.0)
        val observed = scores.values.filter(Double::isFinite)
        if (observed.isEmpty()) scores.replaceAll { _, _ -> 0.0 }
        else {
            val floor = observed.min() - max(1.0, observed.maxOf(::abs) * LANGUAGE_SCORE_TEMPERATURE_RATIO)
            scores.replaceAll { _, score -> score.takeIf(Double::isFinite) ?: floor }
        }
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
