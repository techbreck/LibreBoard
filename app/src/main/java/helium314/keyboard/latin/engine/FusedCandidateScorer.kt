// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

enum class AutoCorrectionAggressiveness(val probability: Double, val margin: Double) {
    CAUTIOUS(0.95, 0.25),
    BALANCED(0.88, 0.18),
    AGGRESSIVE(0.78, 0.10),
}

data class ScoreWeights(
    val spatial: Double = 1.0,
    val staticFrequency: Double = 0.65,
    val personal: Double = 0.9,
    val language: Double = 0.7,
    val exactPersonalBonus: Double = 1.0,
    val contractionBonus: Double = 0.15,
    val compoundBonus: Double = 0.12,
    val editPenalty: Double = 0.18,
    val splitJoinPenalty: Double = 0.30,
    val languageSwitchPenalty: Double = 0.40,
)

data class RankedCandidates(val candidates: List<Candidate>, val autoCorrection: Candidate?)

class FusedCandidateScorer(private val weights: ScoreWeights = ScoreWeights()) {
    fun rank(
        raw: String,
        candidates: Collection<Candidate>,
        wordLock: WordLock,
        neuralStrength: Int,
        aggressiveness: AutoCorrectionAggressiveness,
    ): RankedCandidates {
        require(neuralStrength in 0..100)
        val normalizedRaw = normalizeCandidate(raw)
        val fused = candidates
            .filter { allowedByLock(it, wordLock) }
            .groupBy { it.normalized to it.languageTag }
            .map { (_, duplicates) -> merge(duplicates) }
            .toMutableList()

        if (fused.none { it.normalized == normalizedRaw }) {
            val language = lockedLanguage(wordLock) ?: fused.firstOrNull()?.languageTag.orEmpty()
            fused += Candidate(raw, languageTag = language, sources = setOf(CandidateSource.RAW))
        } else {
            val index = fused.indexOfFirst { it.normalized == normalizedRaw }
            fused[index] = fused[index].copy(sources = fused[index].sources + CandidateSource.RAW)
        }

        val spatial = normalize(fused.map { it.components.spatial })
        val frequency = normalize(fused.map { it.components.staticFrequency })
        val personal = normalize(fused.map { it.components.personal })
        val context = normalize(fused.map { it.components.context })
        val language = normalize(fused.map { it.components.language })
        val neuralWeight = neuralCoefficient(neuralStrength)

        val scored = fused.mapIndexed { index, candidate ->
            var score = spatial[index] * weights.spatial + frequency[index] * weights.staticFrequency +
                personal[index] * weights.personal + context[index] * neuralWeight + language[index] * weights.language
            if (candidate.exactPersonalMatch) score += weights.exactPersonalBonus
            if (CandidateSource.CONTRACTION in candidate.sources) score += weights.contractionBonus
            if (CandidateSource.COMPOUND in candidate.sources) score += weights.compoundBonus
            score -= candidate.edits.size * weights.editPenalty
            if (CandidateSource.SPLIT_JOIN in candidate.sources) score -= weights.splitJoinPenalty
            if (wordLock is WordLock.Unlocked && candidate.languageTag != fused.firstOrNull()?.languageTag) {
                score -= weights.languageSwitchPenalty
            }
            score -= candidate.rejectionPenalty
            candidate.copy(totalScore = score)
        }.sortedWith(compareByDescending<Candidate> { it.totalScore }.thenBy { it.normalized })

        val probabilities = softmax(scored.map { it.totalScore })
        val calibrated = scored.mapIndexed { index, candidate -> candidate.copy(calibratedProbability = probabilities[index]) }
        val winner = calibrated.firstOrNull()
        val rawCandidate = calibrated.firstOrNull { it.normalized == normalizedRaw }
        val margin = if (winner == null || rawCandidate == null) 0.0 else winner.calibratedProbability - rawCandidate.calibratedProbability
        val safeWinner = winner?.takeIf {
            it.normalized != normalizedRaw &&
                !rawCandidate!!.exactPersonalMatch &&
                it.rejectionPenalty <= 0.0 &&
                it.calibratedProbability >= aggressiveness.probability &&
                margin >= aggressiveness.margin &&
                validWordReplacementHasJointEvidence(rawCandidate, it)
        }
        // The strip contract is stable: raw text is always the left slot; the highest-ranked
        // correction can occupy the centre slot and remains independently marked for commit.
        val rawFirst = buildList {
            rawCandidate?.let(::add)
            calibrated.filterNotTo(this) { it === rawCandidate }
        }.take(MAX_CANDIDATES)
        return RankedCandidates(rawFirst, safeWinner)
    }

    fun neuralCoefficient(strength: Int): Double = when {
        strength <= 0 -> 0.0
        strength <= 50 -> strength / 50.0 * 0.7
        else -> 0.7 + (strength - 50) / 50.0 * 0.5
    }

    private fun allowedByLock(candidate: Candidate, wordLock: WordLock): Boolean =
        lockedLanguage(wordLock)?.let { it == candidate.languageTag } ?: true

    private fun lockedLanguage(lock: WordLock): String? = when (lock) {
        WordLock.Unlocked -> null
        is WordLock.Automatic -> lock.languageTag
        is WordLock.Manual -> lock.languageTag
    }

    private fun validWordReplacementHasJointEvidence(raw: Candidate, replacement: Candidate): Boolean {
        val rawIsValid = CandidateSource.STATIC_DICTIONARY in raw.sources || raw.exactPersonalMatch
        if (!rawIsValid) return true
        return replacement.components.context != null && replacement.components.spatial != null
    }

    private fun merge(candidates: List<Candidate>): Candidate {
        val first = candidates.first()
        return first.copy(
            sources = candidates.flatMapTo(mutableSetOf()) { it.sources },
            components = ScoreComponents(
                spatial = candidates.mapNotNull { it.components.spatial }.maxOrNull(),
                staticFrequency = candidates.mapNotNull { it.components.staticFrequency }.maxOrNull(),
                personal = candidates.mapNotNull { it.components.personal }.maxOrNull(),
                context = candidates.mapNotNull { it.components.context }.maxOrNull(),
                language = candidates.mapNotNull { it.components.language }.maxOrNull(),
            ),
            edits = candidates.flatMap { it.edits }.distinct(),
            exactPersonalMatch = candidates.any { it.exactPersonalMatch },
            rejectionPenalty = candidates.maxOf { it.rejectionPenalty },
        )
    }

    private fun normalize(values: List<Double?>): List<Double> {
        val present = values.filterNotNull()
        if (present.size < 2) return List(values.size) { 0.0 }
        val mean = present.average()
        val variance = present.sumOf { (it - mean).pow(2) } / present.size
        val deviation = sqrt(variance)
        if (deviation < 1e-9) return List(values.size) { 0.0 }
        return values.map { if (it == null) 0.0 else (it - mean) / deviation }
    }

    private fun softmax(values: List<Double>): List<Double> {
        if (values.isEmpty()) return emptyList()
        val max = values.max()
        val exponents = values.map { exp((it - max).coerceIn(-50.0, 50.0)) }
        val sum = exponents.sum()
        return exponents.map { it / sum }
    }
}
