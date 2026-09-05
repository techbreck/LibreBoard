// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

interface CandidatePipeline {
    fun suggest(request: TypingRequest, deadline: Deadline): SuggestionBatch
}

data class CandidateKey(val normalized: String, val languageTag: String)

data class NeuralScoreResult(
    val availability: EngineAvailability,
    val scoresByCandidate: Map<CandidateKey, Double> = emptyMap(),
)

val Candidate.key: CandidateKey get() = CandidateKey(normalized, languageTag)

interface NeuralRescorer {
    fun score(request: TypingRequest, candidates: List<Candidate>, deadline: Deadline): NeuralScoreResult
}

data class SwipeDecodeResult(
    val availability: EngineAvailability,
    val candidates: List<Candidate> = emptyList(),
)

fun interface SwipeDecoder {
    fun decode(request: TypingRequest, deadline: Deadline): SwipeDecodeResult
}

data class CommitObservation(
    /** Newly committed tokens only. Editor context must never be copied into this list. */
    val tokens: List<String>,
    val languageTag: String,
    val timestampMillis: Long,
    val wasManualSelection: Boolean,
    val correctionRaw: String? = null,
    val contextFingerprint: String? = null,
    /** Earlier tokens confirmed by this IME session, used only to form bounded n-grams. */
    val precedingTokens: List<String> = emptyList(),
)

data class RejectionObservation(
    val raw: String,
    val replacement: String,
    val languageTag: String,
    val contextFingerprint: String,
    val timestampMillis: Long,
)

interface PersonalStore {
    fun observeCommit(observation: CommitObservation)
    fun observeRejection(observation: RejectionObservation)
    fun suggest(request: TypingRequest, deadline: Deadline): List<Candidate>
    fun export(): ByteArray
    fun wipe()
}
