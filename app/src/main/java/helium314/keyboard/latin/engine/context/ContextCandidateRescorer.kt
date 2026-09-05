// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.context

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateKey
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldClass
import helium314.keyboard.latin.engine.MAX_CANDIDATES
import helium314.keyboard.latin.engine.NeuralRescorer
import helium314.keyboard.latin.engine.NeuralScoreResult
import helium314.keyboard.latin.engine.TypingRequest

/** Fixed context-en-de-v1 tensor ABI. All arrays are row-major. */
data class ContextModelBatch(
    val batchSize: Int,
    val sequenceLength: Int,
    /** Shape `[batchSize, 32]`. */
    val inputIds: LongArray,
    /** Shape `[batchSize, 32]`; one for real tokens and zero for padding. */
    val attentionMask: LongArray,
    /** Shape `[batchSize, 32]`; one only for candidate tokens in the fixed `[24, 32)` suffix. */
    val candidateMask: FloatArray,
    /** Shape `[batchSize]`; stable IDs from [FieldClass]. */
    val fieldClasses: LongArray,
)

data class ContextInferenceResult(
    val availability: EngineAvailability,
    /** One finite log-likelihood per candidate row. */
    val candidateLogLikelihoods: FloatArray? = null,
)

fun interface ContextInferenceSession {
    fun infer(batch: ContextModelBatch, deadline: Deadline): ContextInferenceResult
}

interface ContextTokenizer {
    val paddingTokenId: Int
    val beginningOfSequenceTokenId: Int
    fun languageTokenId(languageTag: String): Int?
    fun encode(text: String, maximumTokens: Int, truncation: TokenTruncation): ContextTokenization
}

data class ContextTokenization(val tokenIds: IntArray, val truncated: Boolean)

enum class TokenTruncation { KEEP_START, KEEP_END }

/**
 * Batches bounded candidate likelihood requests for the static bilingual context model.
 *
 * Tokenization and inference are never entered when field policy forbids context. A malformed
 * tokenizer/model response degrades explicitly and cannot erase the classic suggestion slate.
 */
class ContextCandidateRescorer(
    private val tokenizer: ContextTokenizer,
    private val inferenceSession: ContextInferenceSession,
) : NeuralRescorer {
    override fun score(
        request: TypingRequest,
        candidates: List<Candidate>,
        deadline: Deadline,
    ): NeuralScoreResult {
        if (!request.fieldPolicy.allowsContextRead || !request.fieldPolicy.allowsSuggestions) {
            return NeuralScoreResult(EngineAvailability.DISABLED)
        }
        if (candidates.isEmpty()) return NeuralScoreResult(EngineAvailability.AVAILABLE)
        if (deadline.expired) return NeuralScoreResult(EngineAvailability.TIMEOUT)
        val prepared = runCatching { createBatches(request, candidates.take(MAX_CANDIDATES)) }
            .getOrElse { return NeuralScoreResult(EngineAvailability.INCOMPATIBLE) }
        if (prepared.isEmpty()) return NeuralScoreResult(EngineAvailability.AVAILABLE)
        val byCandidate = LinkedHashMap<CandidateKey, Double>()
        prepared.forEach { group ->
            if (deadline.expired) return NeuralScoreResult(EngineAvailability.TIMEOUT)
            val inference = runCatching { inferenceSession.infer(group.batch, deadline) }
                .getOrElse { return NeuralScoreResult(EngineAvailability.UNAVAILABLE) }
            if (inference.availability != EngineAvailability.AVAILABLE) {
                return NeuralScoreResult(inference.availability)
            }
            val scores = inference.candidateLogLikelihoods
            if (scores == null || scores.size != group.candidates.size || !scores.all(Float::isFinite)) {
                return NeuralScoreResult(EngineAvailability.INCOMPATIBLE)
            }
            group.candidates.forEachIndexed { index, candidate ->
                byCandidate.merge(
                    CandidateKey(candidate.normalized, candidate.languageTag),
                    scores[index].toDouble(),
                    ::maxOf,
                )
            }
        }
        if (deadline.expired) return NeuralScoreResult(EngineAvailability.TIMEOUT)
        return NeuralScoreResult(EngineAvailability.AVAILABLE, byCandidate)
    }

    private fun createBatches(request: TypingRequest, candidates: List<Candidate>): List<PreparedContextBatch> {
        val paddingToken = tokenizer.paddingTokenId.requireTokenId()
        val beginningToken = tokenizer.beginningOfSequenceTokenId.requireTokenId()
        val tokenizedCandidates = candidates.mapNotNull { candidate ->
            val languageToken = tokenizer.languageTokenId(candidate.languageTag)?.requireTokenId()
                ?: return@mapNotNull null
            val tokenization = tokenizer.encode(
                candidate.surface,
                MAX_CANDIDATE_TOKENS,
                TokenTruncation.KEEP_START,
            )
            if (tokenization.truncated || tokenization.tokenIds.isEmpty()) return@mapNotNull null
            require(tokenization.tokenIds.size <= MAX_CANDIDATE_TOKENS) { "candidate tokenization is unbounded" }
            tokenization.tokenIds.forEach { it.requireTokenId() }
            TokenizedCandidate(candidate, languageToken, tokenization.tokenIds)
        }
        if (tokenizedCandidates.isEmpty()) return emptyList()
        val contextTokens = tokenizer.encode(
            request.precedingContext,
            MAX_CONTEXT_TOKENS,
            TokenTruncation.KEEP_END,
        ).tokenIds
        require(contextTokens.size <= MAX_CONTEXT_TOKENS) { "context tokenization is unbounded" }
        contextTokens.forEach { it.requireTokenId() }
        return tokenizedCandidates.groupBy(TokenizedCandidate::languageToken).map { (languageToken, group) ->
            val rows = group.size
            val inputIds = LongArray(rows * SEQUENCE_LENGTH) { paddingToken.toLong() }
            val attentionMask = LongArray(rows * SEQUENCE_LENGTH)
            val candidateMask = FloatArray(rows * SEQUENCE_LENGTH)
            val fieldClasses = LongArray(rows) { request.fieldClass.modelId }
            val prefix = intArrayOf(beginningToken, languageToken) + contextTokens
            group.forEachIndexed { row, tokenized ->
                prefix.forEachIndexed { column, token ->
                    val offset = row * SEQUENCE_LENGTH + column
                    inputIds[offset] = token.toLong()
                    attentionMask[offset] = 1L
                }
                tokenized.tokenIds.forEachIndexed { candidateIndex, token ->
                    val offset = row * SEQUENCE_LENGTH + CANDIDATE_START + candidateIndex
                    inputIds[offset] = token.toLong()
                    attentionMask[offset] = 1L
                    candidateMask[offset] = 1f
                }
            }
            PreparedContextBatch(
                candidates = group.map(TokenizedCandidate::candidate),
                batch = ContextModelBatch(
                    batchSize = rows,
                    sequenceLength = SEQUENCE_LENGTH,
                    inputIds = inputIds,
                    attentionMask = attentionMask,
                    candidateMask = candidateMask,
                    fieldClasses = fieldClasses,
                ),
            )
        }
    }

    private data class TokenizedCandidate(
        val candidate: Candidate,
        val languageToken: Int,
        val tokenIds: IntArray,
    )

    private data class PreparedContextBatch(
        val candidates: List<Candidate>,
        val batch: ContextModelBatch,
    )

    private fun Int.requireTokenId(): Int {
        require(this >= 0) { "token IDs must be non-negative" }
        return this
    }

    private val FieldClass.modelId: Long
        get() = when (this) {
            FieldClass.PLAIN -> 0L
            FieldClass.SHORT_MESSAGE -> 1L
            FieldClass.SEARCH -> 2L
            FieldClass.CODE_OR_TERMINAL -> 3L
            FieldClass.RESTRICTED -> 4L
        }

    companion object {
        const val SEQUENCE_LENGTH = 32
        const val MAX_CANDIDATE_TOKENS = 8
        private const val PREFIX_TOKENS = 2
        const val CANDIDATE_START = SEQUENCE_LENGTH - MAX_CANDIDATE_TOKENS
        const val MAX_CONTEXT_TOKENS = CANDIDATE_START - PREFIX_TOKENS
    }
}
