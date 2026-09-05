// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

fun interface CandidateSourceProvider {
    fun candidates(request: TypingRequest, deadline: Deadline): List<Candidate>
}

data class PipelineConfiguration(
    val neuralStrength: Int = 50,
    val autoCorrectionAggressiveness: AutoCorrectionAggressiveness = AutoCorrectionAggressiveness.BALANCED,
)

/**
 * Shared bounded pipeline for tap, next-word, and swipe results. Android adapters only translate
 * requests and published batches; policy, fallback, fusion, and commit eligibility stay here.
 */
class FusedCandidatePipeline(
    private val classicProvider: CandidateSourceProvider,
    private val personalStore: PersonalStore?,
    private val neuralRescorer: NeuralRescorer?,
    private val ctcSwipeDecoder: SwipeDecoder?,
    private val geometricSwipeDecoder: SwipeDecoder,
    private val scorer: FusedCandidateScorer = FusedCandidateScorer(),
    private val circuitBreaker: DeadlineCircuitBreaker = DeadlineCircuitBreaker(),
    private val configuration: PipelineConfiguration = PipelineConfiguration(),
    private val swipeExecutor: ExecutorService = Executors.newFixedThreadPool(2) { runnable ->
        Thread(runnable, "LibreBoardSwipeDecoder").apply { isDaemon = true }
    },
) : CandidatePipeline, AutoCloseable {
    override fun suggest(request: TypingRequest, deadline: Deadline): SuggestionBatch {
        if (!request.fieldPolicy.allowsSuggestions) {
            return SuggestionBatch(request.sequenceId, emptyList(), null, EngineAvailability.DISABLED)
        }

        val base = when (request.inputStyle) {
            InputStyle.SWIPE -> decodeSwipe(request, deadline)
            InputStyle.TAP, InputStyle.PREDICTION -> classicProvider.candidates(request, deadline) to EngineAvailability.AVAILABLE
        }
        val candidates = base.first.toMutableList()
        if (!deadline.expired && request.fieldPolicy.allowsPersistence) {
            candidates += personalStore?.suggest(request, deadline).orEmpty()
        }

        var neuralAvailability = when {
            configuration.neuralStrength == 0 -> EngineAvailability.DISABLED
            circuitBreaker.isOpen() -> EngineAvailability.CIRCUIT_OPEN
            neuralRescorer == null -> EngineAvailability.UNAVAILABLE
            deadline.expired -> EngineAvailability.TIMEOUT
            else -> EngineAvailability.AVAILABLE
        }
        if (neuralAvailability == EngineAvailability.AVAILABLE) {
            val neural = neuralRescorer!!.score(request.precedingContext, candidates.take(MAX_CANDIDATES), deadline)
            neuralAvailability = neural.availability
            if (neural.availability == EngineAvailability.TIMEOUT) circuitBreaker.recordOverrun()
            if (neural.availability == EngineAvailability.AVAILABLE) {
                candidates.replaceAll { candidate ->
                    candidate.copy(components = candidate.components.copy(context = neural.scoresByNormalizedCandidate[candidate.normalized]))
                }
            }
        }

        val ranked = scorer.rank(
            raw = request.rawText,
            candidates = candidates,
            wordLock = request.wordLock,
            neuralStrength = if (neuralAvailability == EngineAvailability.AVAILABLE) configuration.neuralStrength else 0,
            aggressiveness = configuration.autoCorrectionAggressiveness,
        )
        return SuggestionBatch(
            sequenceId = request.sequenceId,
            candidates = ranked.candidates,
            autoCorrection = ranked.autoCorrection?.takeIf { request.fieldPolicy.allowsAutoCorrection },
            neuralAvailability = neuralAvailability,
        )
    }

    private fun decodeSwipe(request: TypingRequest, deadline: Deadline): Pair<List<Candidate>, EngineAvailability> {
        val ctcDecoder = ctcSwipeDecoder
        if (deadline.expired || ctcDecoder == null || circuitBreaker.isOpen()) {
            val geometric = geometricSwipeDecoder.decode(request, deadline)
            val availability = when {
                circuitBreaker.isOpen() -> EngineAvailability.CIRCUIT_OPEN
                ctcSwipeDecoder == null -> EngineAvailability.UNAVAILABLE
                else -> EngineAvailability.TIMEOUT
            }
            return geometric.candidates to availability
        }

        // Both decoders see the same immutable request and hard deadline. Completion order is
        // irrelevant; a slow/unavailable CTC path can never prevent a completed geometric result.
        val completion = ExecutorCompletionService<Pair<Boolean, SwipeDecodeResult>>(swipeExecutor)
        val futures = listOf(
            completion.submit { false to geometricSwipeDecoder.decode(request, deadline) },
            completion.submit { true to ctcDecoder.decode(request, deadline) },
        )
        var geometric = SwipeDecodeResult(EngineAvailability.TIMEOUT)
        var ctc = SwipeDecodeResult(EngineAvailability.TIMEOUT)
        repeat(2) {
            val remaining = deadline.remainingMillis
            if (remaining <= 0) return@repeat
            val completed = completion.poll(remaining, TimeUnit.MILLISECONDS) ?: return@repeat
            runCatching { completed.get() }.getOrNull()?.let { (isCtc, result) ->
                if (isCtc) ctc = result else geometric = result
            }
        }
        futures.filterNot { it.isDone }.forEach { it.cancel(true) }
        if (ctc.availability == EngineAvailability.TIMEOUT) circuitBreaker.recordOverrun()
        return (ctc.candidates + geometric.candidates) to ctc.availability
    }

    override fun close() = swipeExecutor.shutdownNow().let { Unit }
}
