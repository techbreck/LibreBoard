// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateKey
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.MAX_CANDIDATES
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.key
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.math.max
import kotlin.math.sqrt

/** Runs the pure-Kotlin fallback beside optional CTC inference and preserves either partial slate. */
class ParallelSwipeDecoder(
    private val ctcDecoder: SwipeDecoder,
    private val geometricDecoder: SwipeDecoder,
    private val geometricExecutor: ExecutorService = sharedGeometricExecutor,
) : SwipeDecoder {
    override fun decode(request: TypingRequest, deadline: Deadline): SwipeDecodeResult {
        val geometricFuture = geometricExecutor.submit<SwipeDecodeResult> {
            runCatching { geometricDecoder.decode(request, deadline) }
                .getOrElse { SwipeDecodeResult(EngineAvailability.UNAVAILABLE) }
        }
        val ctc = runCatching { ctcDecoder.decode(request, deadline) }
            .getOrElse { SwipeDecodeResult(EngineAvailability.UNAVAILABLE) }
        val geometric = try {
            geometricFuture.get(deadline.remainingMillis, TimeUnit.MILLISECONDS)
        } catch (_: TimeoutException) {
            geometricFuture.cancel(true)
            SwipeDecodeResult(EngineAvailability.TIMEOUT)
        } catch (_: InterruptedException) {
            geometricFuture.cancel(true)
            Thread.currentThread().interrupt()
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        } catch (_: Throwable) {
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        }
        val candidates = merge(ctc.candidates, geometric.candidates)
        val availability = when {
            ctc.availability == EngineAvailability.AVAILABLE ||
                geometric.availability == EngineAvailability.AVAILABLE -> EngineAvailability.AVAILABLE
            ctc.availability == EngineAvailability.TIMEOUT ||
                geometric.availability == EngineAvailability.TIMEOUT -> EngineAvailability.TIMEOUT
            ctc.availability == EngineAvailability.CIRCUIT_OPEN -> EngineAvailability.CIRCUIT_OPEN
            ctc.availability == EngineAvailability.INCOMPATIBLE -> EngineAvailability.INCOMPATIBLE
            ctc.availability == EngineAvailability.DISABLED &&
                geometric.availability == EngineAvailability.DISABLED -> EngineAvailability.DISABLED
            else -> EngineAvailability.UNAVAILABLE
        }
        return SwipeDecodeResult(availability, candidates)
    }

    private fun merge(ctcCandidates: List<Candidate>, geometricCandidates: List<Candidate>): List<Candidate> {
        // CTC log probabilities and geometric template costs do not share a numerical scale.
        // Normalize each completed slate before the bounded union so either decoder can contribute
        // candidates; the shared scorer normalizes the resulting spatial component again alongside
        // static, personal, language, and context evidence.
        val candidates = normalizeSpatial(ctcCandidates) + normalizeSpatial(geometricCandidates)
        val merged = linkedMapOf<CandidateKey, Candidate>()
        candidates.forEach { candidate ->
            val previous = merged[candidate.key]
            merged[candidate.key] = if (previous == null) candidate else previous.copy(
                sources = previous.sources + candidate.sources,
                components = ScoreComponents(
                    spatial = maximum(previous.components.spatial, candidate.components.spatial),
                    staticFrequency = maximum(previous.components.staticFrequency, candidate.components.staticFrequency),
                    personal = maximum(previous.components.personal, candidate.components.personal),
                    context = maximum(previous.components.context, candidate.components.context),
                    language = maximum(previous.components.language, candidate.components.language),
                ),
                exactPersonalMatch = previous.exactPersonalMatch || candidate.exactPersonalMatch,
                rejectionPenalty = max(previous.rejectionPenalty, candidate.rejectionPenalty),
                totalScore = max(previous.totalScore, candidate.totalScore),
                calibratedProbability = max(previous.calibratedProbability, candidate.calibratedProbability),
            )
        }
        return merged.values.sortedWith(
            compareByDescending<Candidate> { it.components.spatial ?: Double.NEGATIVE_INFINITY }
                .thenByDescending { it.components.staticFrequency ?: Double.NEGATIVE_INFINITY }
                .thenBy { it.surface },
        ).take(MAX_CANDIDATES)
    }

    private fun normalizeSpatial(candidates: List<Candidate>): List<Candidate> {
        val values = candidates.mapNotNull { it.components.spatial }
        if (values.isEmpty()) return candidates
        val normalized = when {
            values.size == 1 -> mapOf(values.single() to 1.0)
            else -> {
                val mean = values.average()
                val variance = values.sumOf { value ->
                    val difference = value - mean
                    difference * difference
                } / values.size
                val deviation = sqrt(variance)
                if (deviation < 1e-9) values.distinct().associateWith { 0.0 }
                else values.distinct().associateWith { (it - mean) / deviation }
            }
        }
        return candidates.map { candidate ->
            val spatial = candidate.components.spatial?.let(normalized::getValue)
            candidate.copy(
                components = candidate.components.copy(spatial = spatial),
                totalScore = spatial ?: candidate.totalScore,
            )
        }
    }

    private fun maximum(first: Double?, second: Double?): Double? = when {
        first == null -> second
        second == null -> first
        else -> max(first, second)
    }

    private companion object {
        val sharedGeometricExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "LibreBoardGeometricSwipe").apply { isDaemon = true }
        }
    }
}
