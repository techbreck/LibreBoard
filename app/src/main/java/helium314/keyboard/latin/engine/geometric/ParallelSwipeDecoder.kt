// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.MAX_CANDIDATES
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.key
import kotlin.math.max
import kotlin.math.sqrt

/**
 * CTC owns the swipe slate whenever it proposes. Geometric runs only after CTC misses, using
 * leftover proposal budget (or a short grace window if CTC already exhausted it).
 *
 * Speculative parallel geometric scan was burning the proposal budget even after CTC had
 * finished: cancel() does not stop the scan, TIMEOUT partials were treated as a miss, and the
 * caller then waited out the remainder.
 */
class ParallelSwipeDecoder(
    private val ctcDecoder: SwipeDecoder,
    private val geometricDecoder: SwipeDecoder,
) : SwipeDecoder {
    override fun decode(request: TypingRequest, deadline: Deadline): SwipeDecodeResult {
        val ctc = runCatching { ctcDecoder.decode(request, deadline) }
            .getOrElse { SwipeDecodeResult(EngineAvailability.UNAVAILABLE) }
        if (hasUsableCtc(ctc)) {
            return SwipeDecodeResult(EngineAvailability.AVAILABLE, leading(ctc.candidates, emptyList()))
        }
        val geometricDeadline = if (deadline.remainingMillis > 0) {
            deadline
        } else {
            Deadline.afterMillis(GEOMETRIC_TIMEOUT_GRACE_MILLIS)
        }
        val geometric = runCatching { geometricDecoder.decode(request, geometricDeadline) }
            .getOrElse { SwipeDecodeResult(EngineAvailability.UNAVAILABLE) }
        val candidates = if (geometric.candidates.isNotEmpty()) {
            leading(geometric.candidates, ctc.candidates)
        } else {
            ctc.candidates.take(MAX_CANDIDATES)
        }
        val availability = when {
            geometric.availability == EngineAvailability.AVAILABLE || candidates.isNotEmpty() ->
                EngineAvailability.AVAILABLE
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

    /** Keep the leading decoder's order; append unique trailing surfaces up to the slate cap. */
    private fun leading(primary: List<Candidate>, secondary: List<Candidate>): List<Candidate> {
        val head = normalizeSpatial(primary)
        val extra = normalizeSpatial(secondary).associateBy { it.key }
        val mergedHead = head.map { candidate ->
            val other = extra[candidate.key] ?: return@map candidate
            candidate.copy(
                sources = candidate.sources + other.sources,
                components = ScoreComponents(
                    spatial = candidate.components.spatial,
                    staticFrequency = maximum(candidate.components.staticFrequency, other.components.staticFrequency),
                    personal = maximum(candidate.components.personal, other.components.personal),
                    context = maximum(candidate.components.context, other.components.context),
                    language = maximum(candidate.components.language, other.components.language),
                ),
                exactPersonalMatch = candidate.exactPersonalMatch || other.exactPersonalMatch,
                rejectionPenalty = max(candidate.rejectionPenalty, other.rejectionPenalty),
            )
        }
        val seen = mergedHead.mapTo(linkedSetOf()) { it.key }
        val tail = extra.values.filter { seen.add(it.key) }
        return (mergedHead + tail).take(MAX_CANDIDATES)
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
        /** Fallback window when CTC already spent the proposal budget and published nothing. */
        const val GEOMETRIC_TIMEOUT_GRACE_MILLIS = 15L

        fun hasUsableCtc(result: SwipeDecodeResult): Boolean =
            result.candidates.isNotEmpty() &&
                (result.availability == EngineAvailability.AVAILABLE ||
                    result.availability == EngineAvailability.TIMEOUT)
    }
}
