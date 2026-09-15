// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TypingRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ParallelSwipeDecoderTest {
    @Test
    fun unavailableCtcNeverSuppressesGeometricCandidates() {
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ -> SwipeDecodeResult(EngineAvailability.UNAVAILABLE) },
            geometricDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(EngineAvailability.AVAILABLE, listOf(candidate("cat", CandidateSource.GEOMETRIC_SWIPE, -0.2)))
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(100))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals("cat", result.candidates.single().surface)
    }

    @Test
    fun duplicateCandidatesMergeCtcAndGeometricProvenance() {
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(EngineAvailability.AVAILABLE, listOf(candidate("cat", CandidateSource.CTC_SWIPE, -0.1)))
            },
            geometricDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(EngineAvailability.AVAILABLE, listOf(candidate("cat", CandidateSource.GEOMETRIC_SWIPE, -0.3)))
            },
        )

        val candidate = decoder.decode(request(), Deadline.afterMillis(100)).candidates.single()

        assertTrue(CandidateSource.CTC_SWIPE in candidate.sources)
        assertTrue(CandidateSource.GEOMETRIC_SWIPE in candidate.sources)
        assertEquals(1.0, candidate.components.spatial!!, 0.0)
    }

    @Test
    fun unrelatedRawScoreScalesCannotSuppressOneDecoderSlate() {
        val ctc = (0 until 32).map { index ->
            candidate("ctc$index", CandidateSource.CTC_SWIPE, -index / 100.0)
        }
        val geometric = listOf(
            candidate("geometric", CandidateSource.GEOMETRIC_SWIPE, -100.0),
        )
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ -> SwipeDecodeResult(EngineAvailability.AVAILABLE, ctc) },
            geometricDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(EngineAvailability.AVAILABLE, geometric)
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(100))

        assertEquals(32, result.candidates.size)
        assertTrue(result.candidates.any { it.surface == "geometric" })
        assertTrue(result.candidates.any { CandidateSource.CTC_SWIPE in it.sources })
    }

    @Test
    fun lateCtcStillReturnsAlreadyCompletedGeometricSlate() {
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ ->
                Thread.sleep(30)
                SwipeDecodeResult(EngineAvailability.TIMEOUT)
            },
            geometricDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(EngineAvailability.AVAILABLE, listOf(candidate("cat", CandidateSource.GEOMETRIC_SWIPE, -0.2)))
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(10))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals("cat", result.candidates.single().surface)
    }

    @Test
    fun geometricPartialSlateSurvivesDeadlineExhaustedByCtc() {
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ ->
                Thread.sleep(60)
                SwipeDecodeResult(EngineAvailability.TIMEOUT)
            },
            geometricDecoder = SwipeDecoder { _, _ ->
                // The production decoder stops scanning at the deadline, then needs a few
                // milliseconds to sort and publish the partial slate.
                Thread.sleep(35)
                SwipeDecodeResult(
                    EngineAvailability.TIMEOUT,
                    listOf(candidate("cat", CandidateSource.GEOMETRIC_SWIPE, -0.2)),
                )
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(25))

        assertEquals(EngineAvailability.TIMEOUT, result.availability)
        assertEquals("cat", result.candidates.single().surface)
    }

    private fun candidate(word: String, source: CandidateSource, spatial: Double) = Candidate(
        word,
        languageTag = "en-US",
        sources = setOf(source),
        components = ScoreComponents(spatial = spatial, staticFrequency = 1.0),
        totalScore = spatial,
    )

    private fun request() = TypingRequest.bounded(
        rawText = "",
        geometry = KeyGeometry(1f, 1f, emptyList()),
        enabledLanguages = listOf("en-US"),
        fieldPolicy = FieldPolicy.NORMAL,
        inputStyle = InputStyle.SWIPE,
        sequenceId = 1,
    )
}
