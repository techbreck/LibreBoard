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

        assertEquals("cat", candidate.surface)
        assertTrue(CandidateSource.CTC_SWIPE in candidate.sources)
        assertEquals(1.0, candidate.components.spatial!!, 0.0)
    }

    @Test
    fun unrelatedRawScoreScalesCannotSuppressOneDecoderSlate() {
        val ctc = (0 until 31).map { index ->
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

        assertEquals("ctc0", result.candidates.first().surface)
        assertEquals(31, result.candidates.size)
        assertTrue(result.candidates.all { CandidateSource.CTC_SWIPE in it.sources })
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
    fun availableCtcDoesNotWaitForSlowGeometric() {
        var geometricCalled = false
        val started = System.nanoTime()
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(
                    EngineAvailability.AVAILABLE,
                    listOf(candidate("cities", CandidateSource.CTC_SWIPE, -0.1)),
                )
            },
            geometricDecoder = SwipeDecoder { _, _ ->
                geometricCalled = true
                Thread.sleep(200)
                SwipeDecodeResult(
                    EngineAvailability.AVAILABLE,
                    listOf(candidate("system", CandidateSource.GEOMETRIC_SWIPE, 5.0)),
                )
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(100))
        val elapsedMs = (System.nanoTime() - started) / 1_000_000.0

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals("cities", result.candidates.first().surface)
        assertTrue("elapsed ${elapsedMs}ms", elapsedMs < 80)
        assertTrue("geometric must not run when CTC proposed", !geometricCalled)
    }

    @Test
    fun timedOutCtcFallsBackToGeometricCandidates() {
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ -> SwipeDecodeResult(EngineAvailability.TIMEOUT) },
            geometricDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(
                    EngineAvailability.AVAILABLE,
                    listOf(candidate("highway", CandidateSource.GEOMETRIC_SWIPE, -0.2)),
                )
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(50))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals("highway", result.candidates.first().surface)
    }

    @Test
    fun timeoutCtcPartialsArePublishedWithoutWaitingForGeometric() {
        val started = System.nanoTime()
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(
                    EngineAvailability.TIMEOUT,
                    listOf(candidate("cities", CandidateSource.CTC_SWIPE, -0.1)),
                )
            },
            geometricDecoder = SwipeDecoder { _, _ ->
                Thread.sleep(200)
                SwipeDecodeResult(
                    EngineAvailability.AVAILABLE,
                    listOf(candidate("system", CandidateSource.GEOMETRIC_SWIPE, 5.0)),
                )
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(100))
        val elapsedMs = (System.nanoTime() - started) / 1_000_000.0

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals("cities", result.candidates.first().surface)
        assertTrue(CandidateSource.CTC_SWIPE in result.candidates.first().sources)
        assertTrue("elapsed ${elapsedMs}ms", elapsedMs < 80)
    }

    @Test
    fun emptyCtcFallsBackToGeometricAfterDeadline() {
        val decoder = ParallelSwipeDecoder(
            ctcDecoder = SwipeDecoder { _, _ ->
                Thread.sleep(30)
                SwipeDecodeResult(EngineAvailability.TIMEOUT)
            },
            geometricDecoder = SwipeDecoder { _, _ ->
                SwipeDecodeResult(
                    EngineAvailability.TIMEOUT,
                    listOf(candidate("cat", CandidateSource.GEOMETRIC_SWIPE, -0.2)),
                )
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(5))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
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
