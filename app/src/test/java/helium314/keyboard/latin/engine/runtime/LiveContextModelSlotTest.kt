// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.DeadlineCircuitBreaker
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.NeuralRescorer
import helium314.keyboard.latin.engine.NeuralScoreResult
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.key
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class LiveContextModelSlotTest {
    @Test
    fun missingAndRestrictedModelsDegradeWithoutCallingInference() {
        LiveContextModelSlot().use { slot ->
            assertEquals(
                EngineAvailability.UNAVAILABLE,
                slot.score(request(), candidates(), Deadline.afterMillis(100)).availability,
            )
            var called = false
            slot.install(NeuralRescorer { _, _, _ ->
                called = true
                NeuralScoreResult(EngineAvailability.AVAILABLE)
            }, AutoCloseable {})
            assertEquals(
                EngineAvailability.DISABLED,
                slot.score(
                    request(FieldPolicy.SENSITIVE),
                    candidates(),
                    Deadline.afterMillis(100),
                ).availability,
            )
            assertFalse(called)
        }
    }

    @Test
    fun installedModelScoresAndIsClosedExactlyOnce() {
        val closes = AtomicInteger()
        LiveContextModelSlot().use { slot ->
            slot.install(NeuralRescorer { _, candidates, _ ->
                NeuralScoreResult(
                    EngineAvailability.AVAILABLE,
                    candidates.associate { it.key to 2.0 },
                )
            }, AutoCloseable { closes.incrementAndGet() })

            val result = slot.score(request(), candidates(), Deadline.afterMillis(100))
            assertEquals(EngineAvailability.AVAILABLE, result.availability)
            assertEquals(2.0, result.scoresByCandidate.values.single(), 0.0)
            slot.clear()
            assertEquals(1, closes.get())
        }
        assertEquals(1, closes.get())
    }

    @Test
    fun nonCooperativeModelTimesOutOpensCircuitAndClosesOnlyAfterReturn() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val ownerClosed = CountDownLatch(1)
        val slot = LiveContextModelSlot(DeadlineCircuitBreaker(maximumOverruns = 3))
        slot.install(NeuralRescorer { _, _, _ ->
            entered.countDown()
            while (true) {
                try {
                    if (release.await(1, TimeUnit.SECONDS)) break
                } catch (_: InterruptedException) {
                    // Deliberately emulate a native call that ignores interruption.
                }
            }
            NeuralScoreResult(EngineAvailability.AVAILABLE)
        }, AutoCloseable { ownerClosed.countDown() })

        try {
            repeat(3) {
                assertEquals(
                    EngineAvailability.TIMEOUT,
                    slot.score(request(), candidates(), Deadline.afterMillis(20)).availability,
                )
            }
            assertTrue(entered.await(100, TimeUnit.MILLISECONDS))
            assertEquals(
                EngineAvailability.CIRCUIT_OPEN,
                slot.score(request(), candidates(), Deadline.afterMillis(100)).availability,
            )
            slot.clear()
            assertFalse(ownerClosed.await(20, TimeUnit.MILLISECONDS))
        } finally {
            release.countDown()
            assertTrue(ownerClosed.await(1, TimeUnit.SECONDS))
            slot.close()
        }
    }

    @Test
    fun cooperativeTimeoutsAlsoOpenTheSessionCircuit() {
        LiveContextModelSlot(DeadlineCircuitBreaker(maximumOverruns = 3)).use { slot ->
            slot.install(NeuralRescorer { _, _, _ ->
                NeuralScoreResult(EngineAvailability.TIMEOUT)
            }, AutoCloseable {})

            repeat(3) {
                assertEquals(
                    EngineAvailability.TIMEOUT,
                    slot.score(request(), candidates(), Deadline.afterMillis(100)).availability,
                )
            }
            assertEquals(
                EngineAvailability.CIRCUIT_OPEN,
                slot.score(request(), candidates(), Deadline.afterMillis(100)).availability,
            )
        }
    }

    private fun candidates() = listOf(
        Candidate("this", languageTag = "en-US", sources = setOf(CandidateSource.STATIC_DICTIONARY)),
    )

    private fun request(policy: FieldPolicy = FieldPolicy.NORMAL) = TypingRequest.bounded(
        rawText = "thsi",
        precedingContext = "type",
        geometry = KeyGeometry(1f, 1f, emptyList()),
        enabledLanguages = listOf("en-US"),
        fieldPolicy = policy,
        inputStyle = InputStyle.TAP,
        sequenceId = 1,
    )
}
