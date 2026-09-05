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
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TypingRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class LiveSwipeModelSlotTest {
    @Test
    fun missingAndRestrictedModelsDegradeWithoutCallingDecoder() {
        LiveSwipeModelSlot().use { slot ->
            assertEquals(EngineAvailability.UNAVAILABLE, slot.decode(request(), Deadline.afterMillis(100)).availability)
            var called = false
            slot.install(SwipeDecoder { _, _ ->
                called = true
                SwipeDecodeResult(EngineAvailability.AVAILABLE)
            }, AutoCloseable {})

            assertEquals(
                EngineAvailability.DISABLED,
                slot.decode(request(FieldPolicy.SENSITIVE), Deadline.afterMillis(100)).availability,
            )
            assertFalse(called)
        }
    }

    @Test
    fun installedDecoderPublishesAndOwnerClosesExactlyOnce() {
        var closes = 0
        LiveSwipeModelSlot().use { slot ->
            slot.install(SwipeDecoder { _, _ ->
                SwipeDecodeResult(
                    EngineAvailability.AVAILABLE,
                    listOf(Candidate("cat", languageTag = "en-US", sources = setOf(CandidateSource.CTC_SWIPE))),
                )
            }, AutoCloseable { closes++ })

            assertEquals("cat", slot.decode(request(), Deadline.afterMillis(100)).candidates.single().surface)
            slot.clear()
            assertEquals(1, closes)
        }
        assertEquals(1, closes)
    }

    @Test
    fun reinstallingTheSameOwnerRefreshesDecoderWithoutClosingNativeState() {
        var closes = 0
        val owner = AutoCloseable { closes++ }
        LiveSwipeModelSlot().use { slot ->
            slot.install(decoder("old"), owner)
            assertEquals("old", slot.decode(request(), Deadline.afterMillis(100)).candidates.single().surface)

            slot.install(decoder("new"), owner)
            assertEquals("new", slot.decode(request(), Deadline.afterMillis(100)).candidates.single().surface)
            assertEquals(0, closes)
        }
        assertEquals(1, closes)
    }

    @Test
    fun refreshingTheSameOwnerResetsItsDeadlineCircuit() {
        val owner = AutoCloseable {}
        LiveSwipeModelSlot(DeadlineCircuitBreaker(maximumOverruns = 1)).use { slot ->
            slot.install(SwipeDecoder { _, _ -> SwipeDecodeResult(EngineAvailability.TIMEOUT) }, owner)
            assertEquals(EngineAvailability.TIMEOUT, slot.decode(request(), Deadline.afterMillis(100)).availability)
            assertEquals(EngineAvailability.CIRCUIT_OPEN, slot.decode(request(), Deadline.afterMillis(100)).availability)

            slot.install(decoder("recovered"), owner)
            assertEquals("recovered", slot.decode(request(), Deadline.afterMillis(100)).candidates.single().surface)
        }
    }

    @Test
    fun nonCooperativeDecoderTimesOutOpensCircuitAndDefersClose() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val ownerClosed = CountDownLatch(1)
        val slot = LiveSwipeModelSlot(DeadlineCircuitBreaker(maximumOverruns = 3))
        slot.install(SwipeDecoder { _, _ ->
            entered.countDown()
            while (true) {
                try {
                    if (release.await(1, TimeUnit.SECONDS)) break
                } catch (_: InterruptedException) {
                    // Emulate a native invocation that cannot be interrupted safely.
                }
            }
            SwipeDecodeResult(EngineAvailability.AVAILABLE)
        }, AutoCloseable { ownerClosed.countDown() })

        try {
            repeat(3) {
                assertEquals(EngineAvailability.TIMEOUT, slot.decode(request(), Deadline.afterMillis(20)).availability)
            }
            assertTrue(entered.await(100, TimeUnit.MILLISECONDS))
            assertEquals(EngineAvailability.CIRCUIT_OPEN, slot.decode(request(), Deadline.afterMillis(100)).availability)
            slot.clear()
            assertFalse(ownerClosed.await(20, TimeUnit.MILLISECONDS))
        } finally {
            release.countDown()
            assertTrue(ownerClosed.await(1, TimeUnit.SECONDS))
            slot.close()
        }
    }

    private fun request(policy: FieldPolicy = FieldPolicy.NORMAL) = TypingRequest.bounded(
        rawText = "",
        geometry = KeyGeometry(1f, 1f, emptyList()),
        enabledLanguages = listOf("en-US"),
        fieldPolicy = policy,
        inputStyle = InputStyle.SWIPE,
        sequenceId = 1,
    )

    private fun decoder(surface: String) = SwipeDecoder { _, _ ->
        SwipeDecodeResult(
            EngineAvailability.AVAILABLE,
            listOf(Candidate(surface, languageTag = "en-US", sources = setOf(CandidateSource.CTC_SWIPE))),
        )
    }
}
