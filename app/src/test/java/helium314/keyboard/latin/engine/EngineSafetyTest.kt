// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EngineSafetyTest {
    @Test
    fun boundedRequestKeepsOnlyRecentContextAndPointerHistory() {
        val geometry = KeyGeometry(10f, 10f, emptyList())
        val request = TypingRequest.bounded(
            rawText = "x",
            path = List(1200) { TouchPoint(0f, 0f) },
            precedingContext = "a".repeat(300),
            geometry = geometry,
            enabledLanguages = listOf("en-US", "en-US"),
            fieldPolicy = FieldPolicy.NORMAL,
            inputStyle = InputStyle.TAP,
            sequenceId = 42,
        )
        assertEquals(256, request.precedingContext.length)
        assertEquals(1024, request.path.size)
        assertEquals(listOf("en-US"), request.enabledLanguages)
    }

    @Test
    fun sequenceGateRejectsLateResults() {
        val gate = SequenceGate()
        val old = gate.next()
        val current = gate.next()
        assertFalse(gate.accepts(old))
        assertTrue(gate.accepts(current))
    }

    @Test
    fun threeOverrunsOpenCircuitWithinWindow() {
        var now = 1_000L
        val breaker = DeadlineCircuitBreaker(clockMillis = { now })
        repeat(3) { breaker.recordOverrun() }
        assertTrue(breaker.isOpen())
        now += 60_001
        assertFalse(breaker.isOpen())
    }
}
