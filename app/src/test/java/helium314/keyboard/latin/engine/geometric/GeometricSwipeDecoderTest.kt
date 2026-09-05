// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.KeySlot
import helium314.keyboard.latin.engine.TouchPoint
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GeometricSwipeDecoderTest {
    private val geometry = KeyGeometry(
        300f,
        100f,
        listOf(
            KeySlot(1, "c", 50f, 50f, 90f, 90f),
            KeySlot(2, "a", 150f, 50f, 90f, 90f),
            KeySlot(3, "t", 250f, 50f, 90f, 90f),
        ),
    )

    @Test
    fun nearestKeyTraceCollapsesOnlyConsecutiveDuplicates() {
        val path = listOf(
            TouchPoint(50f, 50f), TouchPoint(55f, 50f),
            TouchPoint(150f, 50f), TouchPoint(250f, 50f),
        )
        assertEquals("cat", TraceKeySequence.decode(path, geometry))
    }

    @Test
    fun ranksMatchingLiveGeometryTemplateFirst() {
        val lexicon = SwipeLexicon { _, _ ->
            sequenceOf(LexiconWord("cat", "en-US", 100), LexiconWord("tat", "en-US", 200))
        }
        val request = TypingRequest.bounded(
            rawText = "",
            path = listOf(TouchPoint(50f, 50f), TouchPoint(150f, 50f), TouchPoint(250f, 50f)),
            geometry = geometry,
            enabledLanguages = listOf("en-US"),
            wordLock = WordLock.Automatic("en-US"),
            fieldPolicy = FieldPolicy.NORMAL,
            inputStyle = InputStyle.SWIPE,
            sequenceId = 1,
        )
        val result = GeometricSwipeDecoder(lexicon).decode(request, Deadline.afterMillis(100))
        assertEquals("cat", result.candidates.first().surface)
        assertTrue(result.candidates.isNotEmpty())
    }
}
