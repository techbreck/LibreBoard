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
    @Test
    fun spellingPathsPreserveJoinersUnicodeAndGermanAlternativeOrder() {
        assertEquals(listOf("keyboard"), SwipeWordGesture.variants("KEYBOARD", "en-US"))
        assertEquals(listOf("dont"), SwipeWordGesture.variants("don't", "en-US"))
        assertEquals(listOf("cafés"), SwipeWordGesture.variants("Cafe\u0301’s", "fr-FR"))
        assertEquals(listOf("fußball"), SwipeWordGesture.variants("Fuß-ball", "en-US"))
        assertEquals(listOf("fußball", "fusball", "fussball"), SwipeWordGesture.variants("Fuß-ball", "de-DE"))
        assertEquals(listOf("äß", "äs", "äss", "aß", "as", "ass"), SwipeWordGesture.variants("Ä-ß", "de"))
        assertEquals(emptyList<String>(), SwipeWordGesture.variants("'-’", "en-US"))
    }

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

    @Test
    fun continuousCrossedKeysDoNotMasqueradeAsWordLength() {
        val rows = listOf(
            "qwertyuiop" to 50f,
            "asdfghjkl" to 150f,
            "zxcvbnm" to 250f,
        )
        val keys = mutableListOf<KeySlot>()
        rows.forEachIndexed { rowIndex, (labels, y) ->
            val offset = when (rowIndex) {
                0 -> 50f
                1 -> 100f
                else -> 200f
            }
            labels.forEachIndexed { index, character ->
                keys += KeySlot(keys.size + 1, character.toString(), offset + index * 100f, y, 90f, 90f)
            }
        }
        val qwerty = KeyGeometry(1_000f, 300f, keys)
        val centers = "both".map { character ->
            qwerty.keys.single { it.label == character.toString() }.let { TouchPoint(it.centerX, it.centerY) }
        }
        val densePath = centers.zipWithNext().flatMapIndexed { segment, (start, end) ->
            (0..16).map { step ->
                val fraction = step / 16f
                TouchPoint(
                    start.x + (end.x - start.x) * fraction,
                    start.y + (end.y - start.y) * fraction,
                )
            }.drop(if (segment == 0) 0 else 1)
        }

        val crossedKeyCount = TraceKeySequence.decode(densePath, qwerty).length
        val estimatedLength = SwipeLengthEstimate.fromPath(densePath, qwerty)

        assertTrue(crossedKeyCount > "both".length)
        assertEquals("both".length, estimatedLength)
    }

    @Test
    fun germanPopupLetterSurfaceCanUseItsBaseGestureKey() {
        val germanGeometry = KeyGeometry(
            300f,
            100f,
            listOf(
                KeySlot(1, "f", 50f, 50f, 90f, 90f),
                KeySlot(2, "u", 150f, 50f, 90f, 90f),
                KeySlot(3, "r", 250f, 50f, 90f, 90f),
            ),
        )
        val request = TypingRequest.bounded(
            rawText = "",
            path = listOf(TouchPoint(50f, 50f), TouchPoint(150f, 50f), TouchPoint(250f, 50f)),
            geometry = germanGeometry,
            enabledLanguages = listOf("de"),
            wordLock = WordLock.Automatic("de"),
            fieldPolicy = FieldPolicy.NORMAL,
            inputStyle = InputStyle.SWIPE,
            sequenceId = 1,
        )

        val result = GeometricSwipeDecoder(
            SwipeLexicon { _, _ -> sequenceOf(LexiconWord("für", "de", 100)) },
        ).decode(request, Deadline.afterMillis(100))

        assertEquals("für", result.candidates.first().surface)
        assertEquals(listOf("tät"), SwipeWordGesture.variants("tät", "en-US"))
        assertTrue("tat" in SwipeWordGesture.variants("tät", "de"))
    }
}
