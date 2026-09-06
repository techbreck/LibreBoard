// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.ctc

import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.KeySlot
import helium314.keyboard.latin.engine.TouchPoint
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.geometric.LexiconWord
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CtcSwipeDecoderTest {
    private val labels = listOf("a", "l", "p", "o", "c", "t", "d", "n")
    private val geometry = KeyGeometry(
        width = 800f,
        height = 100f,
        keys = labels.mapIndexed { index, label ->
            KeySlot(index + 1, label, index * 100f + 50f, 50f, 90f, 90f)
        },
    )

    @Test
    fun featureTensorUsesTheFixedAbiAndLiveGeometry() {
        val features = CtcSwipeDecoder.featureTensor(
            listOf(TouchPoint(50f, 50f), TouchPoint(750f, 50f)),
            geometry,
        )

        assertEquals(CtcSwipeDecoder.PATH_POINTS * 2, features.pathCoordinates.size)
        assertEquals(CtcSwipeDecoder.KEY_SLOTS * 2, features.keyCenters.size)
        assertEquals(CtcSwipeDecoder.KEY_SLOTS, features.keyMask.size)
        assertEquals(features.keyCenters.first(), features.pathCoordinates.first(), 0.0001f)
        assertEquals(
            features.keyCenters[(labels.lastIndex * 2)],
            features.pathCoordinates[features.pathCoordinates.lastIndex - 1],
            0.0001f,
        )
        assertEquals(labels, features.keyLabels.take(labels.size))
        assertTrue(features.keyMask.take(labels.size).all { it == 1f })
        assertTrue(features.keyMask.drop(labels.size).all { it == 0f })
    }

    @Test
    fun blankBetweenRepeatedLettersDecodesDoubleLetter() {
        val decoder = decoder(
            emissions = listOf(classFor("a"), classFor("l"), 0, classFor("l")),
            words = listOf("al", "all"),
        )

        val result = decoder.decode(request(), Deadline.afterMillis(500))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals("all", result.candidates.first().surface)
    }

    @Test
    fun repeatedClassWithoutBlankCollapsesToOneLetter() {
        val decoder = decoder(
            emissions = listOf(classFor("l"), classFor("l")),
            words = listOf("l", "ll"),
        )

        val result = decoder.decode(request(), Deadline.afterMillis(500))

        assertEquals("l", result.candidates.first().surface)
    }

    @Test
    fun returnTripWordsRetainTheRepeatedNonConsecutiveKey() {
        listOf("pop", "lol").forEach { expected ->
            val decoder = decoder(expected.map(::classFor), listOf(expected))
            val result = decoder.decode(request(), Deadline.afterMillis(500))
            assertEquals(expected, result.candidates.first().surface)
        }
    }

    @Test
    fun lexiconLengthComesFromUnconstrainedCtcEmissionsInsteadOfCrossedKeys() {
        var requestedLength = -1
        val decoder = CtcSwipeDecoder(
            fixedSession("cat".map(::classFor)),
            SwipeLexicon { languageTags, approximateLength ->
                requestedLength = approximateLength
                sequenceOf(LexiconWord("cat", languageTags.first(), 100))
            },
        )

        val result = decoder.decode(request(), Deadline.afterMillis(500))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals(3, requestedLength)
        assertEquals("cat", result.candidates.first().surface)
    }

    @Test
    fun contractionSurfaceCanUseAnApostropheFreeGestureSequence() {
        val decoder = decoder("dont".map(::classFor), listOf("don't"))

        val result = decoder.decode(request(), Deadline.afterMillis(500))

        assertEquals("don't", result.candidates.first().surface)
    }

    @Test
    fun germanUmlautSurfaceUsesLanguageScopedBaseKeyGesture() {
        val decoder = CtcSwipeDecoder(
            fixedSession("tat".map(::classFor)),
            SwipeLexicon { _, _ -> sequenceOf(LexiconWord("tät", "de", 100)) },
        )

        val result = decoder.decode(
            request(enabledLanguages = listOf("de"), wordLock = WordLock.Manual("de")),
            Deadline.afterMillis(500),
        )

        assertEquals("tät", result.candidates.first().surface)
    }

    @Test
    fun manualLanguageLockRestrictsTheDecodedSlate() {
        val session = fixedSession("cat".map(::classFor))
        val lexicon = SwipeLexicon { _, _ ->
            sequenceOf(
                LexiconWord("cat", "en-US", 100),
                LexiconWord("cat", "de", 100),
            )
        }
        val decoder = CtcSwipeDecoder(session, lexicon)

        val result = decoder.decode(
            request(
                enabledLanguages = listOf("en-US", "de"),
                wordLock = WordLock.Manual("de"),
            ),
            Deadline.afterMillis(500),
        )

        assertEquals(listOf("de"), result.candidates.map { it.languageTag }.distinct())
    }

    @Test
    fun invalidPathOrGeometryCannotReachNativeInference() {
        var inferenceCalls = 0
        val session = CtcInferenceSession { _, _ ->
            inferenceCalls++
            fixedSession(emptyList()).infer(
                CtcSwipeDecoder.featureTensor(
                    listOf(TouchPoint(0f, 0f), TouchPoint(1f, 1f)),
                    geometry,
                ),
                Deadline.afterMillis(100),
            )
        }
        val decoder = CtcSwipeDecoder(
            session,
            SwipeLexicon { _, _ -> sequenceOf(LexiconWord("cat", "en-US", 1)) },
        )
        val request = TypingRequest.bounded(
            rawText = "",
            path = listOf(TouchPoint(Float.NaN, 0f), TouchPoint(1f, 1f)),
            geometry = geometry,
            enabledLanguages = listOf("en-US"),
            fieldPolicy = FieldPolicy.NORMAL,
            inputStyle = InputStyle.SWIPE,
            sequenceId = 1,
        )

        assertEquals(
            EngineAvailability.INCOMPATIBLE,
            decoder.decode(request, Deadline.afterMillis(500)).availability,
        )
        assertEquals(0, inferenceCalls)
    }

    @Test
    fun runtimeFailureAndMalformedTensorDegradeExplicitly() {
        val lexicon = SwipeLexicon { _, _ -> sequenceOf(LexiconWord("cat", "en-US", 1)) }
        val unavailable = CtcSwipeDecoder(
            CtcInferenceSession { _, _ -> throw IllegalStateException("runtime unavailable") },
            lexicon,
        ).decode(request(), Deadline.afterMillis(500))
        assertEquals(EngineAvailability.UNAVAILABLE, unavailable.availability)

        val malformed = CtcSwipeDecoder(
            CtcInferenceSession { _, _ ->
                CtcInferenceResult(EngineAvailability.AVAILABLE, FloatArray(3), 1, 3)
            },
            lexicon,
        ).decode(request(), Deadline.afterMillis(500))
        assertEquals(EngineAvailability.INCOMPATIBLE, malformed.availability)
    }

    @Test
    fun deadlineAndRuntimeTimeoutNeverMasqueradeAsAnEmptySuccess() {
        val decoder = CtcSwipeDecoder(
            CtcInferenceSession { _, _ -> CtcInferenceResult(EngineAvailability.TIMEOUT) },
            SwipeLexicon { _, _ -> sequenceOf(LexiconWord("cat", "en-US", 1)) },
        )

        assertEquals(
            EngineAvailability.TIMEOUT,
            decoder.decode(request(), Deadline.afterMillis(500)).availability,
        )
        assertEquals(
            EngineAvailability.TIMEOUT,
            decoder.decode(request(), Deadline.afterMillis(0)).availability,
        )
    }

    private fun decoder(emissions: List<Int>, words: List<String>) = CtcSwipeDecoder(
        fixedSession(emissions),
        SwipeLexicon { languageTags, _ ->
            words.asSequence().map { LexiconWord(it, languageTags.first(), 100) }
        },
    )

    private fun fixedSession(emissions: List<Int>) = CtcInferenceSession { _, _ ->
        val logits = FloatArray(CtcSwipeDecoder.OUTPUT_FRAMES * CtcSwipeDecoder.OUTPUT_CLASSES) { -12f }
        repeat(CtcSwipeDecoder.OUTPUT_FRAMES) { frame ->
            val outputClass = emissions.getOrElse(frame) { CtcSwipeDecoder.BLANK_CLASS }
            logits[frame * CtcSwipeDecoder.OUTPUT_CLASSES + outputClass] = 12f
        }
        CtcInferenceResult(
            EngineAvailability.AVAILABLE,
            logits,
            CtcSwipeDecoder.OUTPUT_FRAMES,
            CtcSwipeDecoder.OUTPUT_CLASSES,
        )
    }

    private fun classFor(character: Char): Int = classFor(character.toString())

    private fun classFor(label: String): Int = labels.indexOf(label).also {
        require(it >= 0) { "missing test key $label" }
    } + 1

    private fun request(
        enabledLanguages: List<String> = listOf("en-US"),
        wordLock: WordLock = WordLock.Unlocked,
    ) = TypingRequest.bounded(
        rawText = "",
        path = listOf(TouchPoint(50f, 50f), TouchPoint(750f, 50f)),
        geometry = geometry,
        enabledLanguages = enabledLanguages,
        wordLock = wordLock,
        fieldPolicy = FieldPolicy.NORMAL,
        inputStyle = InputStyle.SWIPE,
        sequenceId = 1,
    )
}
