// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class FusedCandidatePipelineTest {
    private val geometry = KeyGeometry(10f, 10f, emptyList())
    private val emptySwipe = SwipeDecoder { _, _ -> SwipeDecodeResult(EngineAvailability.AVAILABLE) }

    @Test
    fun modelTimeoutFallsBackAndKeepsRawCandidate() {
        val classic = CandidateSourceProvider { _, _ ->
            listOf(Candidate("this", languageTag = "en-US", sources = setOf(CandidateSource.STATIC_DICTIONARY),
                components = ScoreComponents(spatial = 1.0)))
        }
        val neural = object : NeuralRescorer {
            override fun score(request: TypingRequest, candidates: List<Candidate>, deadline: Deadline) =
                NeuralScoreResult(EngineAvailability.TIMEOUT)
        }
        val pipeline = FusedCandidatePipeline(classic, null, neural, null, emptySwipe)
        val batch = pipeline.suggest(request("thsi"), Deadline.afterMillis(100))
        assertEquals(EngineAvailability.TIMEOUT, batch.neuralAvailability)
        assertEquals("thsi", batch.candidates.first().surface)
        assertTrue(batch.candidates.any { it.surface == "this" })
    }

    @Test
    fun restrictedFieldReturnsNoCandidatesAndNoAutocorrection() {
        val pipeline = FusedCandidatePipeline(
            CandidateSourceProvider { _, _ -> error("must not query private field") },
            null,
            null,
            null,
            emptySwipe,
        )
        val batch = pipeline.suggest(request("secret", FieldPolicy.SENSITIVE), Deadline.afterMillis(100))
        assertTrue(batch.candidates.isEmpty())
        assertNull(batch.autoCorrection)
    }

    @Test
    fun missingCtcStillPublishesGeometricSwipe() {
        val geometric = SwipeDecoder { _, _ ->
            SwipeDecodeResult(EngineAvailability.AVAILABLE, listOf(
                Candidate("hello", languageTag = "en-US", sources = setOf(CandidateSource.GEOMETRIC_SWIPE),
                    components = ScoreComponents(spatial = 1.0))
            ))
        }
        val pipeline = FusedCandidatePipeline(CandidateSourceProvider { _, _ -> emptyList() }, null, null, null, geometric)
        val batch = pipeline.suggest(request("", style = InputStyle.SWIPE), Deadline.afterMillis(100))
        assertTrue(batch.candidates.any { it.surface == "hello" })
    }

    @Test
    fun ctcAndGeometricSwipeStartInParallel() {
        val bothStarted = CountDownLatch(2)
        fun decoder(word: String, source: CandidateSource) = SwipeDecoder { _, _ ->
            bothStarted.countDown()
            assertTrue("both decoders must start before either finishes", bothStarted.await(250, TimeUnit.MILLISECONDS))
            SwipeDecodeResult(EngineAvailability.AVAILABLE, listOf(
                Candidate(word, languageTag = "en-US", sources = setOf(source),
                    components = ScoreComponents(spatial = 1.0))
            ))
        }
        FusedCandidatePipeline(
            CandidateSourceProvider { _, _ -> emptyList() },
            null,
            null,
            decoder("ctc", CandidateSource.CTC_SWIPE),
            decoder("geometry", CandidateSource.GEOMETRIC_SWIPE),
        ).use { pipeline ->
            val batch = pipeline.suggest(request("", style = InputStyle.SWIPE), Deadline.afterMillis(500))
            assertTrue(batch.candidates.any { it.surface == "ctc" })
            assertTrue(batch.candidates.any { it.surface == "geometry" })
        }
    }

    @Test
    fun neuralScoresRemainScopedToCandidateLanguage() {
        val english = Candidate(
            "gift",
            languageTag = "en-US",
            sources = setOf(CandidateSource.STATIC_DICTIONARY),
            components = ScoreComponents(spatial = 1.0),
        )
        val german = english.copy(surface = "Gift", languageTag = "de")
        val neural = object : NeuralRescorer {
            override fun score(request: TypingRequest, candidates: List<Candidate>, deadline: Deadline) =
                NeuralScoreResult(
                    EngineAvailability.AVAILABLE,
                    mapOf(english.key to 4.0, german.key to -4.0),
                )
        }
        val pipeline = FusedCandidatePipeline(
            CandidateSourceProvider { _, _ -> listOf(english, german) },
            null,
            neural,
            null,
            emptySwipe,
        )

        val batch = pipeline.suggest(
            request("gif", enabledLanguages = listOf("en-US", "de")),
            Deadline.afterMillis(100),
        )

        assertEquals(4.0, batch.candidates.single { it.languageTag == "en-US" && it.normalized == "gift" }.components.context)
        assertEquals(-4.0, batch.candidates.single { it.languageTag == "de" && it.normalized == "gift" }.components.context)
    }

    private fun request(
        raw: String,
        policy: FieldPolicy = FieldPolicy.NORMAL,
        style: InputStyle = InputStyle.TAP,
        enabledLanguages: List<String> = listOf("en-US"),
    ) = TypingRequest.bounded(
        rawText = raw,
        geometry = geometry,
        enabledLanguages = enabledLanguages,
        fieldPolicy = policy,
        inputStyle = style,
        sequenceId = 1,
    )
}
