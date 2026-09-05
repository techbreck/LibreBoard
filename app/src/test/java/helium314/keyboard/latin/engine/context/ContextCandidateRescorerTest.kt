// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.context

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateKey
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldClass
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.TypingRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ContextCandidateRescorerTest {
    private val tokenizer = FixtureTokenizer()

    @Test
    fun createsBoundedMasksAndMapsFiniteScores() {
        var captured: ContextModelBatch? = null
        val rescorer = ContextCandidateRescorer(tokenizer) { batch, _ ->
            captured = batch
            ContextInferenceResult(EngineAvailability.AVAILABLE, floatArrayOf(-2f, -0.5f))
        }
        val candidates = listOf(candidate("there", "en-US"), candidate("their", "en-US"))

        val result = rescorer.score(
            request("parked over", FieldPolicy.NORMAL, FieldClass.SHORT_MESSAGE),
            candidates,
            Deadline.afterMillis(100),
        )

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals(-2.0, result.scoresByCandidate[CandidateKey("there", "en-US")]!!, 0.0001)
        assertEquals(-0.5, result.scoresByCandidate[CandidateKey("their", "en-US")]!!, 0.0001)
        val batch = requireNotNull(captured)
        assertEquals(2, batch.batchSize)
        assertEquals(ContextCandidateRescorer.SEQUENCE_LENGTH, batch.sequenceLength)
        assertEquals(64, batch.inputIds.size)
        assertTrue(batch.attentionMask.count { it == 1L } > 0)
        assertEquals(2, batch.candidateMask.count { it == 1f })
        assertTrue(batch.fieldClasses.all { it == 1L })
    }

    @Test
    fun restrictedPolicyNeverTokenizesOrInvokesInference() {
        var inferenceCalls = 0
        val tokenizer = object : ContextTokenizer {
            override val paddingTokenId = 0
            override val beginningOfSequenceTokenId = 1
            override fun languageTokenId(languageTag: String) = error("must not inspect language")
            override fun encode(text: String, maximumTokens: Int, truncation: TokenTruncation) =
                error("must not tokenize")
        }
        val rescorer = ContextCandidateRescorer(tokenizer) { _, _ ->
            inferenceCalls++
            ContextInferenceResult(EngineAvailability.AVAILABLE, floatArrayOf(1f))
        }

        val result = rescorer.score(
            request("private context", FieldPolicy.SENSITIVE, FieldClass.RESTRICTED),
            listOf(candidate("secret", "en-US")),
            Deadline.afterMillis(100),
        )

        assertEquals(EngineAvailability.DISABLED, result.availability)
        assertEquals(0, inferenceCalls)
    }

    @Test
    fun unsupportedLanguageFallsBackWhileUnboundedTokenizerIsIncompatible() {
        val session = ContextInferenceSession { _, _ -> error("must not infer") }
        val rescorer = ContextCandidateRescorer(tokenizer, session)
        assertEquals(
            EngineAvailability.AVAILABLE,
            rescorer.score(
                request("", FieldPolicy.NORMAL, FieldClass.PLAIN),
                listOf(candidate("bonjour", "fr")),
                Deadline.afterMillis(100),
            ).availability,
        )

        val unbounded = ContextCandidateRescorer(object : ContextTokenizer by tokenizer {
            override fun encode(text: String, maximumTokens: Int, truncation: TokenTruncation) =
                ContextTokenization(IntArray(maximumTokens + 1) { 2 }, truncated = false)
        }, session)
        assertEquals(
            EngineAvailability.INCOMPATIBLE,
            unbounded.score(
                request("context", FieldPolicy.NORMAL, FieldClass.PLAIN),
                listOf(candidate("word", "en-US")),
                Deadline.afterMillis(100),
            ).availability,
        )
    }

    @Test
    fun runtimeFailuresTimeoutsAndMalformedScoresDegradeExplicitly() {
        val candidates = listOf(candidate("word", "en-US"))
        val request = request("context", FieldPolicy.NORMAL, FieldClass.PLAIN)
        val unavailable = ContextCandidateRescorer(tokenizer) { _, _ -> error("runtime failure") }
            .score(request, candidates, Deadline.afterMillis(100))
        assertEquals(EngineAvailability.UNAVAILABLE, unavailable.availability)

        val timedOut = ContextCandidateRescorer(tokenizer) { _, _ ->
            ContextInferenceResult(EngineAvailability.TIMEOUT)
        }.score(request, candidates, Deadline.afterMillis(100))
        assertEquals(EngineAvailability.TIMEOUT, timedOut.availability)

        val malformed = ContextCandidateRescorer(tokenizer) { _, _ ->
            ContextInferenceResult(EngineAvailability.AVAILABLE, floatArrayOf(Float.NaN))
        }.score(request, candidates, Deadline.afterMillis(100))
        assertEquals(EngineAvailability.INCOMPATIBLE, malformed.availability)
    }

    @Test
    fun equalNormalizedSurfacesRemainLanguageScoped() {
        val rescorer = ContextCandidateRescorer(tokenizer) { _, _ ->
            ContextInferenceResult(EngineAvailability.AVAILABLE, floatArrayOf(-3f, -1f))
        }
        val result = rescorer.score(
            request("", FieldPolicy.NORMAL, FieldClass.PLAIN),
            listOf(candidate("LibreBoard", "en-US"), candidate("libreboard", "de")),
            Deadline.afterMillis(100),
        )
        assertEquals(-3.0, result.scoresByCandidate[CandidateKey("libreboard", "en-US")]!!, 0.0001)
        assertEquals(-1.0, result.scoresByCandidate[CandidateKey("libreboard", "de")]!!, 0.0001)
    }

    @Test
    fun truncatedCandidateIsNotScoredFromOnlyItsPrefix() {
        var rows = 0
        val truncatingTokenizer = object : ContextTokenizer by tokenizer {
            override fun encode(text: String, maximumTokens: Int, truncation: TokenTruncation): ContextTokenization {
                if (text == "Datenschutzgrundverordnung") {
                    return ContextTokenization(IntArray(maximumTokens) { 12 }, truncated = true)
                }
                return tokenizer.encode(text, maximumTokens, truncation)
            }
        }
        val rescorer = ContextCandidateRescorer(truncatingTokenizer) { batch, _ ->
            rows = batch.batchSize
            ContextInferenceResult(EngineAvailability.AVAILABLE, floatArrayOf(-0.25f))
        }

        val result = rescorer.score(
            request("die", FieldPolicy.NORMAL, FieldClass.PLAIN),
            listOf(
                candidate("Datenschutzgrundverordnung", "de"),
                candidate("Regelung", "de"),
            ),
            Deadline.afterMillis(100),
        )

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals(1, rows)
        assertTrue(CandidateKey("datenschutzgrundverordnung", "de") !in result.scoresByCandidate)
        assertEquals(-0.25, result.scoresByCandidate[CandidateKey("regelung", "de")]!!, 0.0001)
    }

    private fun candidate(surface: String, language: String) = Candidate(
        surface = surface,
        languageTag = language,
        sources = setOf(CandidateSource.STATIC_DICTIONARY),
    )

    private fun request(context: String, policy: FieldPolicy, fieldClass: FieldClass) = TypingRequest.bounded(
        rawText = "word",
        precedingContext = context,
        geometry = KeyGeometry(1f, 1f, emptyList()),
        enabledLanguages = listOf("en-US", "de"),
        fieldPolicy = policy,
        fieldClass = fieldClass,
        inputStyle = InputStyle.TAP,
        sequenceId = 1,
    )

    private class FixtureTokenizer : ContextTokenizer {
        override val paddingTokenId = 0
        override val beginningOfSequenceTokenId = 1
        override fun languageTokenId(languageTag: String) = when (languageTag) {
            "en-US" -> 2
            "de" -> 3
            else -> null
        }

        override fun encode(text: String, maximumTokens: Int, truncation: TokenTruncation): ContextTokenization {
            val tokens = text.split(Regex("\\s+")).filter(String::isNotEmpty)
            val selected = when (truncation) {
                TokenTruncation.KEEP_START -> tokens.take(maximumTokens)
                TokenTruncation.KEEP_END -> tokens.takeLast(maximumTokens)
            }.mapIndexed { index, _ -> 10 + index }.toIntArray()
            return ContextTokenization(selected, truncated = tokens.size > maximumTokens)
        }
    }
}
