// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.onnx

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateKey
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.ModelKind
import helium314.keyboard.latin.engine.ModelManifest
import helium314.keyboard.latin.engine.ModelRegistry
import helium314.keyboard.latin.engine.ProvenanceEntry
import helium314.keyboard.latin.engine.TypingRequest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.io.path.createTempDirectory

class InstalledNeuralModelLoaderTest {
    @Test
    fun verifiedContextFilesOpenAsRunnableRescorerAndShareSessionLifecycle() {
        val active = activeModel(ModelKind.CONTEXT_RESCORER, tokenizer = tokenizer())
        val runtime = FixtureRuntime(
            OnnxRunResult(
                EngineAvailability.AVAILABLE,
                mapOf("candidate_log_likelihood" to OnnxFloatOutput(floatArrayOf(-0.75f), longArrayOf(1))),
            ),
        )

        val opened = InstalledNeuralModelLoader.openContext(active) { _, contract ->
            assertEquals(LibreBoardOnnxContracts.contextEnDe, contract)
            OnnxSessionOpenResult(EngineAvailability.AVAILABLE, runtime)
        }

        assertEquals(EngineAvailability.AVAILABLE, opened.availability)
        val request = TypingRequest.bounded(
            rawText = "word",
            precedingContext = "the",
            geometry = KeyGeometry(1f, 1f, emptyList()),
            enabledLanguages = listOf("en-US", "de"),
            fieldPolicy = FieldPolicy.NORMAL,
            inputStyle = InputStyle.TAP,
            sequenceId = 7,
        )
        val candidate = Candidate("word", languageTag = "en-US", sources = setOf(CandidateSource.STATIC_DICTIONARY))
        val scores = opened.component!!.rescorer.score(request, listOf(candidate), Deadline.afterMillis(100))
        assertEquals(-0.75, scores.scoresByCandidate[CandidateKey("word", "en-US")]!!, 0.0001)
        opened.component.close()
        assertTrue(runtime.closed)
    }

    @Test
    fun absentWrongKindAndRuntimeFailureRemainExplicit() {
        assertEquals(
            EngineAvailability.UNAVAILABLE,
            InstalledNeuralModelLoader.openSwipe(null) { _, _ -> error("must not open") }.availability,
        )
        var runtimeCalls = 0
        val wrongKind = InstalledNeuralModelLoader.openSwipe(
            activeModel(ModelKind.CONTEXT_RESCORER, tokenizer = tokenizer()),
        ) { _, _ ->
            runtimeCalls++
            error("must not open")
        }
        assertEquals(EngineAvailability.INCOMPATIBLE, wrongKind.availability)
        assertNull(wrongKind.component)
        assertEquals(0, runtimeCalls)

        val unavailable = InstalledNeuralModelLoader.openSwipe(activeModel(ModelKind.SWIPE_CTC)) { _, contract ->
            assertEquals(LibreBoardOnnxContracts.swipeCtc, contract)
            OnnxSessionOpenResult(EngineAvailability.UNAVAILABLE)
        }
        assertEquals(EngineAvailability.UNAVAILABLE, unavailable.availability)
        assertNull(unavailable.component)
    }

    private fun activeModel(kind: ModelKind, tokenizer: ByteArray? = null): ModelRegistry.ActiveModel {
        val directory = createTempDirectory("libreboard-model-loader-").toFile()
        directory.deleteOnExit()
        val model = File(directory, "model.onnx").apply { writeText("fixture"); deleteOnExit() }
        val tokenizerFile = tokenizer?.let { bytes ->
            File(directory, "tokenizer.json").apply { writeBytes(bytes); deleteOnExit() }
        }
        return ModelRegistry.ActiveModel(
            manifest = ModelManifest(
                schemaVersion = 1,
                engineAbi = 1,
                modelKind = kind,
                tensorAbi = if (kind == ModelKind.SWIPE_CTC) "swipe-latin-v1" else "context-en-de-v1",
                locales = listOf("en-US", "de"),
                architecture = "fixture",
                parameterCount = 1,
                quantization = "fixture",
                modelSha256 = "0".repeat(64),
                tokenizerSha256 = tokenizer?.let { "1".repeat(64) },
                requiredOnnxOperators = listOf("MatMul"),
                license = "Apache-2.0",
                provenance = listOf(ProvenanceEntry("fixture", "1", "Apache-2.0", "https://example.invalid")),
                minimumAppVersionCode = 1,
            ),
            model = model,
            tokenizer = tokenizerFile,
        )
    }

    private fun tokenizer() = """
        {
          "schemaVersion": 1,
          "normalization": "NFKC_LOWER",
          "vocabulary": {
            "<pad>": 0,
            "<bos>": 1,
            "<unk>": 2,
            "<lang:en>": 3,
            "<lang:de>": 4,
            "▁": 5,
            "t": 6,
            "h": 7,
            "e": 8,
            "w": 9,
            "o": 10,
            "r": 11,
            "d": 12
          },
          "merges": [],
          "specialTokens": {
            "padding": "<pad>",
            "beginningOfSequence": "<bos>",
            "unknown": "<unk>",
            "languages": { "en": "<lang:en>", "de": "<lang:de>" }
          }
        }
    """.trimIndent().encodeToByteArray()

    private class FixtureRuntime(private val result: OnnxRunResult) : OnnxRuntimeSession {
        var closed = false
        override fun run(inputs: Map<String, OnnxInputTensor>, deadline: Deadline) = result
        override fun close() { closed = true }
    }
}
