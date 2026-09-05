// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.onnx

import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.context.ContextCandidateRescorer
import helium314.keyboard.latin.engine.context.ContextModelBatch
import helium314.keyboard.latin.engine.ctc.CtcSwipeDecoder
import helium314.keyboard.latin.engine.ctc.CtcSwipeFeatures
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Optional
import kotlin.test.assertFailsWith

class OnnxInferenceAdaptersTest {
    @Test
    fun fixedContractsAcceptOnlyExactModelMetadata() {
        val swipe = LibreBoardOnnxContracts.swipeCtc
        swipe.requireCompatible(metadata(swipe))

        val wrong = metadata(swipe).copy(
            outputs = mapOf("logits" to (OnnxElementType.FLOAT32 to listOf(1, 31, 65))),
        )
        assertFailsWith<IllegalArgumentException> { swipe.requireCompatible(wrong) }

        val context = LibreBoardOnnxContracts.contextEnDe
        context.requireCompatible(metadata(context))
        val fixedBatch = metadata(context).copy(
            inputs = metadata(context).inputs.toMutableMap().apply {
                this["input_ids"] = OnnxElementType.INT64 to listOf(1, 32)
            },
        )
        assertFailsWith<IllegalArgumentException> { context.requireCompatible(fixedBatch) }
    }

    @Test
    fun coreOnlyBuildReportsUnavailableWithoutLoadingRuntimeClasses() {
        val result = OnnxRuntimeSessionFactory.open(
            model = File("missing.onnx"),
            contract = LibreBoardOnnxContracts.swipeCtc,
            runtimePackaged = false,
            classLoader = javaClass.classLoader!!,
        )

        assertEquals(EngineAvailability.UNAVAILABLE, result.availability)
        assertNull(result.session)
    }

    @Test
    fun reflectiveBindingExecutesAndClosesEveryNativeWrapper() {
        FixtureLifecycle.reset()
        val model = File.createTempFile("libreboard-ort-fixture", ".onnx").apply {
            writeText("fixture")
            deleteOnExit()
        }
        val opened = OnnxRuntimeSessionFactory.open(
            model = model,
            contract = LibreBoardOnnxContracts.swipeCtc,
            runtimePackaged = true,
            classLoader = javaClass.classLoader!!,
            classNames = OrtClassNames(
                FixtureOrtEnvironment::class.java.name,
                FixtureOnnxTensor::class.java.name,
                FixtureSessionOptions::class.java.name,
                FixtureTensorInfo::class.java.name,
            ),
        )

        assertEquals(EngineAvailability.AVAILABLE, opened.availability)
        val result = opened.session!!.run(
            mapOf(
                "path_coordinates" to OnnxInputTensor.Float32(FloatArray(128), longArrayOf(1, 64, 2)),
                "key_centers" to OnnxInputTensor.Float32(FloatArray(128), longArrayOf(1, 64, 2)),
                "key_mask" to OnnxInputTensor.Float32(FloatArray(64), longArrayOf(1, 64)),
            ),
            Deadline.afterMillis(100),
        )
        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertEquals(2_080, result.floatOutputs.getValue("logits").values.size)
        opened.session.close()
        assertEquals(4, FixtureLifecycle.tensorCloses)
        assertEquals(1, FixtureLifecycle.resultCloses)
        assertEquals(1, FixtureLifecycle.sessionCloses)
        assertEquals(1, FixtureLifecycle.optionsCloses)
        assertEquals(1, FixtureLifecycle.interOpThreads)
        assertEquals(2, FixtureLifecycle.intraOpThreads)
    }

    @Test
    fun ctcAdapterUsesExactNamesShapesAndReturnsFixedLogits() {
        val logits = FloatArray(CtcSwipeDecoder.OUTPUT_FRAMES * CtcSwipeDecoder.OUTPUT_CLASSES) { it.toFloat() }
        val runtime = CapturingRuntime(
            OnnxRunResult(
                EngineAvailability.AVAILABLE,
                mapOf("logits" to OnnxFloatOutput(logits, longArrayOf(1, 32, 65))),
            ),
        )
        val features = CtcSwipeFeatures(
            pathCoordinates = FloatArray(128) { 0.25f },
            keyCenters = FloatArray(128) { 0.5f },
            keyMask = FloatArray(64) { 1f },
            keyLabels = List(64) { null },
        )

        val result = OnnxCtcInferenceSession(runtime).infer(features, Deadline.afterMillis(100))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertArrayEquals(logits, result.logits, 0f)
        assertEquals(setOf("path_coordinates", "key_centers", "key_mask"), runtime.inputs!!.keys)
        assertTrue(runtime.inputs!!["path_coordinates"]!!.shape.contentEquals(longArrayOf(1, 64, 2)))
    }

    @Test
    fun contextAdapterUsesDynamicBatchAndReturnsOneScorePerRow() {
        val runtime = CapturingRuntime(
            OnnxRunResult(
                EngineAvailability.AVAILABLE,
                mapOf("candidate_log_likelihood" to OnnxFloatOutput(floatArrayOf(-2f, -1f), longArrayOf(2))),
            ),
        )
        val batch = ContextModelBatch(
            batchSize = 2,
            sequenceLength = ContextCandidateRescorer.SEQUENCE_LENGTH,
            inputIds = LongArray(64),
            attentionMask = LongArray(64) { 1 },
            candidateMask = FloatArray(64),
            fieldClasses = longArrayOf(0, 1),
        )

        val result = OnnxContextInferenceSession(runtime).infer(batch, Deadline.afterMillis(100))

        assertEquals(EngineAvailability.AVAILABLE, result.availability)
        assertArrayEquals(floatArrayOf(-2f, -1f), result.candidateLogLikelihoods, 0f)
        assertTrue(runtime.inputs!!["input_ids"]!!.shape.contentEquals(longArrayOf(2, 32)))
        assertTrue(runtime.inputs!!["field_class"]!!.shape.contentEquals(longArrayOf(2)))
    }

    @Test
    fun runtimeStateAndMalformedOutputsDegradeExplicitly() {
        val features = CtcSwipeFeatures(FloatArray(128), FloatArray(128), FloatArray(64), List(64) { null })
        val timeout = OnnxCtcInferenceSession(CapturingRuntime(OnnxRunResult(EngineAvailability.TIMEOUT)))
            .infer(features, Deadline.afterMillis(100))
        assertEquals(EngineAvailability.TIMEOUT, timeout.availability)

        val malformed = OnnxCtcInferenceSession(CapturingRuntime(
            OnnxRunResult(
                EngineAvailability.AVAILABLE,
                mapOf("logits" to OnnxFloatOutput(FloatArray(3), longArrayOf(1, 1, 3))),
            ),
        )).infer(features, Deadline.afterMillis(100))
        assertEquals(EngineAvailability.INCOMPATIBLE, malformed.availability)
    }

    private fun metadata(contract: OnnxModelContract) = OnnxModelMetadata(
        inputs = contract.inputs.associate { it.name to (it.type to it.shape) },
        outputs = contract.outputs.associate { it.name to (it.type to it.shape) },
    )

    private class CapturingRuntime(private val result: OnnxRunResult) : OnnxRuntimeSession {
        var inputs: Map<String, OnnxInputTensor>? = null

        override fun run(inputs: Map<String, OnnxInputTensor>, deadline: Deadline): OnnxRunResult {
            this.inputs = inputs
            return result
        }

        override fun close() = Unit
    }
}

private object FixtureLifecycle {
    var tensorCloses = 0
    var resultCloses = 0
    var sessionCloses = 0
    var optionsCloses = 0
    var interOpThreads = 0
    var intraOpThreads = 0

    fun reset() {
        tensorCloses = 0
        resultCloses = 0
        sessionCloses = 0
        optionsCloses = 0
        interOpThreads = 0
        intraOpThreads = 0
    }
}

private class FixtureOrtEnvironment {
    fun createSession(path: String, options: FixtureSessionOptions): FixtureOrtSession {
        require(path.isNotBlank())
        require(options.open)
        return FixtureOrtSession()
    }

    companion object {
        private val environment = FixtureOrtEnvironment()

        @JvmStatic
        fun getEnvironment(): FixtureOrtEnvironment = environment
    }
}

private class FixtureSessionOptions : AutoCloseable {
    var open = true
        private set

    fun setInterOpNumThreads(count: Int) {
        FixtureLifecycle.interOpThreads = count
    }

    fun setIntraOpNumThreads(count: Int) {
        FixtureLifecycle.intraOpThreads = count
    }

    override fun close() {
        check(open)
        open = false
        FixtureLifecycle.optionsCloses++
    }
}

private enum class FixtureJavaType { FLOAT, INT64 }

private class FixtureTensorInfo(
    @JvmField val type: FixtureJavaType,
    private val dimensions: LongArray,
) {
    fun getShape(): LongArray = dimensions.copyOf()
}

private class FixtureNodeInfo(private val info: FixtureTensorInfo) {
    fun getInfo(): FixtureTensorInfo = info
}

private class FixtureOnnxTensor private constructor(
    private val info: FixtureTensorInfo,
    private val floats: FloatBuffer? = null,
) : AutoCloseable {
    fun getInfo(): FixtureTensorInfo = info
    fun getFloatBuffer(): FloatBuffer = requireNotNull(floats).duplicate()

    override fun close() {
        FixtureLifecycle.tensorCloses++
    }

    companion object {
        @JvmStatic
        fun createTensor(
            environment: FixtureOrtEnvironment,
            values: FloatBuffer,
            shape: LongArray,
        ): FixtureOnnxTensor {
            requireNotNull(environment)
            require(values.remaining() > 0)
            return FixtureOnnxTensor(FixtureTensorInfo(FixtureJavaType.FLOAT, shape))
        }

        @JvmStatic
        fun createTensor(
            environment: FixtureOrtEnvironment,
            values: LongBuffer,
            shape: LongArray,
        ): FixtureOnnxTensor {
            requireNotNull(environment)
            require(values.remaining() > 0)
            return FixtureOnnxTensor(FixtureTensorInfo(FixtureJavaType.INT64, shape))
        }

        fun output(values: FloatArray, shape: LongArray) = FixtureOnnxTensor(
            FixtureTensorInfo(FixtureJavaType.FLOAT, shape),
            FloatBuffer.wrap(values),
        )
    }
}

private class FixtureOrtResult(
    private val values: Map<String, FixtureOnnxTensor>,
) : AutoCloseable {
    fun get(name: String): Optional<FixtureOnnxTensor> = Optional.ofNullable(values[name])

    override fun close() {
        values.values.forEach(FixtureOnnxTensor::close)
        FixtureLifecycle.resultCloses++
    }
}

private class FixtureOrtSession : AutoCloseable {
    fun getInputInfo(): Map<String, FixtureNodeInfo> = linkedMapOf(
        "path_coordinates" to node(FixtureJavaType.FLOAT, 1, 64, 2),
        "key_centers" to node(FixtureJavaType.FLOAT, 1, 64, 2),
        "key_mask" to node(FixtureJavaType.FLOAT, 1, 64),
    )

    fun getOutputInfo(): Map<String, FixtureNodeInfo> = linkedMapOf(
        "logits" to node(FixtureJavaType.FLOAT, 1, 32, 65),
    )

    fun run(inputs: Map<String, FixtureOnnxTensor>): FixtureOrtResult {
        assertEquals(setOf("path_coordinates", "key_centers", "key_mask"), inputs.keys)
        return FixtureOrtResult(mapOf(
            "logits" to FixtureOnnxTensor.output(FloatArray(2_080) { it.toFloat() }, longArrayOf(1, 32, 65)),
        ))
    }

    override fun close() {
        FixtureLifecycle.sessionCloses++
    }

    private fun node(type: FixtureJavaType, vararg shape: Long) =
        FixtureNodeInfo(FixtureTensorInfo(type, shape))
}
