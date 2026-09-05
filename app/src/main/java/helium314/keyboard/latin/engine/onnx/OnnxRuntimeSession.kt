// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.onnx

import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import java.io.File
import java.lang.reflect.InvocationTargetException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.Optional

enum class OnnxElementType { FLOAT32, INT64 }

data class OnnxTensorSpec(
    val name: String,
    val type: OnnxElementType,
    /** A value of -1 is a required dynamic dimension in model metadata. */
    val shape: List<Long>,
) {
    init {
        require(name.isNotBlank())
        require(shape.isNotEmpty() && shape.all { it == DYNAMIC_DIMENSION || it > 0 })
    }
}

data class OnnxModelContract(
    val inputs: List<OnnxTensorSpec>,
    val outputs: List<OnnxTensorSpec>,
) {
    init {
        require(inputs.isNotEmpty() && outputs.isNotEmpty())
        require(inputs.map(OnnxTensorSpec::name).distinct().size == inputs.size)
        require(outputs.map(OnnxTensorSpec::name).distinct().size == outputs.size)
    }
}

sealed interface OnnxInputTensor {
    val shape: LongArray
    val elementCount: Int

    data class Float32(val values: FloatArray, override val shape: LongArray) : OnnxInputTensor {
        override val elementCount: Int get() = values.size
    }

    data class Int64(val values: LongArray, override val shape: LongArray) : OnnxInputTensor {
        override val elementCount: Int get() = values.size
    }
}

data class OnnxFloatOutput(val values: FloatArray, val shape: LongArray)

data class OnnxRunResult(
    val availability: EngineAvailability,
    val floatOutputs: Map<String, OnnxFloatOutput> = emptyMap(),
)

interface OnnxRuntimeSession : AutoCloseable {
    fun run(inputs: Map<String, OnnxInputTensor>, deadline: Deadline): OnnxRunResult
}

data class OnnxSessionOpenResult(
    val availability: EngineAvailability,
    val session: OnnxRuntimeSession? = null,
)

/**
 * Loads only the Java API packaged in LibreBoard's source-built ONNX Runtime AAR. Reflection keeps
 * ordinary core-only builds compilable without pulling a stock Maven runtime into the dependency
 * graph. No class name or native path comes from a model pack.
 */
object OnnxRuntimeSessionFactory {
    fun open(model: File, contract: OnnxModelContract): OnnxSessionOpenResult = open(
        model = model,
        contract = contract,
        runtimePackaged = BuildConfig.LIBREBOARD_ONNX_RUNTIME_PACKAGED,
        classLoader = requireNotNull(OnnxRuntimeSessionFactory::class.java.classLoader),
    )

    internal fun open(
        model: File,
        contract: OnnxModelContract,
        runtimePackaged: Boolean,
        classLoader: ClassLoader,
        classNames: OrtClassNames = OrtClassNames.PRODUCTION,
    ): OnnxSessionOpenResult {
        if (!runtimePackaged) return OnnxSessionOpenResult(EngineAvailability.UNAVAILABLE)
        if (!model.isFile || model.length() <= 0L) return OnnxSessionOpenResult(EngineAvailability.INCOMPATIBLE)
        return try {
            OnnxSessionOpenResult(
                EngineAvailability.AVAILABLE,
                ReflectiveOrtSession.open(model, contract, classLoader, classNames),
            )
        } catch (failure: Throwable) {
            val cause = failure.unwrapInvocation()
            val availability = if (cause is ClassNotFoundException || cause is LinkageError) {
                EngineAvailability.UNAVAILABLE
            } else {
                EngineAvailability.INCOMPATIBLE
            }
            OnnxSessionOpenResult(availability)
        }
    }
}

internal data class OrtClassNames(
    val environment: String,
    val tensor: String,
    val sessionOptions: String,
    val tensorInfo: String,
) {
    companion object {
        val PRODUCTION = OrtClassNames(
            environment = "ai.onnxruntime.OrtEnvironment",
            tensor = "ai.onnxruntime.OnnxTensor",
            sessionOptions = "ai.onnxruntime.OrtSession\$SessionOptions",
            tensorInfo = "ai.onnxruntime.TensorInfo",
        )
    }
}

internal data class OnnxModelMetadata(
    val inputs: Map<String, Pair<OnnxElementType, List<Long>>>,
    val outputs: Map<String, Pair<OnnxElementType, List<Long>>>,
)

internal fun OnnxModelContract.requireCompatible(metadata: OnnxModelMetadata) {
    requireMetadata(inputs, metadata.inputs, "input")
    requireMetadata(outputs, metadata.outputs, "output")
}

private fun requireMetadata(
    expected: List<OnnxTensorSpec>,
    actual: Map<String, Pair<OnnxElementType, List<Long>>>,
    label: String,
) {
    require(actual.keys == expected.mapTo(linkedSetOf(), OnnxTensorSpec::name)) {
        "ONNX $label names do not match the LibreBoard tensor ABI"
    }
    expected.forEach { spec ->
        val (type, shape) = requireNotNull(actual[spec.name])
        require(type == spec.type) { "ONNX $label ${spec.name} has the wrong element type" }
        require(shape == spec.shape) { "ONNX $label ${spec.name} has the wrong model shape" }
    }
}

private class ReflectiveOrtSession(
    private val environment: Any,
    private val nativeSession: Any,
    private val sessionOptions: AutoCloseable,
    private val contract: OnnxModelContract,
    private val environmentClass: Class<*>,
    private val tensorClass: Class<*>,
) : OnnxRuntimeSession {
    private val createFloatTensor = tensorClass.getMethod(
        "createTensor",
        environmentClass,
        FloatBuffer::class.java,
        LongArray::class.java,
    )
    private val createLongTensor = tensorClass.getMethod(
        "createTensor",
        environmentClass,
        LongBuffer::class.java,
        LongArray::class.java,
    )
    private val runMethod = nativeSession.javaClass.getMethod("run", Map::class.java)
    private val expectedInputs = contract.inputs.associateBy(OnnxTensorSpec::name)
    private var closed = false

    @Synchronized
    override fun run(inputs: Map<String, OnnxInputTensor>, deadline: Deadline): OnnxRunResult {
        if (closed) return OnnxRunResult(EngineAvailability.UNAVAILABLE)
        if (deadline.expired) return OnnxRunResult(EngineAvailability.TIMEOUT)
        val nativeInputs = LinkedHashMap<String, Any>()
        return try {
            validateInputs(inputs)
            inputs.forEach { (name, tensor) ->
                nativeInputs[name] = requireNotNull(when (tensor) {
                    is OnnxInputTensor.Float32 -> createFloatTensor.invoke(
                        null,
                        environment,
                        tensor.values.toDirectBuffer(),
                        tensor.shape,
                    )
                    is OnnxInputTensor.Int64 -> createLongTensor.invoke(
                        null,
                        environment,
                        tensor.values.toDirectBuffer(),
                        tensor.shape,
                    )
                })
            }
            val result = requireNotNull(runMethod.invoke(nativeSession, nativeInputs))
            try {
                if (deadline.expired) return OnnxRunResult(EngineAvailability.TIMEOUT)
                val outputs = contract.outputs.associateTo(LinkedHashMap()) { spec ->
                    spec.name to readFloatOutput(result, spec)
                }
                if (deadline.expired) OnnxRunResult(EngineAvailability.TIMEOUT)
                else OnnxRunResult(EngineAvailability.AVAILABLE, outputs)
            } finally {
                (result as AutoCloseable).close()
            }
        } catch (failure: Throwable) {
            val cause = failure.unwrapInvocation()
            if (cause is IllegalArgumentException) OnnxRunResult(EngineAvailability.INCOMPATIBLE)
            else OnnxRunResult(EngineAvailability.UNAVAILABLE)
        } finally {
            nativeInputs.values.forEach { runCatching { (it as AutoCloseable).close() } }
        }
    }

    private fun validateInputs(inputs: Map<String, OnnxInputTensor>) {
        require(inputs.keys == expectedInputs.keys) { "ONNX input names do not match the tensor ABI" }
        inputs.forEach { (name, tensor) ->
            val spec = requireNotNull(expectedInputs[name])
            val actualType = when (tensor) {
                is OnnxInputTensor.Float32 -> OnnxElementType.FLOAT32
                is OnnxInputTensor.Int64 -> OnnxElementType.INT64
            }
            require(actualType == spec.type) { "ONNX input $name has the wrong element type" }
            requireResolvedShape(spec.shape, tensor.shape.toList(), name)
            require(tensor.elementCount == elementCount(tensor.shape.toList())) {
                "ONNX input $name element count does not match its shape"
            }
        }
    }

    private fun readFloatOutput(result: Any, spec: OnnxTensorSpec): OnnxFloatOutput {
        require(spec.type == OnnxElementType.FLOAT32) { "Only bounded float outputs are supported" }
        @Suppress("UNCHECKED_CAST")
        val optional = result.javaClass.getMethod("get", String::class.java).invoke(result, spec.name) as Optional<Any>
        val value = optional.orElseThrow { IllegalArgumentException("Missing ONNX output ${spec.name}") }
        require(tensorClass.isInstance(value)) { "ONNX output ${spec.name} is not a tensor" }
        val info = requireNotNull(value.javaClass.getMethod("getInfo").invoke(value))
        val shape = info.javaClass.getMethod("getShape").invoke(info) as LongArray
        requireResolvedShape(spec.shape, shape.toList(), spec.name)
        val count = elementCount(shape.toList())
        require(count <= MAX_OUTPUT_ELEMENTS) { "ONNX output is unbounded" }
        val source = value.javaClass.getMethod("getFloatBuffer").invoke(value) as FloatBuffer
        require(source.remaining() == count) { "ONNX output buffer size does not match its shape" }
        val values = FloatArray(count)
        source.get(values)
        return OnnxFloatOutput(values, shape)
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        try {
            (nativeSession as AutoCloseable).close()
        } finally {
            sessionOptions.close()
        }
    }

    companion object {
        fun open(
            model: File,
            contract: OnnxModelContract,
            classLoader: ClassLoader,
            classNames: OrtClassNames,
        ): ReflectiveOrtSession {
            val environmentClass = Class.forName(classNames.environment, true, classLoader)
            val tensorClass = Class.forName(classNames.tensor, true, classLoader)
            val optionsClass = Class.forName(classNames.sessionOptions, true, classLoader)
            val environment = requireNotNull(environmentClass.getMethod("getEnvironment").invoke(null))
            val options = optionsClass.getConstructor().newInstance() as AutoCloseable
            val nativeSession = try {
                optionsClass.getMethod("setInterOpNumThreads", Int::class.javaPrimitiveType).invoke(options, 1)
                optionsClass.getMethod("setIntraOpNumThreads", Int::class.javaPrimitiveType).invoke(options, 2)
                requireNotNull(
                    environmentClass.getMethod("createSession", String::class.java, optionsClass)
                        .invoke(environment, model.absolutePath, options),
                )
            } catch (failure: Throwable) {
                runCatching { options.close() }
                throw failure
            }
            try {
                contract.requireCompatible(readMetadata(nativeSession, classNames.tensorInfo))
                return ReflectiveOrtSession(
                    environment,
                    nativeSession,
                    options,
                    contract,
                    environmentClass,
                    tensorClass,
                )
            } catch (failure: Throwable) {
                runCatching { (nativeSession as AutoCloseable).close() }
                runCatching { options.close() }
                throw failure
            }
        }

        private fun readMetadata(session: Any, tensorInfoClass: String): OnnxModelMetadata = OnnxModelMetadata(
            inputs = readNodes(session, "getInputInfo", tensorInfoClass),
            outputs = readNodes(session, "getOutputInfo", tensorInfoClass),
        )

        private fun readNodes(
            session: Any,
            method: String,
            tensorInfoClass: String,
        ): Map<String, Pair<OnnxElementType, List<Long>>> {
            @Suppress("UNCHECKED_CAST")
            val nodes = session.javaClass.getMethod(method).invoke(session) as Map<String, Any>
            return nodes.mapValues { (_, node) ->
                val info = requireNotNull(node.javaClass.getMethod("getInfo").invoke(node))
                require(info.javaClass.name == tensorInfoClass) { "ONNX value is not a tensor" }
                val shape = (info.javaClass.getMethod("getShape").invoke(info) as LongArray).toList()
                val javaType = requireNotNull(info.javaClass.getField("type").get(info)).toString()
                val type = when (javaType) {
                    "FLOAT" -> OnnxElementType.FLOAT32
                    "INT64" -> OnnxElementType.INT64
                    else -> error("Unsupported ONNX element type $javaType")
                }
                type to shape
            }
        }
    }
}

private fun requireResolvedShape(expected: List<Long>, actual: List<Long>, name: String) {
    require(expected.size == actual.size && expected.indices.all { index ->
        val expectedDimension = expected[index]
        val actualDimension = actual[index]
        actualDimension > 0 && (expectedDimension == DYNAMIC_DIMENSION || expectedDimension == actualDimension)
    }) { "ONNX tensor $name has an incompatible resolved shape" }
}

private fun elementCount(shape: List<Long>): Int {
    var count = 1L
    shape.forEach { dimension ->
        require(dimension > 0 && count <= MAX_TENSOR_ELEMENTS / dimension) { "ONNX tensor is unbounded" }
        count *= dimension
    }
    return count.toInt()
}

private fun FloatArray.toDirectBuffer(): FloatBuffer = ByteBuffer
    .allocateDirect(size * Float.SIZE_BYTES)
    .order(ByteOrder.nativeOrder())
    .asFloatBuffer()
    .apply { put(this@toDirectBuffer); flip() }

private fun LongArray.toDirectBuffer(): LongBuffer = ByteBuffer
    .allocateDirect(size * Long.SIZE_BYTES)
    .order(ByteOrder.nativeOrder())
    .asLongBuffer()
    .apply { put(this@toDirectBuffer); flip() }

private tailrec fun Throwable.unwrapInvocation(): Throwable =
    if (this is InvocationTargetException && targetException != null) targetException.unwrapInvocation() else this

private const val DYNAMIC_DIMENSION = -1L
private const val MAX_TENSOR_ELEMENTS = 1_000_000L
private const val MAX_OUTPUT_ELEMENTS = 1_000_000
