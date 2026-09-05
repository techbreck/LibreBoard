// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import java.io.File
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.charset.CodingErrorAction
import java.nio.file.StandardOpenOption

/**
 * Reads only the protobuf fields needed to enumerate operators in an ONNX model. This keeps model
 * validation independent from the inference runtime and avoids loading a model into native code
 * before its signed manifest has been checked.
 *
 * The parser intentionally covers nodes and tensors in the inference graph, graph/tensor-valued
 * attributes, sparse tensors, and model-local functions. Training graphs and external tensor data
 * are rejected: LibreBoard model packs are self-contained and inference-only.
 */
object OnnxOperatorInspector {
    private const val MODEL_GRAPH_FIELD = 7
    private const val MODEL_TRAINING_INFO_FIELD = 20
    private const val MODEL_FUNCTIONS_FIELD = 25
    private const val GRAPH_NODE_FIELD = 1
    private const val GRAPH_INITIALIZER_FIELD = 5
    private const val GRAPH_SPARSE_INITIALIZER_FIELD = 15
    private const val NODE_OPERATOR_FIELD = 4
    private const val NODE_ATTRIBUTE_FIELD = 5
    private const val NODE_DOMAIN_FIELD = 7
    private const val ATTRIBUTE_GRAPH_FIELD = 6
    private const val ATTRIBUTE_TENSOR_FIELD = 5
    private const val ATTRIBUTE_TENSORS_FIELD = 10
    private const val ATTRIBUTE_GRAPHS_FIELD = 11
    private const val ATTRIBUTE_SPARSE_TENSOR_FIELD = 22
    private const val ATTRIBUTE_SPARSE_TENSORS_FIELD = 23
    private const val FUNCTION_NODE_FIELD = 7
    private const val FUNCTION_ATTRIBUTE_FIELD = 11
    private const val TENSOR_EXTERNAL_DATA_FIELD = 13
    private const val TENSOR_DATA_LOCATION_FIELD = 14
    private const val SPARSE_TENSOR_VALUES_FIELD = 1
    private const val SPARSE_TENSOR_INDICES_FIELD = 2
    private const val LENGTH_DELIMITED = 2
    private const val MAX_NESTED_GRAPH_DEPTH = 32
    private const val MAX_NODE_COUNT = 100_000
    private const val MAX_IDENTIFIER_BYTES = 256

    fun inspect(model: File, maximumBytes: Long): Set<String> {
        require(model.isFile) { "ONNX model is missing" }
        val size = model.length()
        require(size in 1..maximumBytes) { "ONNX model size is outside allowed bounds" }
        require(size <= Int.MAX_VALUE) { "ONNX model is too large to inspect" }

        FileChannel.open(model.toPath(), StandardOpenOption.READ).use { channel ->
            val state = State()
            parseModel(ProtoReader(channel.map(FileChannel.MapMode.READ_ONLY, 0, size)), state)
            require(state.sawInferenceGraph) { "ONNX model has no inference graph" }
            require(state.operators.isNotEmpty()) { "ONNX model has no operators" }
            return state.operators.toSet()
        }
    }

    private fun parseModel(reader: ProtoReader, state: State) {
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            when (tag.fieldNumber) {
                MODEL_GRAPH_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    require(!state.sawInferenceGraph) { "ONNX model contains multiple inference graphs" }
                    state.sawInferenceGraph = true
                    parseGraph(reader.readMessage(), state, 0)
                }
                MODEL_TRAINING_INFO_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    throw IllegalArgumentException("ONNX training graphs are not permitted")
                }
                MODEL_FUNCTIONS_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseFunction(reader.readMessage(), state)
                }
                else -> reader.skip(tag.wireType)
            }
        }
    }

    private fun parseGraph(reader: ProtoReader, state: State, depth: Int) {
        require(depth <= MAX_NESTED_GRAPH_DEPTH) { "ONNX graph nesting is too deep" }
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            when (tag.fieldNumber) {
                GRAPH_NODE_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseNode(reader.readMessage(), state, depth)
                }
                GRAPH_INITIALIZER_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseTensor(reader.readMessage())
                }
                GRAPH_SPARSE_INITIALIZER_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseSparseTensor(reader.readMessage())
                }
                else -> reader.skip(tag.wireType)
            }
        }
    }

    private fun parseNode(reader: ProtoReader, state: State, depth: Int) {
        state.nodeCount += 1
        require(state.nodeCount <= MAX_NODE_COUNT) { "ONNX model contains too many nodes" }
        var operator: String? = null
        var domain = ""
        var sawDomain = false
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            when (tag.fieldNumber) {
                NODE_OPERATOR_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    require(operator == null) { "ONNX node contains duplicate operator fields" }
                    operator = reader.readUtf8(MAX_IDENTIFIER_BYTES, "operator")
                }
                NODE_DOMAIN_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    require(!sawDomain) { "ONNX node contains duplicate domain fields" }
                    sawDomain = true
                    domain = reader.readUtf8(MAX_IDENTIFIER_BYTES, "operator domain")
                }
                NODE_ATTRIBUTE_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseAttribute(reader.readMessage(), state, depth)
                }
                else -> reader.skip(tag.wireType)
            }
        }
        val name = requireNotNull(operator) { "ONNX node is missing an operator" }
        require(name.isNotBlank()) { "ONNX node has a blank operator" }
        require(domain.none(Char::isWhitespace)) { "ONNX operator domain contains whitespace" }
        require(name.none(Char::isWhitespace)) { "ONNX operator contains whitespace" }
        state.operators += canonicalOperator(domain, name)
    }

    private fun parseAttribute(reader: ProtoReader, state: State, depth: Int) {
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            when (tag.fieldNumber) {
                ATTRIBUTE_TENSOR_FIELD, ATTRIBUTE_TENSORS_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseTensor(reader.readMessage())
                }
                ATTRIBUTE_GRAPH_FIELD, ATTRIBUTE_GRAPHS_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseGraph(reader.readMessage(), state, depth + 1)
                }
                ATTRIBUTE_SPARSE_TENSOR_FIELD, ATTRIBUTE_SPARSE_TENSORS_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseSparseTensor(reader.readMessage())
                }
                else -> reader.skip(tag.wireType)
            }
        }
    }

    private fun parseTensor(reader: ProtoReader) {
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            when (tag.fieldNumber) {
                TENSOR_EXTERNAL_DATA_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    throw IllegalArgumentException("ONNX external tensor data is not permitted")
                }
                TENSOR_DATA_LOCATION_FIELD -> {
                    requireWireType(tag, 0)
                    require(reader.readScalarVarint() == 0L) {
                        "ONNX external tensor data is not permitted"
                    }
                }
                else -> reader.skip(tag.wireType)
            }
        }
    }

    private fun parseSparseTensor(reader: ProtoReader) {
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            if (tag.fieldNumber == SPARSE_TENSOR_VALUES_FIELD || tag.fieldNumber == SPARSE_TENSOR_INDICES_FIELD) {
                requireWireType(tag, LENGTH_DELIMITED)
                parseTensor(reader.readMessage())
            } else {
                reader.skip(tag.wireType)
            }
        }
    }

    private fun parseFunction(reader: ProtoReader, state: State) {
        while (reader.hasRemaining()) {
            val tag = reader.readTag()
            when (tag.fieldNumber) {
                FUNCTION_NODE_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseNode(reader.readMessage(), state, 0)
                }
                FUNCTION_ATTRIBUTE_FIELD -> {
                    requireWireType(tag, LENGTH_DELIMITED)
                    parseAttribute(reader.readMessage(), state, 0)
                }
                else -> reader.skip(tag.wireType)
            }
        }
    }

    private fun canonicalOperator(domain: String, operator: String): String =
        if (domain.isEmpty() || domain == "ai.onnx") operator else "$domain::$operator"

    private fun requireWireType(tag: Tag, expected: Int) {
        require(tag.wireType == expected) {
            "Invalid ONNX protobuf wire type for field ${tag.fieldNumber}"
        }
    }

    private class State {
        val operators = linkedSetOf<String>()
        var nodeCount = 0
        var sawInferenceGraph = false
    }

    private data class Tag(val fieldNumber: Int, val wireType: Int)

    private class ProtoReader(private val buffer: ByteBuffer) {
        fun hasRemaining(): Boolean = buffer.hasRemaining()

        fun readTag(): Tag {
            val raw = readVarint()
            val field = raw ushr 3
            val wire = (raw and 0x07).toInt()
            require(field in 1..536_870_911) { "Invalid ONNX protobuf field number" }
            return Tag(field.toInt(), wire)
        }

        fun readMessage(): ProtoReader {
            val length = readLength()
            require(length <= buffer.remaining()) { "Truncated ONNX protobuf message" }
            val nested = buffer.slice().apply { limit(length) }
            buffer.position(buffer.position() + length)
            return ProtoReader(nested)
        }

        fun readScalarVarint(): Long = readVarint()

        fun readUtf8(maximumBytes: Int, label: String): String {
            val length = readLength()
            require(length <= maximumBytes) { "ONNX $label is too long" }
            require(length <= buffer.remaining()) { "Truncated ONNX $label" }
            val bytes = buffer.slice().apply { limit(length) }
            buffer.position(buffer.position() + length)
            return Charsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(bytes)
                .toString()
        }

        fun skip(wireType: Int) {
            when (wireType) {
                0 -> readVarint()
                1 -> advance(8)
                LENGTH_DELIMITED -> advance(readLength())
                5 -> advance(4)
                else -> throw IllegalArgumentException("Unsupported ONNX protobuf wire type: $wireType")
            }
        }

        private fun advance(count: Int) {
            require(count <= buffer.remaining()) { "Truncated ONNX protobuf field" }
            buffer.position(buffer.position() + count)
        }

        private fun readLength(): Int {
            val value = readVarint()
            require(value <= Int.MAX_VALUE) { "ONNX protobuf field is too large" }
            return value.toInt()
        }

        private fun readVarint(): Long {
            var value = 0L
            for (shift in 0..63 step 7) {
                require(buffer.hasRemaining()) { "Truncated ONNX protobuf varint" }
                val byte = buffer.get().toInt() and 0xff
                if (shift == 63) require(byte and 0xfe == 0) { "Invalid ONNX protobuf varint" }
                value = value or ((byte and 0x7f).toLong() shl shift)
                if (byte and 0x80 == 0) return value
            }
            throw IllegalArgumentException("Invalid ONNX protobuf varint")
        }
    }
}
