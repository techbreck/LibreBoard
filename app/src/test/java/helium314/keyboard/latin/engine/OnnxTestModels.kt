// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

internal object OnnxTestModels {
    fun model(
        nodes: List<ByteArray>,
        functions: List<ByteArray> = emptyList(),
        marker: String = "fixture",
        trainingInfo: ByteArray? = null,
        initializers: List<ByteArray> = emptyList(),
        sparseInitializers: List<ByteArray> = emptyList(),
    ): ByteArray = concat(
        varintField(1, 9),
        messageField(2, marker.encodeToByteArray()),
        messageField(7, graph(nodes, initializers, sparseInitializers)),
        messageField(8, varintField(2, 18)),
        *functions.map { messageField(25, it) }.toTypedArray(),
        trainingInfo?.let { messageField(20, it) } ?: byteArrayOf(),
    )

    fun node(
        operator: String,
        domain: String = "",
        attributes: List<ByteArray> = emptyList(),
    ): ByteArray = concat(
        messageField(4, operator.encodeToByteArray()),
        *attributes.map { messageField(5, it) }.toTypedArray(),
        if (domain.isEmpty()) byteArrayOf() else messageField(7, domain.encodeToByteArray()),
    )

    fun graph(
        nodes: List<ByteArray>,
        initializers: List<ByteArray> = emptyList(),
        sparseInitializers: List<ByteArray> = emptyList(),
    ): ByteArray = concat(
        *nodes.map { messageField(1, it) }.toTypedArray(),
        *initializers.map { messageField(5, it) }.toTypedArray(),
        *sparseInitializers.map { messageField(15, it) }.toTypedArray(),
    )

    fun graphAttribute(nodes: List<ByteArray>): ByteArray = messageField(6, graph(nodes))

    fun graphsAttribute(vararg graphs: List<ByteArray>): ByteArray =
        concat(*graphs.map { messageField(11, graph(it)) }.toTypedArray())

    fun tensorAttribute(tensor: ByteArray, repeated: Boolean = false): ByteArray =
        messageField(if (repeated) 10 else 5, tensor)

    fun sparseTensorAttribute(sparseTensor: ByteArray, repeated: Boolean = false): ByteArray =
        messageField(if (repeated) 23 else 22, sparseTensor)

    fun externalTensor(useLocationOnly: Boolean = false): ByteArray = if (useLocationOnly) {
        varintField(14, 1)
    } else {
        concat(
            messageField(13, concat(
                messageField(1, "location".encodeToByteArray()),
                messageField(2, "../outside.bin".encodeToByteArray()),
            )),
            varintField(14, 1),
        )
    }

    fun sparseTensor(values: ByteArray, indices: ByteArray = byteArrayOf()): ByteArray = concat(
        messageField(1, values),
        messageField(2, indices),
    )

    fun function(nodes: List<ByteArray>, attributes: List<ByteArray> = emptyList()): ByteArray = concat(
        messageField(1, "fixture_function".encodeToByteArray()),
        *nodes.map { messageField(7, it) }.toTypedArray(),
        *attributes.map { messageField(11, it) }.toTypedArray(),
    )

    fun truncatedMessage(): ByteArray = byteArrayOf((7 shl 3 or 2).toByte(), 10, 1)

    private fun varintField(field: Int, value: Long): ByteArray =
        concat(varint((field shl 3).toLong()), varint(value))

    private fun messageField(field: Int, value: ByteArray): ByteArray =
        concat(varint((field shl 3 or 2).toLong()), varint(value.size.toLong()), value)

    private fun varint(input: Long): ByteArray {
        var value = input
        val bytes = mutableListOf<Byte>()
        do {
            var next = (value and 0x7f).toInt()
            value = value ushr 7
            if (value != 0L) next = next or 0x80
            bytes += next.toByte()
        } while (value != 0L)
        return bytes.toByteArray()
    }

    private fun concat(vararg values: ByteArray): ByteArray {
        val output = ByteArray(values.sumOf(ByteArray::size))
        var offset = 0
        values.forEach { value ->
            value.copyInto(output, offset)
            offset += value.size
        }
        return output
    }
}
