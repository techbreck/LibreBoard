// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Files
import kotlin.test.assertFailsWith

class OnnxOperatorInspectorTest {
    @Test
    fun enumeratesTopLevelNestedAndFunctionOperatorsWithDomains() {
        val nested = OnnxTestModels.graphAttribute(
            listOf(OnnxTestModels.node("Add", domain = "ai.onnx")),
        )
        val repeatedGraphs = OnnxTestModels.graphsAttribute(
            listOf(OnnxTestModels.node("Where")),
            listOf(OnnxTestModels.node("Identity")),
        )
        val function = OnnxTestModels.function(
            nodes = listOf(OnnxTestModels.node("MatMulNBits", domain = "com.microsoft")),
            attributes = listOf(
                OnnxTestModels.graphAttribute(listOf(OnnxTestModels.node("Softmax"))),
            ),
        )
        val model = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node("If", attributes = listOf(nested, repeatedGraphs))),
            functions = listOf(function),
        )

        assertEquals(
            setOf("If", "Add", "Where", "Identity", "com.microsoft::MatMulNBits", "Softmax"),
            inspect(model),
        )
    }

    @Test
    fun rejectsMalformedAndTrainingModels() {
        assertFailsWith<IllegalArgumentException> { inspect(OnnxTestModels.truncatedMessage()) }
        val trainingModel = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node("MatMul")),
            trainingInfo = byteArrayOf(),
        )
        assertFailsWith<IllegalArgumentException> { inspect(trainingModel) }
    }

    @Test
    fun rejectsNodeWithoutOperator() {
        val model = OnnxTestModels.model(nodes = listOf(byteArrayOf()))
        assertFailsWith<IllegalArgumentException> { inspect(model) }
    }

    @Test
    fun rejectsDuplicateDomainEvenWhenFirstValueIsEmpty() {
        val duplicateDomainNode = byteArrayOf(
            0x22, 0x06, 'M'.code.toByte(), 'a'.code.toByte(), 't'.code.toByte(),
            'M'.code.toByte(), 'u'.code.toByte(), 'l'.code.toByte(),
            0x3a, 0x00,
            0x3a, 0x00,
        )
        val model = OnnxTestModels.model(nodes = listOf(duplicateDomainNode))
        assertFailsWith<IllegalArgumentException> { inspect(model) }
    }

    @Test
    fun rejectsExternalTopLevelAndAttributeTensors() {
        val topLevel = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node("MatMul")),
            initializers = listOf(OnnxTestModels.externalTensor()),
        )
        val attribute = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node(
                "Constant",
                attributes = listOf(OnnxTestModels.tensorAttribute(OnnxTestModels.externalTensor())),
            )),
        )
        val repeatedAttribute = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node(
                "Custom",
                attributes = listOf(OnnxTestModels.tensorAttribute(
                    OnnxTestModels.externalTensor(useLocationOnly = true),
                    repeated = true,
                )),
            )),
        )

        listOf(topLevel, attribute, repeatedAttribute).forEach { model ->
            val failure = assertFailsWith<IllegalArgumentException> { inspect(model) }
            assertEquals("ONNX external tensor data is not permitted", failure.message)
        }
    }

    @Test
    fun rejectsExternalSparseTensorValuesAndIndices() {
        val sparseInitializer = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node("Add")),
            sparseInitializers = listOf(
                OnnxTestModels.sparseTensor(OnnxTestModels.externalTensor()),
            ),
        )
        val sparseAttribute = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node(
                "Custom",
                attributes = listOf(OnnxTestModels.sparseTensorAttribute(
                    OnnxTestModels.sparseTensor(
                        values = byteArrayOf(),
                        indices = OnnxTestModels.externalTensor(useLocationOnly = true),
                    ),
                    repeated = true,
                )),
            )),
        )

        listOf(sparseInitializer, sparseAttribute).forEach { model ->
            val failure = assertFailsWith<IllegalArgumentException> { inspect(model) }
            assertEquals("ONNX external tensor data is not permitted", failure.message)
        }
    }

    private fun inspect(bytes: ByteArray): Set<String> {
        val file = Files.createTempFile("libreboard-onnx", ".onnx").toFile()
        return try {
            file.writeBytes(bytes)
            OnnxOperatorInspector.inspect(file, 1024 * 1024)
        } finally {
            file.delete()
        }
    }
}
