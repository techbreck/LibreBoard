// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.MessageDigest
import java.security.Signature
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.test.assertFailsWith

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class ModelRegistryTest {
    private lateinit var keyPair: KeyPair
    private lateinit var registry: ModelRegistry

    @Before
    fun setUp() {
        keyPair = KeyPairGenerator.getInstance("RSA").apply { initialize(3072) }.generateKeyPair()
        registry = ModelRegistry(
            ApplicationProvider.getApplicationContext<Context>(),
            ModelValidationLimits(
                allowedOperators = setOf("MatMul", "Add", "If", "com.microsoft::MatMulNBits"),
                acceptedModelKinds = setOf(ModelKind.CONTEXT_RESCORER),
                appVersionCode = 1,
            ),
            keyPair.public,
        )
        registry.wipe()
    }

    @After
    fun tearDown() = registry.wipe()

    @Test
    fun activatesOnlyHashedSignedDataAndRollsBackCorruptActiveModel() {
        val first = model("MatMul", marker = "first")
        val second = model("MatMul", marker = "second")
        registry.activate(ByteArrayInputStream(archive(first)))
        registry.activate(ByteArrayInputStream(archive(second)))
        registry.activeModel()!!.model.writeBytes("corrupt".encodeToByteArray())

        val recovered = registry.activeModel()
        assertArrayEquals(first, recovered!!.model.readBytes())
        assertEquals(listOf("en-US", "de"), recovered.manifest.locales)
    }

    @Test
    fun rejectsTamperedModelWithoutReplacingActiveModel() {
        val original = model("MatMul", marker = "trusted")
        registry.activate(ByteArrayInputStream(archive(original)))
        val signed = model("MatMul", marker = "signed")
        val tampered = archive(signed, modelOverride = model("Add", marker = "different"))
        assertFailsWith<IllegalArgumentException> { registry.activate(ByteArrayInputStream(tampered)) }
        assertArrayEquals(original, registry.activeModel()!!.model.readBytes())
    }

    @Test
    fun rejectsUndeclaredNestedOperatorWithoutReplacingActiveModel() {
        val original = model("MatMul", marker = "trusted")
        registry.activate(ByteArrayInputStream(archive(original)))
        val hiddenAdd = OnnxTestModels.model(
            nodes = listOf(
                OnnxTestModels.node(
                    "If",
                    attributes = listOf(
                        OnnxTestModels.graphAttribute(listOf(OnnxTestModels.node("Add"))),
                    ),
                ),
            ),
        )

        val failure = assertFailsWith<IllegalArgumentException> {
            registry.activate(ByteArrayInputStream(archive(hiddenAdd, operators = listOf("If"))))
        }
        assertEquals("ONNX operator manifest mismatch; undeclared: Add", failure.message)
        assertArrayEquals(original, registry.activeModel()!!.model.readBytes())
    }

    @Test
    fun rejectsCustomDomainMasqueradingAsCoreOperator() {
        val custom = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node("MatMul", domain = "untrusted.example")),
        )

        val failure = assertFailsWith<IllegalArgumentException> {
            registry.activate(ByteArrayInputStream(archive(custom, operators = listOf("MatMul"))))
        }
        assertEquals(
            "ONNX operator manifest mismatch; undeclared: untrusted.example::MatMul; absent: MatMul",
            failure.message,
        )
    }

    @Test
    fun rejectsDeclaredOperatorThatIsAbsentFromGraph() {
        val model = model("MatMul")
        val failure = assertFailsWith<IllegalArgumentException> {
            registry.activate(ByteArrayInputStream(archive(model, operators = listOf("MatMul", "Add"))))
        }
        assertEquals("ONNX operator manifest mismatch; absent: Add", failure.message)
    }

    @Test
    fun rejectsMalformedSignedModel() {
        val malformed = OnnxTestModels.truncatedMessage()
        assertFailsWith<IllegalArgumentException> {
            registry.activate(ByteArrayInputStream(archive(malformed)))
        }
    }

    @Test
    fun rejectsSignedModelWithExternalTensorReference() {
        val external = OnnxTestModels.model(
            nodes = listOf(OnnxTestModels.node("MatMul")),
            initializers = listOf(OnnxTestModels.externalTensor()),
        )
        val failure = assertFailsWith<IllegalArgumentException> {
            registry.activate(ByteArrayInputStream(archive(external)))
        }
        assertEquals("ONNX external tensor data is not permitted", failure.message)
    }

    @Test
    fun modelKindsUseIndependentActiveAndRollbackSlots() {
        val swipeRegistry = ModelRegistry(
            ApplicationProvider.getApplicationContext(),
            ModelValidationLimits(
                maximumBytes = 3L * 1024 * 1024,
                allowedOperators = setOf("MatMul"),
                acceptedModelKinds = setOf(ModelKind.SWIPE_CTC),
                appVersionCode = 1,
            ),
            keyPair.public,
        )
        swipeRegistry.wipe()
        val contextModel = model("MatMul", marker = "context")
        val swipeModel = model("MatMul", marker = "swipe")
        registry.activate(ByteArrayInputStream(archive(contextModel)))
        swipeRegistry.activate(ByteArrayInputStream(archive(
            swipeModel,
            kind = ModelKind.SWIPE_CTC,
        )))

        assertArrayEquals(contextModel, registry.activeModel()!!.model.readBytes())
        assertArrayEquals(swipeModel, swipeRegistry.activeModel()!!.model.readBytes())
        swipeRegistry.wipe()
        assertNull(swipeRegistry.activeModel())
        assertArrayEquals(contextModel, registry.activeModel()!!.model.readBytes())
    }

    @Test
    fun registryRejectsAmbiguousModelKindOwnership() {
        assertFailsWith<IllegalArgumentException> {
            ModelRegistry(
                ApplicationProvider.getApplicationContext(),
                ModelValidationLimits(
                    allowedOperators = setOf("MatMul"),
                    acceptedModelKinds = setOf(ModelKind.SWIPE_CTC, ModelKind.CONTEXT_RESCORER),
                    appVersionCode = 1,
                ),
                keyPair.public,
            )
        }
    }

    private fun archive(
        signedModel: ByteArray,
        modelOverride: ByteArray = signedModel,
        operators: List<String> = listOf("MatMul"),
        kind: ModelKind = ModelKind.CONTEXT_RESCORER,
    ): ByteArray {
        val tokenizer = "{}".encodeToByteArray().takeIf { kind == ModelKind.CONTEXT_RESCORER }
        val manifest = ModelManifest(
            schemaVersion = 1,
            engineAbi = 1,
            modelKind = kind,
            tensorAbi = when (kind) {
                ModelKind.SWIPE_CTC -> "swipe-latin-v1"
                ModelKind.CONTEXT_RESCORER -> "context-en-de-v1"
            },
            locales = listOf("en-US", "de"),
            architecture = "fixture",
            parameterCount = 1,
            quantization = "fixture",
            modelSha256 = sha256(signedModel),
            tokenizerSha256 = tokenizer?.let(::sha256),
            requiredOnnxOperators = operators,
            license = "Apache-2.0",
            provenance = listOf(ProvenanceEntry("fixture", "1", "Apache-2.0", "https://example.invalid/model")),
            minimumAppVersionCode = 1,
        )
        val manifestBytes = Json.encodeToString(manifest).encodeToByteArray()
        val signature = Signature.getInstance("SHA256withRSA").run {
            initSign(keyPair.private)
            update(manifestBytes)
            sign()
        }
        return ByteArrayOutputStream().also { output ->
            ZipOutputStream(output).use { zip ->
                buildList {
                    add("manifest.json" to manifestBytes)
                    add("model.onnx" to modelOverride)
                    tokenizer?.let { add("tokenizer.json" to it) }
                    add("signature.der" to signature)
                }.forEach { (name, bytes) ->
                    zip.putNextEntry(ZipEntry(name))
                    zip.write(bytes)
                    zip.closeEntry()
                }
            }
        }.toByteArray()
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }

    private fun model(operator: String, marker: String = "fixture") = OnnxTestModels.model(
        nodes = listOf(OnnxTestModels.node(operator)),
        marker = marker,
    )
}
