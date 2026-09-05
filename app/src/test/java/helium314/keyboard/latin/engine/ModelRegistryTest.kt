// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
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
        keyPair = KeyPairGenerator.getInstance("EC").apply { initialize(256) }.generateKeyPair()
        registry = ModelRegistry(
            ApplicationProvider.getApplicationContext<Context>(),
            ModelValidationLimits(allowedOperators = setOf("MatMul"), appVersionCode = 1),
            keyPair.public,
        )
        registry.wipe()
    }

    @After
    fun tearDown() = registry.wipe()

    @Test
    fun activatesOnlyHashedSignedDataAndRollsBackCorruptActiveModel() {
        val first = "first model".encodeToByteArray()
        val second = "second model".encodeToByteArray()
        registry.activate(ByteArrayInputStream(archive(first)))
        registry.activate(ByteArrayInputStream(archive(second)))
        registry.activeModel()!!.model.writeBytes("corrupt".encodeToByteArray())

        val recovered = registry.activeModel()
        assertArrayEquals(first, recovered!!.model.readBytes())
        assertEquals(listOf("en-US", "de"), recovered.manifest.locales)
    }

    @Test
    fun rejectsTamperedModelWithoutReplacingActiveModel() {
        val original = "trusted".encodeToByteArray()
        registry.activate(ByteArrayInputStream(archive(original)))
        val tampered = archive("signed bytes".encodeToByteArray(), modelOverride = "different".encodeToByteArray())
        assertFailsWith<IllegalArgumentException> { registry.activate(ByteArrayInputStream(tampered)) }
        assertArrayEquals(original, registry.activeModel()!!.model.readBytes())
    }

    private fun archive(signedModel: ByteArray, modelOverride: ByteArray = signedModel): ByteArray {
        val tokenizer = "{}".encodeToByteArray()
        val manifest = ModelManifest(
            schemaVersion = 1,
            engineAbi = 1,
            locales = listOf("en-US", "de"),
            architecture = "fixture",
            parameterCount = 1,
            quantization = "fixture",
            modelSha256 = sha256(signedModel),
            tokenizerSha256 = sha256(tokenizer),
            requiredOnnxOperators = listOf("MatMul"),
            license = "Apache-2.0",
            provenance = listOf(ProvenanceEntry("fixture", "1", "Apache-2.0", "https://example.invalid/model")),
            minimumAppVersionCode = 1,
        )
        val manifestBytes = Json.encodeToString(manifest).encodeToByteArray()
        val signature = Signature.getInstance("SHA256withECDSA").run {
            initSign(keyPair.private)
            update(manifestBytes)
            sign()
        }
        return ByteArrayOutputStream().also { output ->
            ZipOutputStream(output).use { zip ->
                listOf(
                    "manifest.json" to manifestBytes,
                    "model.onnx" to modelOverride,
                    "tokenizer.json" to tokenizer,
                    "signature.der" to signature,
                ).forEach { (name, bytes) ->
                    zip.putNextEntry(ZipEntry(name))
                    zip.write(bytes)
                    zip.closeEntry()
                }
            }
        }.toByteArray()
    }

    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256")
        .digest(bytes).joinToString("") { "%02x".format(it) }
}
