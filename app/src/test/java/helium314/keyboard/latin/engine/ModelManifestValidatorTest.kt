// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelManifestValidatorTest {
    private val manifest = ModelManifest(
        schemaVersion = 1,
        engineAbi = 1,
        modelKind = ModelKind.CONTEXT_RESCORER,
        tensorAbi = "context-en-de-v1",
        locales = listOf("en-US", "de"),
        architecture = "decoder-transformer",
        parameterCount = 35_700_000,
        quantization = "INT4",
        modelSha256 = "a".repeat(64),
        tokenizerSha256 = "b".repeat(64),
        requiredOnnxOperators = listOf("MatMul", "Softmax"),
        license = "Apache-2.0",
        provenance = listOf(ProvenanceEntry("fixture", "1", "Apache-2.0", "https://example.invalid")),
        minimumAppVersionCode = 1,
    )
    private val limits = ModelValidationLimits(
        allowedOperators = setOf("MatMul", "Softmax"),
        acceptedModelKinds = setOf(ModelKind.CONTEXT_RESCORER),
        appVersionCode = 1,
    )

    @Test
    fun acceptsCompatibleBoundedDataModel() {
        assertEquals(ModelValidationResult.Valid, ModelManifestValidator.validate(manifest, 1_000, limits))
    }

    @Test
    fun rejectsUnknownOperatorAndOversizePayload() {
        val unknown = manifest.copy(requiredOnnxOperators = listOf("CustomSecretOp"))
        assertTrue(ModelManifestValidator.validate(unknown, 1_000, limits) is ModelValidationResult.Invalid)
        assertTrue(ModelManifestValidator.validate(manifest, limits.maximumBytes + 1, limits) is ModelValidationResult.Invalid)
    }

    @Test
    fun rejectsCrossPurposeModelAndMissingRequiredTokenizer() {
        val wrongKind = manifest.copy(modelKind = ModelKind.SWIPE_CTC, tensorAbi = "swipe-latin-v1")
        assertEquals(
            ModelValidationResult.Invalid("model kind is not accepted"),
            ModelManifestValidator.validate(wrongKind, 1_000, limits),
        )
        val wrongTensorAbi = manifest.copy(tensorAbi = "swipe-latin-v1")
        assertEquals(
            ModelValidationResult.Invalid("unsupported model tensor ABI"),
            ModelManifestValidator.validate(wrongTensorAbi, 1_000, limits),
        )
        assertEquals(
            ModelValidationResult.Invalid("model tokenizer is required"),
            ModelManifestValidator.validate(manifest.copy(tokenizerSha256 = null), 1_000, limits),
        )
    }

    @Test
    fun rejectsEmptyDuplicateAndMalformedOperatorLists() {
        listOf(
            emptyList(),
            listOf("MatMul", "MatMul"),
            listOf(""),
            listOf("bad operator"),
        ).forEach { operators ->
            assertTrue(
                ModelManifestValidator.validate(
                    manifest.copy(requiredOnnxOperators = operators),
                    1_000,
                    limits,
                ) is ModelValidationResult.Invalid,
            )
        }
    }
}
