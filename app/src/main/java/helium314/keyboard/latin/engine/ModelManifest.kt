// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ModelManifest(
    val schemaVersion: Int,
    val engineAbi: Int,
    val locales: List<String>,
    val architecture: String,
    val parameterCount: Long,
    val quantization: String,
    val modelSha256: String,
    val tokenizerSha256: String? = null,
    val requiredOnnxOperators: List<String>,
    val license: String,
    val provenance: List<ProvenanceEntry>,
    val minimumAppVersionCode: Int,
)

@Serializable
data class ProvenanceEntry(
    val name: String,
    val revision: String,
    val license: String,
    @SerialName("source_url") val sourceUrl: String,
)

data class ModelValidationLimits(
    val supportedSchemaVersion: Int = 1,
    val supportedEngineAbi: Int = 1,
    val maximumBytes: Long = 24L * 1024 * 1024,
    val allowedOperators: Set<String>,
    val appVersionCode: Int,
    val maximumParameterCount: Long = 50_000_000,
    val allowedModelLicenses: Set<String> = setOf("Apache-2.0"),
)

sealed interface ModelValidationResult {
    data object Valid : ModelValidationResult
    data class Invalid(val reason: String) : ModelValidationResult
}

object ModelManifestValidator {
    private val sha256 = Regex("^[a-f0-9]{64}$")

    fun validate(manifest: ModelManifest, modelBytes: Long, limits: ModelValidationLimits): ModelValidationResult {
        if (manifest.schemaVersion != limits.supportedSchemaVersion) return invalid("unsupported schema")
        if (manifest.engineAbi != limits.supportedEngineAbi) return invalid("incompatible engine ABI")
        if (manifest.minimumAppVersionCode > limits.appVersionCode) return invalid("app is too old")
        if (manifest.architecture.isBlank() || manifest.architecture.length > 128) return invalid("invalid architecture")
        if (manifest.parameterCount <= 0 || manifest.parameterCount > limits.maximumParameterCount) return invalid("parameter count is outside allowed bounds")
        if (manifest.quantization.isBlank() || manifest.quantization.length > 32) return invalid("invalid quantization")
        if (modelBytes <= 0 || modelBytes > limits.maximumBytes) return invalid("model size is outside allowed bounds")
        if (!sha256.matches(manifest.modelSha256)) return invalid("invalid model hash")
        if (manifest.tokenizerSha256 != null && !sha256.matches(manifest.tokenizerSha256)) return invalid("invalid tokenizer hash")
        if (manifest.requiredOnnxOperators.isEmpty()
            || manifest.requiredOnnxOperators.size > 256
            || manifest.requiredOnnxOperators.distinct().size != manifest.requiredOnnxOperators.size
            || manifest.requiredOnnxOperators.any {
                it.isBlank() || it.length > 256 || it.any(Char::isWhitespace)
            }) return invalid("invalid ONNX operator list")
        val forbidden = manifest.requiredOnnxOperators.toSet() - limits.allowedOperators
        if (forbidden.isNotEmpty()) return invalid("unsupported ONNX operators: ${forbidden.sorted().joinToString()}")
        if (manifest.locales.isEmpty() || manifest.locales.size > 16
            || manifest.locales.distinct().size != manifest.locales.size
            || manifest.locales.any { it.isBlank() || it.length > 64 }) return invalid("invalid model locales")
        if (manifest.license.isBlank() || manifest.provenance.isEmpty()) return invalid("license provenance is incomplete")
        if (manifest.license !in limits.allowedModelLicenses) return invalid("model license is not allowlisted")
        if (manifest.provenance.size > 64 || manifest.provenance.any {
                it.name.isBlank() || it.revision.isBlank() || it.license.isBlank()
                    || !it.sourceUrl.startsWith("https://")
            }) return invalid("invalid model provenance")
        return ModelValidationResult.Valid
    }

    private fun invalid(reason: String) = ModelValidationResult.Invalid(reason)
}
