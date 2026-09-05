// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Context
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.security.MessageDigest
import java.security.PublicKey
import java.security.Signature
import java.util.UUID
import java.util.zip.ZipInputStream

/**
 * Installs data-only .lbmodel archives into credential-encrypted storage. It deliberately has no
 * downloader or executable-plugin path. The release layer must inject the pinned project key.
 */
class ModelRegistry(
    context: Context,
    private val limits: ModelValidationLimits,
    private val trustedProjectKey: PublicKey,
) {
    private val privateContext = CredentialEncryptedStorage.contextOrNull(context)
        ?: throw IllegalStateException("Models are unavailable before first unlock")
    private val root = File(privateContext.filesDir, "models")
    private val json = Json { ignoreUnknownKeys = false }

    data class ActiveModel(val manifest: ModelManifest, val model: File, val tokenizer: File?)

    @Synchronized
    fun activate(archive: InputStream): ActiveModel {
        root.mkdirs()
        val staging = File(root, ".staging-${UUID.randomUUID()}")
        require(staging.mkdir()) { "Could not create model staging directory" }
        try {
            var manifestBytes: ByteArray? = null
            var signatureBytes: ByteArray? = null
            var modelHash: String? = null
            var modelSize = 0L
            var tokenizerHash: String? = null
            val seen = mutableSetOf<String>()

            ZipInputStream(archive.buffered()).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    require(!entry.isDirectory && entry.name in ALLOWED_ENTRIES) { "Unexpected model entry" }
                    require(seen.add(entry.name)) { "Duplicate model entry" }
                    when (entry.name) {
                        MANIFEST -> manifestBytes = readBounded(zip, MAX_MANIFEST_BYTES)
                        SIGNATURE -> signatureBytes = readBounded(zip, MAX_SIGNATURE_BYTES)
                        MODEL -> {
                            val copied = copyHashed(zip, File(staging, MODEL), limits.maximumBytes)
                            modelSize = copied.first
                            modelHash = copied.second
                        }
                        TOKENIZER -> tokenizerHash = copyHashed(zip, File(staging, TOKENIZER), MAX_TOKENIZER_BYTES).second
                    }
                    zip.closeEntry()
                }
            }

            val rawManifest = requireNotNull(manifestBytes) { "Missing model manifest" }
            val rawSignature = requireNotNull(signatureBytes) { "Missing model signature" }
            val manifest = json.decodeFromString<ModelManifest>(rawManifest.decodeToString())
            val validation = ModelManifestValidator.validate(manifest, modelSize, limits)
            require(validation is ModelValidationResult.Valid) {
                (validation as ModelValidationResult.Invalid).reason
            }
            require(modelHash == manifest.modelSha256) { "Model hash mismatch" }
            require(manifest.tokenizerSha256 == tokenizerHash) { "Tokenizer hash mismatch" }
            require(verifySignature(rawManifest, rawSignature)) { "Untrusted model signature" }

            File(staging, MANIFEST).writeBytes(rawManifest)
            File(staging, SIGNATURE).writeBytes(rawSignature)
            activateStaging(staging)
            return requireNotNull(activeModel()) { "Activated model failed verification" }
        } catch (failure: Throwable) {
            staging.deleteRecursively()
            throw failure
        }
    }

    /** Revalidates the on-disk hashes before returning files to an inference runtime. */
    @Synchronized
    fun activeModel(): ActiveModel? {
        val active = File(root, ACTIVE)
        validatedModel(active)?.let { return it }
        val previous = File(root, PREVIOUS)
        if (validatedModel(previous) != null) {
            active.deleteRecursively()
            if (previous.renameTo(active)) return validatedModel(active)
        }
        return null
    }

    private fun validatedModel(directory: File): ActiveModel? {
        val manifestFile = File(directory, MANIFEST)
        val modelFile = File(directory, MODEL)
        val signatureFile = File(directory, SIGNATURE)
        if (!manifestFile.isFile || !modelFile.isFile || !signatureFile.isFile) return null
        return runCatching {
            val rawManifest = manifestFile.inputStream().use { readBounded(it, MAX_MANIFEST_BYTES) }
            val manifest = json.decodeFromString<ModelManifest>(rawManifest.decodeToString())
            require(ModelManifestValidator.validate(manifest, modelFile.length(), limits) is ModelValidationResult.Valid)
            require(sha256(modelFile) == manifest.modelSha256)
            val tokenizer = File(directory, TOKENIZER).takeIf(File::isFile)
            require(tokenizer?.let(::sha256) == manifest.tokenizerSha256)
            val signature = signatureFile.inputStream().use { readBounded(it, MAX_SIGNATURE_BYTES) }
            require(verifySignature(rawManifest, signature))
            ActiveModel(manifest, modelFile, tokenizer)
        }.getOrNull()
    }

    @Synchronized
    fun wipe() {
        root.deleteRecursively()
    }

    private fun activateStaging(staging: File) {
        val active = File(root, ACTIVE)
        val previous = File(root, PREVIOUS)
        previous.deleteRecursively()
        if (active.exists()) require(active.renameTo(previous)) { "Could not retain previous model" }
        if (!staging.renameTo(active)) {
            if (previous.exists()) previous.renameTo(active)
            error("Could not atomically activate model")
        }
    }

    private fun verifySignature(manifest: ByteArray, signature: ByteArray): Boolean = runCatching {
        Signature.getInstance(SIGNATURE_ALGORITHM).run {
            initVerify(trustedProjectKey)
            update(manifest)
            verify(signature)
        }
    }.getOrDefault(false)

    private fun copyHashed(input: InputStream, destination: File, maximum: Long): Pair<Long, String> {
        val digest = MessageDigest.getInstance("SHA-256")
        var total = 0L
        FileOutputStream(destination).use { output ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                total += count
                require(total <= maximum) { "Model entry is too large" }
                digest.update(buffer, 0, count)
                output.write(buffer, 0, count)
            }
            output.fd.sync()
        }
        require(total > 0) { "Model entry is empty" }
        return total to digest.digest().toHex()
    }

    private fun readBounded(input: InputStream, maximum: Int): ByteArray {
        val output = java.io.ByteArrayOutputStream(minOf(maximum, 8192))
        val buffer = ByteArray(8192)
        var total = 0
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            total += count
            require(total <= maximum) { "Model metadata entry is too large" }
            output.write(buffer, 0, count)
        }
        return output.toByteArray()
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex() = joinToString("") { "%02x".format(it) }

    companion object {
        private const val MANIFEST = "manifest.json"
        private const val MODEL = "model.onnx"
        private const val TOKENIZER = "tokenizer.json"
        private const val SIGNATURE = "signature.der"
        private const val ACTIVE = "active"
        private const val PREVIOUS = "previous"
        private const val SIGNATURE_ALGORITHM = "SHA256withECDSA"
        private const val MAX_MANIFEST_BYTES = 256 * 1024
        private const val MAX_SIGNATURE_BYTES = 16 * 1024
        private const val MAX_TOKENIZER_BYTES = 2L * 1024 * 1024
        private val ALLOWED_ENTRIES = setOf(MANIFEST, MODEL, TOKENIZER, SIGNATURE)
    }
}
