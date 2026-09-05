// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Context
import android.net.Uri
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import java.io.InputStream
import java.util.concurrent.CancellationException

sealed interface OfficialModelPackImportResult {
    data class Installed(val manifest: ModelManifest) : OfficialModelPackImportResult
    data object Locked : OfficialModelPackImportResult
    data object Unavailable : OfficialModelPackImportResult
    data object Rejected : OfficialModelPackImportResult
}

/**
 * Copies the optional official en/de model pack through its one fixed data URI. The provider is not
 * trusted: [ModelRegistry] still verifies the signed archive before anything becomes active.
 * Callers must invoke this off the UI thread because a cross-process provider may be slow.
 */
class OfficialContextModelPackImporter private constructor(
    private val credentialStorageAvailable: () -> Boolean,
    private val openArchive: () -> InputStream?,
    private val activate: (InputStream) -> ModelRegistry.ActiveModel,
) {
    constructor(context: Context, registry: ModelRegistry) : this(
        credentialStorageAvailable = { CredentialEncryptedStorage.contextOrNull(context) != null },
        openArchive = { context.contentResolver.openInputStream(MODEL_URI) },
        activate = registry::activate,
    )

    fun installFromOfficialPack(): OfficialModelPackImportResult {
        if (!credentialStorageAvailable()) return OfficialModelPackImportResult.Locked
        val archive = try {
            openArchive()
        } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            return OfficialModelPackImportResult.Unavailable
        } ?: return OfficialModelPackImportResult.Unavailable

        val active = try {
            activate(archive)
        } catch (failure: Exception) {
            if (failure is CancellationException) throw failure
            return OfficialModelPackImportResult.Rejected
        } finally {
            // A provider-side close failure cannot undo an already atomic ModelRegistry activation.
            runCatching { archive.close() }
        }
        return if (active.manifest.modelKind == ModelKind.CONTEXT_RESCORER) {
            OfficialModelPackImportResult.Installed(active.manifest)
        } else {
            OfficialModelPackImportResult.Rejected
        }
    }

    internal constructor(
        credentialStorageAvailable: Boolean,
        openArchive: () -> InputStream?,
        activate: (InputStream) -> ModelRegistry.ActiveModel,
    ) : this({ credentialStorageAvailable }, openArchive, activate)

    companion object {
        const val MODEL_PACK_AUTHORITY = "org.libreboard.model.en_de"
        val MODEL_URI: Uri = Uri.parse("content://$MODEL_PACK_AUTHORITY/model.lbmodel")
    }
}
