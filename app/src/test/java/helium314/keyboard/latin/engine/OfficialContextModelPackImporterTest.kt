// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayInputStream
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class OfficialContextModelPackImporterTest {
    @Test
    fun usesOneFixedOfficialContentUri() {
        assertEquals(
            "content://org.libreboard.model.en_de/model.lbmodel",
            OfficialContextModelPackImporter.MODEL_URI.toString(),
        )
    }

    @Test
    fun refusesProviderAccessBeforeCredentialStorageUnlock() {
        var opened = false
        var activated = false
        val importer = OfficialContextModelPackImporter(
            credentialStorageAvailable = false,
            openArchive = { opened = true; ByteArrayInputStream(byteArrayOf(1)) },
            activate = { activated = true; activeModel(ModelKind.CONTEXT_RESCORER) },
        )

        assertSame(OfficialModelPackImportResult.Locked, importer.installFromOfficialPack())
        assertFalse(opened)
        assertFalse(activated)
    }

    @Test
    fun closesArchiveAndAcceptsOnlyContextModel() {
        val archive = CloseTrackingInputStream(byteArrayOf(1, 2, 3))
        val manifest = activeModel(ModelKind.CONTEXT_RESCORER).manifest
        val importer = OfficialContextModelPackImporter(
            credentialStorageAvailable = true,
            openArchive = { archive },
            activate = { input ->
                assertSame(archive, input)
                ModelRegistry.ActiveModel(manifest, File("model"), File("tokenizer"))
            },
        )

        assertEquals(
            OfficialModelPackImportResult.Installed(manifest),
            importer.installFromOfficialPack(),
        )
        assertTrue(archive.closed)
    }

    @Test
    fun missingProviderIsAUsableFallbackRatherThanAnError() {
        val importer = OfficialContextModelPackImporter(
            credentialStorageAvailable = true,
            openArchive = { throw SecurityException("not visible") },
            activate = { error("must not activate") },
        )

        assertSame(OfficialModelPackImportResult.Unavailable, importer.installFromOfficialPack())
    }

    @Test
    fun rejectsWrongKindAndInvalidArchive() {
        val wrongKind = OfficialContextModelPackImporter(
            credentialStorageAvailable = true,
            openArchive = { ByteArrayInputStream(byteArrayOf(1)) },
            activate = { activeModel(ModelKind.SWIPE_CTC) },
        )
        val invalid = OfficialContextModelPackImporter(
            credentialStorageAvailable = true,
            openArchive = { ByteArrayInputStream(byteArrayOf(1)) },
            activate = { throw IllegalArgumentException("bad signature") },
        )

        assertSame(OfficialModelPackImportResult.Rejected, wrongKind.installFromOfficialPack())
        assertSame(OfficialModelPackImportResult.Rejected, invalid.installFromOfficialPack())
    }

    private fun activeModel(kind: ModelKind): ModelRegistry.ActiveModel {
        val tokenizer = if (kind == ModelKind.CONTEXT_RESCORER) File("tokenizer") else null
        return ModelRegistry.ActiveModel(
            manifest = ModelManifest(
                schemaVersion = 1,
                engineAbi = 1,
                modelKind = kind,
                tensorAbi = if (kind == ModelKind.CONTEXT_RESCORER) {
                    "context-en-de-v1"
                } else {
                    "swipe-latin-v1"
                },
                locales = listOf("en-US", "de"),
                architecture = "fixture",
                parameterCount = 1,
                quantization = "fixture",
                modelSha256 = "a".repeat(64),
                tokenizerSha256 = tokenizer?.let { "b".repeat(64) },
                requiredOnnxOperators = listOf("MatMul"),
                license = "Apache-2.0",
                provenance = listOf(
                    ProvenanceEntry("fixture", "1", "Apache-2.0", "https://example.invalid/model"),
                ),
                minimumAppVersionCode = 1,
            ),
            model = File("model"),
            tokenizer = tokenizer,
        )
    }

    private class CloseTrackingInputStream(bytes: ByteArray) : ByteArrayInputStream(bytes) {
        var closed = false

        override fun close() {
            closed = true
            super.close()
        }
    }
}
