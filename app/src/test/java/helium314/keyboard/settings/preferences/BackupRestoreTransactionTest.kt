// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import kotlin.test.assertEquals

class BackupRestoreTransactionTest {
    private val root = Files.createTempDirectory("libreboard-restore-transaction").toFile()
    private val credential = root.resolve("credential/files")
    private val device = root.resolve("device/files")
    private val journal = root.resolve("no-backup")
    private val isSelected: (String) -> Boolean = { it.startsWith("selected/") }

    @After
    fun tearDown() {
        root.deleteRecursively()
    }

    @Test
    fun stagedCrashRestoresSelectedFilesWithoutTouchingExcludedState() {
        writeOriginals()
        val transaction = begin()
        transaction.stageSelectedFiles()
        credential.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("credential-partial") }
        device.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("device-partial") }
        credential.resolve("models/model.onnx").writeText("model-updated-concurrently")

        recover()

        assertEquals("credential-old", credential.resolve("selected/state").readText())
        assertEquals("device-old", device.resolve("selected/state").readText())
        assertEquals("model-updated-concurrently", credential.resolve("models/model.onnx").readText())
        assertEquals("clip", credential.resolve("clipboard/attachment").readText())
        assertFalse(transaction.stagedCredential.exists())
        assertFalse(transaction.stagedDevice.exists())
        assertTrue(journal.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun stagedCrashAfterOnlyFirstFileMoveStillRollsThatFileBack() {
        writeOriginals()
        val transaction = begin()
        val source = credential.resolve("selected/state")
        val staged = transaction.stagedCredential.resolve("selected/state")
        staged.parentFile!!.mkdirs()
        assertTrue(source.renameTo(staged))

        recover()

        assertEquals("credential-old", credential.resolve("selected/state").readText())
        assertEquals("device-old", device.resolve("selected/state").readText())
        assertEquals("model", credential.resolve("models/model.onnx").readText())
    }

    @Test
    fun committedCrashKeepsRestoredSelectionsAndDeletesRollbackCopies() {
        writeOriginals()
        val transaction = begin()
        transaction.stageSelectedFiles()
        credential.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("credential-new") }
        device.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("device-new") }
        transaction.markCommitted()

        recover()

        assertEquals("credential-new", credential.resolve("selected/state").readText())
        assertEquals("device-new", device.resolve("selected/state").readText())
        assertEquals("model", credential.resolve("models/model.onnx").readText())
        assertFalse(transaction.stagedCredential.exists())
        assertFalse(transaction.stagedDevice.exists())
        assertTrue(journal.listFiles().orEmpty().isEmpty())
    }

    @Test
    fun inProcessFailureAfterJournalCommitCanStillRollBack() {
        writeOriginals()
        val transaction = begin()
        transaction.stageSelectedFiles()
        credential.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("credential-new") }
        device.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("device-new") }
        transaction.markCommitted()

        transaction.rollback()

        assertEquals("credential-old", credential.resolve("selected/state").readText())
        assertEquals("device-old", device.resolve("selected/state").readText())
        assertEquals("model", credential.resolve("models/model.onnx").readText())
        assertEquals("clip", credential.resolve("clipboard/attachment").readText())
        assertTrue(journal.listFiles().orEmpty().isEmpty())
    }

    private fun begin() = BackupRestoreTransaction.begin(credential, device, journal, isSelected)

    private fun recover() = BackupRestoreTransaction.recover(credential, device, journal, isSelected)

    private fun writeOriginals() {
        credential.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("credential-old") }
        device.resolve("selected/state").apply { parentFile!!.mkdirs(); writeText("device-old") }
        credential.resolve("models/model.onnx").apply { parentFile!!.mkdirs(); writeText("model") }
        credential.resolve("clipboard/attachment").apply { parentFile!!.mkdirs(); writeText("clip") }
    }
}
