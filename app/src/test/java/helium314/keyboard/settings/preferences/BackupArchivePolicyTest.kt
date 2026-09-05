// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.ByteArrayInputStream
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class BackupArchivePolicyTest {
    private val root = Files.createTempDirectory("libreboard-backup-test").toFile()

    @After
    fun tearDown() {
        root.deleteRecursively()
    }

    @Test
    fun archiveTargetsStayStrictlyInsideTheRestoreRoot() {
        assertNull(BackupArchivePolicy.safeTarget(root, ""))
        assertNull(BackupArchivePolicy.safeTarget(root, "."))
        assertNull(BackupArchivePolicy.safeTarget(root, "../outside"))
        assertNull(BackupArchivePolicy.safeTarget(root, root.absolutePath))
        assertNull(BackupArchivePolicy.safeTarget(root, "..\\outside"))
        assertNull(BackupArchivePolicy.safeTarget(root, "layouts/../models/model.onnx"))
        assertNull(BackupArchivePolicy.safeTarget(root, "layouts//custom.json"))
        assertEquals(
            root.resolve("layouts/custom.json").canonicalFile,
            BackupArchivePolicy.safeTarget(root, "layouts/custom.json"),
        )
    }

    @Test
    fun entryCountAndExpandedByteBudgetsAreEnforced() {
        val entryBudget = BackupArchivePolicy.ReadBudget(maximumEntries = 1)
        entryBudget.beginEntry()
        assertFailsWith<IllegalArgumentException> { entryBudget.beginEntry() }

        val totalBudget = BackupArchivePolicy.ReadBudget(maximumTotalBytes = 6)
        totalBudget.beginEntry()
        assertArrayEquals(ByteArray(4), totalBudget.read(ByteArrayInputStream(ByteArray(4)), 4))
        totalBudget.beginEntry()
        assertFailsWith<IllegalArgumentException> {
            totalBudget.read(ByteArrayInputStream(ByteArray(3)), 3)
        }
    }

    @Test
    fun perEntryLimitDeletesAPartialDestination() {
        val target = root.resolve("layouts/partial")
        val budget = BackupArchivePolicy.ReadBudget()
        budget.beginEntry()

        assertFailsWith<IllegalArgumentException> {
            budget.copyTo(ByteArrayInputStream(ByteArray(5)), target, maximumEntryBytes = 4)
        }
        assertFalse(target.exists())
    }

    @Test
    fun duplicateTargetCannotOverwriteAnExtractedFile() {
        val target = root.resolve("layouts/custom")
        val budget = BackupArchivePolicy.ReadBudget()
        budget.beginEntry()
        budget.copyTo(ByteArrayInputStream("first".encodeToByteArray()), target)
        budget.beginEntry()

        assertFailsWith<IllegalArgumentException> {
            budget.copyTo(ByteArrayInputStream("second".encodeToByteArray()), target)
        }
        assertEquals("first", target.readText())
    }

    @Test
    fun settingsPayloadRequiresEveryTypedSectionExactlyOnce() {
        val valid = listOf(
            "boolean settings", "{\"enabled\":true}",
            "int settings", "{\"count\":2}",
            "long settings", "{\"epoch\":3}",
            "float settings", "{\"scale\":1.5}",
            "string settings", "{\"name\":\"LibreBoard\"}",
            "string set settings", "{\"languages\":[\"en-US\",\"de\"]}",
        )

        val parsed = requireNotNull(parseBackupSettings(valid))
        assertEquals(true, parsed["enabled"])
        assertEquals(2, parsed["count"])
        assertEquals(3L, parsed["epoch"])
        assertEquals("LibreBoard", parsed["name"])
        assertEquals(setOf("en-US", "de"), parsed["languages"])
        assertNull(parseBackupSettings(valid.dropLast(1)))
        assertNull(parseBackupSettings(valid + listOf("unexpected", "{}")))
        assertNull(parseBackupSettings(valid.toMutableList().apply { this[0] = "int settings" }))
    }

    @Test
    fun settingsPayloadRejectsAKeyDeclaredWithMultipleTypes() {
        val duplicate = listOf(
            "boolean settings", "{\"same\":true}",
            "int settings", "{\"same\":2}",
            "long settings", "{}",
            "float settings", "{}",
            "string settings", "{}",
            "string set settings", "{}",
        )
        assertNull(parseBackupSettings(duplicate))
    }
}
