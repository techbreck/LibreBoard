// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import java.io.File
import java.io.FileOutputStream

/** Crash-consistent journal for replacement of allowlisted CE and device-protected files. */
object BackupRestoreTransaction {
    private const val MARKER_PREFIX = "libreboard-restore-"
    private const val STAGED_SUFFIX = ".staged"
    private const val COMMITTED_SUFFIX = ".committed"
    private val markerPattern = Regex("^${MARKER_PREFIX}([0-9]+)(\\.staged|\\.committed)$")

    class Transaction internal constructor(
        private val activeCredential: File,
        private val activeDevice: File,
        private val marker: File,
        val stagedCredential: File,
        val stagedDevice: File,
        private val isSelectedPath: (String) -> Boolean,
    ) {
        /** Moves only files governed by the backup contract into the rollback trees. */
        fun stageSelectedFiles() {
            stageSelected(activeCredential, stagedCredential, isSelectedPath)
            stageSelected(activeDevice, stagedDevice, isSelectedPath)
        }

        fun markCommitted() {
            require(marker.name.endsWith(STAGED_SUFFIX)) { "restore transaction is not staged" }
            val committed = marker.resolveSibling(marker.name.removeSuffix(STAGED_SUFFIX) + COMMITTED_SUFFIX)
            require(!committed.exists() && marker.renameTo(committed)) {
                "could not commit restore transaction"
            }
        }

        fun rollback() {
            restoreSelected(activeCredential, stagedCredential, isSelectedPath)
            restoreSelected(activeDevice, stagedDevice, isSelectedPath)
            require(!marker.exists() || marker.delete()) { "could not remove restore journal" }
            val committed = marker.resolveSibling(marker.name.removeSuffix(STAGED_SUFFIX) + COMMITTED_SUFFIX)
            require(!committed.exists() || committed.delete()) { "could not remove committed restore journal" }
        }

        fun cleanupCommitted() {
            val committed = marker.resolveSibling(marker.name.removeSuffix(STAGED_SUFFIX) + COMMITTED_SUFFIX)
            val failures = listOf(stagedCredential, stagedDevice).filter { it.exists() && !it.deleteRecursively() }
            require(failures.isEmpty()) { "could not remove committed restore rollback data" }
            require(!committed.exists() || committed.delete()) { "could not remove committed restore journal" }
        }
    }

    @Synchronized
    fun begin(
        activeCredential: File,
        activeDevice: File,
        journalDir: File,
        isSelectedPath: (String) -> Boolean,
    ): Transaction {
        recover(activeCredential, activeDevice, journalDir, isSelectedPath)
        require(activeCredential.isDirectory && activeDevice.isDirectory) {
            "restore storage roots are unavailable"
        }
        require(journalDir.isDirectory || journalDir.mkdirs()) { "could not create restore journal directory" }
        val id = nextId(journalDir)
        val marker = journalDir.resolve("$MARKER_PREFIX$id$STAGED_SUFFIX")
        require(marker.createNewFile()) { "could not create restore journal" }
        FileOutputStream(marker).use { output ->
            output.write(byteArrayOf(1))
            output.fd.sync()
        }
        return Transaction(
            activeCredential,
            activeDevice,
            marker,
            File(activeCredential.absolutePath + ".restore-$id"),
            File(activeDevice.absolutePath + ".restore-$id"),
            isSelectedPath,
        )
    }

    /** Completes or rolls back a file transaction interrupted by process death. */
    @Synchronized
    fun recover(
        activeCredential: File,
        activeDevice: File,
        journalDir: File,
        isSelectedPath: (String) -> Boolean,
    ) {
        val markers = journalDir.listFiles().orEmpty().mapNotNull { marker ->
            markerPattern.matchEntire(marker.name)?.let { match ->
                Triple(marker, match.groupValues[1], match.groupValues[2])
            }
        }
        require(markers.size <= 1) { "multiple restore journals require manual recovery" }
        val (marker, id, phase) = markers.singleOrNull() ?: return
        val stagedCredential = File(activeCredential.absolutePath + ".restore-$id")
        val stagedDevice = File(activeDevice.absolutePath + ".restore-$id")
        if (phase == STAGED_SUFFIX) {
            restoreSelected(activeCredential, stagedCredential, isSelectedPath)
            restoreSelected(activeDevice, stagedDevice, isSelectedPath)
        } else {
            require(!stagedCredential.exists() || stagedCredential.deleteRecursively()) {
                "could not clean credential-protected restore rollback data"
            }
            require(!stagedDevice.exists() || stagedDevice.deleteRecursively()) {
                "could not clean device-protected restore rollback data"
            }
        }
        require(marker.delete()) { "could not remove recovered restore journal" }
    }

    private fun stageSelected(active: File, staged: File, isSelectedPath: (String) -> Boolean) {
        if (!active.isDirectory) return
        require(!staged.exists() && staged.mkdirs()) { "could not prepare restore rollback tree" }
        active.walkTopDown().filter(File::isFile).toList().forEach { source ->
            val relative = source.relativeTo(active).invariantSeparatorsPath
            if (!isSelectedPath(relative)) return@forEach
            val destination = BackupArchivePolicy.safeTarget(staged, relative)
                ?: throw IllegalArgumentException("unsafe rollback path")
            require(destination.parentFile?.let { it.isDirectory || it.mkdirs() } == true) {
                "could not create restore rollback directory"
            }
            require(!destination.exists() && source.renameTo(destination)) {
                "could not stage restore rollback file"
            }
        }
    }

    private fun restoreSelected(active: File, staged: File, isSelectedPath: (String) -> Boolean) {
        if (!staged.isDirectory) return
        if (active.isDirectory) {
            active.walkTopDown().filter(File::isFile).toList().forEach { restored ->
                val relative = restored.relativeTo(active).invariantSeparatorsPath
                if (isSelectedPath(relative)) {
                    require(restored.delete()) { "could not remove partial restored file" }
                }
            }
        }
        staged.walkTopDown().filter(File::isFile).toList().forEach { source ->
            val relative = source.relativeTo(staged).invariantSeparatorsPath
            require(isSelectedPath(relative)) { "rollback contains a file outside the backup contract" }
            val destination = BackupArchivePolicy.safeTarget(active, relative)
                ?: throw IllegalArgumentException("unsafe rollback target")
            require(destination.parentFile?.let { it.isDirectory || it.mkdirs() } == true) {
                "could not recreate restored directory"
            }
            require(!destination.exists() && source.renameTo(destination)) {
                "could not reinstate restore rollback file"
            }
        }
        require(staged.deleteRecursively()) { "could not remove restored rollback tree" }
    }

    private fun nextId(journalDir: File): String {
        repeat(10) { offset ->
            val candidate = (System.nanoTime().toULong() + offset.toUInt()).toString()
            val staged = journalDir.resolve("$MARKER_PREFIX$candidate$STAGED_SUFFIX")
            val committed = journalDir.resolve("$MARKER_PREFIX$candidate$COMMITTED_SUFFIX")
            if (!staged.exists() && !committed.exists()) return candidate
        }
        throw IllegalStateException("could not allocate restore transaction id")
    }

    private fun File.resolveSibling(name: String): File =
        requireNotNull(parentFile) { "restore journal has no parent" }.resolve(name)
}
