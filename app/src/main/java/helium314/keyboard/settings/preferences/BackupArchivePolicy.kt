// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

/** Resource and path limits applied while reading a user-selected backup archive. */
object BackupArchivePolicy {
    const val MAX_ARCHIVE_ENTRIES = 2_048
    const val MAX_TOTAL_UNCOMPRESSED_BYTES = 128L * 1024 * 1024
    const val MAX_FILE_BYTES = 64L * 1024 * 1024

    class ReadBudget(
        private val maximumEntries: Int = MAX_ARCHIVE_ENTRIES,
        private val maximumTotalBytes: Long = MAX_TOTAL_UNCOMPRESSED_BYTES,
    ) {
        private var entries = 0
        private var totalBytes = 0L

        fun beginEntry() {
            entries++
            require(entries <= maximumEntries) { "backup has too many entries" }
        }

        fun read(input: InputStream, maximumEntryBytes: Long): ByteArray {
            require(maximumEntryBytes <= Int.MAX_VALUE) { "entry limit is unsupported" }
            val output = ByteArrayOutputStream(minOf(maximumEntryBytes.toInt(), 8192))
            copy(input, maximumEntryBytes) { bytes, count -> output.write(bytes, 0, count) }
            return output.toByteArray()
        }

        fun copyTo(input: InputStream, target: File, maximumEntryBytes: Long = MAX_FILE_BYTES) {
            require(target.parentFile?.let { it.isDirectory || it.mkdirs() } == true) {
                "could not create backup target parent"
            }
            require(!target.exists()) { "duplicate backup target" }
            try {
                FileOutputStream(target).use { output ->
                    copy(input, maximumEntryBytes) { bytes, count -> output.write(bytes, 0, count) }
                    output.fd.sync()
                }
            } catch (failure: Throwable) {
                target.delete()
                throw failure
            }
        }

        fun drain(input: InputStream, maximumEntryBytes: Long = MAX_FILE_BYTES) {
            copy(input, maximumEntryBytes) { _, _ -> }
        }

        private inline fun copy(
            input: InputStream,
            maximumEntryBytes: Long,
            write: (ByteArray, Int) -> Unit,
        ) {
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            var entryBytes = 0L
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                entryBytes += count
                totalBytes += count
                require(entryBytes <= maximumEntryBytes) { "backup entry is too large" }
                require(totalBytes <= maximumTotalBytes) { "backup expands beyond the total size limit" }
                write(buffer, count)
            }
        }
    }

    fun safeTarget(baseDir: File, entryName: String): File? {
        val segments = entryName.split('/')
        if (entryName.isEmpty()
            || entryName.indexOf('\u0000') >= 0
            || entryName.startsWith('/')
            || entryName.indexOf('\\') >= 0
            || File(entryName).isAbsolute
            || segments.any { it.isEmpty() || it == "." || it == ".." }
        ) return null
        val canonicalBase = baseDir.canonicalFile
        val canonicalTarget = File(baseDir, entryName).canonicalFile
        return canonicalTarget.takeIf {
            it.path != canonicalBase.path && it.path.startsWith(canonicalBase.path + File.separator)
        }
    }
}
