// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.database

import android.content.ClipDescription
import android.content.ContentValues
import android.content.Context
import android.content.SharedPreferences
import android.content.pm.ProviderInfo
import android.net.Uri
import android.os.SystemClock
import android.webkit.MimeTypeMap
import androidx.core.content.FileProvider
import androidx.core.database.getStringOrNull
import helium314.keyboard.latin.ClipboardHistoryEntry
import helium314.keyboard.latin.common.FileUtils
import helium314.keyboard.latin.settings.Defaults
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import helium314.keyboard.latin.utils.ChecksumCalculator
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.prefs
import java.io.File
import kotlin.collections.joinToString

/** Class providing cached access to the clipboard table */
// currently we should not need to worry about synchronizing access (though maybe we could addClip in a coroutine, then it might be relevant)
class ClipboardDao private constructor(private val db: Database) {
    interface Listener {
        fun onClipInserted(position: Int)
        fun onClipsRemoved(position: Int, count: Int)
        fun onClipMoved(oldPosition: Int, newPosition: Int)
        fun onClipsChanged()
    }

    var listener: Listener? = null

    // we clean up old clips when a new clip is added, but not too frequently
    private var lastClearOldClips = 0L

    // cache is loaded at start and never dropped
    private val cache = mutableListOf<ClipboardHistoryEntry>().apply {
        db.readableDatabase.query(
            TABLE,
            arrayOf(COLUMN_ID, COLUMN_TIMESTAMP, COLUMN_PINNED, COLUMN_TEXT, COLUMN_FILE, COLUMN_MIME_TYPE),
            null,
            null,
            null,
            null,
            "$COLUMN_PINNED, $COLUMN_TIMESTAMP DESC" // was only relevant in the initial approach of using a cursor instead of a cache
        ).use {
            while (it.moveToNext()) {
                add(ClipboardHistoryEntry(
                    it.getLong(0),
                    it.getLong(1),
                    it.getInt(2) != 0,
                    it.getStringOrNull(3),
                    it.getStringOrNull(4),
                    it.getStringOrNull(5)?.split('§')?.filter { it.isNotEmpty() },
                ))
            }
        }
        sort()
    }

    private val searchIndex = cache.associateTo(HashMap()) { it.id to ClipboardHistoryPolicy.searchDocument(it) }

    init {
        // Old databases predate the hard limits. Fail closed during migration instead of carrying
        // oversized clipboard payloads forward indefinitely.
        val invalid = cache.filter {
            if (it.filename == null) it.text == null || !ClipboardHistoryPolicy.acceptsText(it.text)
            else !ClipboardHistoryPolicy.acceptsAttachment(it.filename, it.text, it.mimeTypes)
        }
        delete(invalid, notify = false)
        enforceExistingEntryLimits(notify = false)
    }

    fun addClip(timestamp: Long, pinned: Boolean, text: String): Boolean = synchronized(this) {
        if (!ClipboardHistoryPolicy.acceptsText(text)) return@synchronized false
        clearOldClips()
        val existingIndex = cache.indexOfFirst { it.text == text }
        if (existingIndex >= 0 && cache[existingIndex].timeStamp == timestamp)
            return@synchronized true // nothing to do
        if (existingIndex >= 0) {
            updateTimestampAt(existingIndex, timestamp)
            return@synchronized true
        }
        insertNewEntry(timestamp, pinned, text, null, null, null)
    }

    fun addClipUri(timestamp: Long, pinned: Boolean, uri: Uri, description: ClipDescription, context: Context): Boolean = synchronized(this) {
        clearOldClips()
        val extension = if (description.mimeTypeCount == 0) ""
            else ".${MimeTypeMap.getSingleton().getExtensionFromMimeType(description.getMimeType(0))}"
        // clipFilesDir is explicitly credential encrypted; the application default context is
        // device protected so it must never be used for private clipboard payloads.
        val tempFile = File(clipFilesDir, "temp_clip")
        tempFile.delete()
        runCatching { FileUtils.copyContentUriToNewFile(uri, context, tempFile) }.onFailure {
            tempFile.delete()
            return@synchronized false
        }

        // we set the file name to the sha256 of the content to have virtually unique names and an easy way to find duplicates
        val sha256 = ChecksumCalculator.checksum(tempFile)
        val file = File(clipFilesDir, sha256 + extension)

        val existingIndex = cache.indexOfFirst { it.filename == file.name }
        if (existingIndex >= 0) {
            if (cache[existingIndex].timeStamp != timestamp)
                updateTimestampAt(existingIndex, timestamp)
            tempFile.delete()
            return@synchronized true
        }
        if (!tempFile.renameTo(file)) {
            tempFile.delete()
            return@synchronized false
        }
        // we could try getting a thumbnail using context.contentResolver.loadThumbnail(uri, Size(a, b), null)
        // but currently we don't cache them anyway, so no use for that
        val inserted = insertNewEntry(timestamp, pinned, description.label?.toString(), file.name, description.getMimeTypes(), context)
        if (!inserted) file.delete()
        inserted
    }

    // keep pinned and the first non-pinned, others can be deleted
    private fun deleteIfSizeExceeded(prefs: SharedPreferences) {
        val sizeLimit = prefs.getInt(Settings.PREF_CLIPBOARD_FILES_SIZE_LIMIT, Defaults.PREF_CLIPBOARD_FILES_SIZE_LIMIT) * 1000000
        var size = 0L
        var keepMin = 1
        val toRemove = mutableListOf<ClipboardHistoryEntry>()
        cache.forEach {
            if (it.filename == null) return@forEach
            val file = File(clipFilesDir, it.filename)
            size += file.length()
            if (it.isPinned)
                return@forEach
            if (size > sizeLimit) {
                if (keepMin > 0) --keepMin
                else toRemove.add(it)
            }
        }
        delete(toRemove)
    }

    /** only public for restoring backups */
    fun insertNewEntry(timestamp: Long, pinned: Boolean, text: String?, filename: String?, mimeTypes: List<String>?, context: Context?): Boolean = synchronized(this) {
        if (filename == null && (text == null || !ClipboardHistoryPolicy.acceptsText(text))) return@synchronized false
        if (filename != null && !ClipboardHistoryPolicy.acceptsAttachment(filename, text, mimeTypes)) return@synchronized false
        if (!makeRoomForEntry(pinned)) return@synchronized false
        val cv = ContentValues(5)
        cv.put(COLUMN_TIMESTAMP, timestamp)
        cv.put(COLUMN_PINNED, pinned)
        cv.put(COLUMN_TEXT, text)
        cv.put(COLUMN_FILE, filename)
        // § should be a safe separator, not allowed in mime types: https://datatracker.ietf.org/doc/html/rfc6838#section-4.2
        cv.put(COLUMN_MIME_TYPE, mimeTypes?.joinToString("§"))
        val rowId = db.writableDatabase.insert(TABLE, null, cv)

        if (rowId < 0) return@synchronized false

        val entry = ClipboardHistoryEntry(rowId, timestamp, pinned, text, filename, mimeTypes)
        cache.add(entry)
        searchIndex[entry.id] = ClipboardHistoryPolicy.searchDocument(entry)
        cache.sort()
        listener?.onClipInserted(cache.indexOf(entry))
        if (filename != null && context != null)
            deleteIfSizeExceeded(context.prefs())
        true
    }

    private fun makeRoomForEntry(incomingPinned: Boolean): Boolean {
        val removeCount = ClipboardHistoryPolicy.requiredUnpinnedEvictions(
            totalEntries = cache.size,
            unpinnedEntries = cache.count { !it.isPinned },
            incomingPinned = incomingPinned,
        )
        if (removeCount == 0) return true
        val removable = cache.asSequence()
            .filter { !it.isPinned }
            .sortedBy { it.timeStamp }
            .take(removeCount)
            .toList()
        if (removable.size != removeCount) return false
        delete(removable)
        return true
    }

    /** Brings databases created by older versions under the current absolute limits. */
    private fun enforceExistingEntryLimits(notify: Boolean) {
        val removals = LinkedHashSet<ClipboardHistoryEntry>()
        val unpinnedOldestFirst = cache.filter { !it.isPinned }.sortedBy { it.timeStamp }
        val unpinnedOverflow = (unpinnedOldestFirst.size - ClipboardHistoryPolicy.MAX_UNPINNED_ENTRIES).coerceAtLeast(0)
        removals.addAll(unpinnedOldestFirst.take(unpinnedOverflow))

        val totalOverflow = (cache.size - removals.size - ClipboardHistoryPolicy.MAX_TOTAL_ENTRIES).coerceAtLeast(0)
        if (totalOverflow > 0) {
            val remainingOldestFirst = cache.asSequence()
                .filterNot(removals::contains)
                .sortedBy { it.timeStamp }
                .take(totalOverflow)
                .toList()
            removals.addAll(remainingOldestFirst)
        }
        delete(removals.toList(), notify)
    }

    private fun updateTimestampAt(index: Int, timestamp: Long) {
        val entry = cache[index]
        entry.timeStamp = timestamp
        cache.sort()
        listener?.onClipMoved(index, cache.indexOf(entry))
        val cv = ContentValues(1)
        cv.put(COLUMN_TIMESTAMP, timestamp)
        db.writableDatabase.update(TABLE, cv, "$COLUMN_ID = ${entry.id}", null)
    }

    fun get(id: Long) = synchronized(this) { cache.firstOrNull { it.id == id } }

    fun getAll(): List<ClipboardHistoryEntry> = synchronized(this) { cache.toList() }

    fun search(query: String): List<ClipboardHistoryEntry> = synchronized(this) {
        val tokens = ClipboardHistoryPolicy.normalizeQuery(query)
        if (tokens.isEmpty()) cache.toList()
        else cache.filter { ClipboardHistoryPolicy.matches(searchIndex[it.id].orEmpty(), tokens) }
    }

    fun count() = synchronized(this) { cache.size }

    fun sort() = synchronized(this) { cache.sort() }

    fun togglePinned(id: Long) = synchronized(this) {
        val entry = cache.firstOrNull { it.id == id } ?: return@synchronized
        entry.isPinned = !entry.isPinned
        entry.timeStamp = System.currentTimeMillis()
        if (listener != null) {
            val oldPos = cache.indexOf(entry)
            cache.sort()
            val newPos = cache.indexOf(entry)
            listener?.onClipMoved(oldPos, newPos)
        } else {
            cache.sort()
        }
        val cv = ContentValues(2)
        cv.put(COLUMN_PINNED, entry.isPinned)
        cv.put(COLUMN_TIMESTAMP, entry.timeStamp)
        db.writableDatabase.update(TABLE, cv, "$COLUMN_ID = ${entry.id}", null)
    }

    fun deleteClip(id: Long) = synchronized(this) {
        cache.firstOrNull { it.id == id }?.let { delete(listOf(it)) }
    }

    private fun delete(entries: List<ClipboardHistoryEntry>, notify: Boolean = true) = synchronized(this) {
        if (entries.isEmpty()) return@synchronized
        cache.removeAll(entries)
        entries.forEach { searchIndex.remove(it.id) }
        db.writableDatabase.delete(TABLE, "$COLUMN_ID IN (${entries.joinToString(",") { it.id.toString() }})", null)
        entries.forEach { entry ->
            entry.filename?.takeIf(ClipboardHistoryPolicy::isSafeLeafFilename)?.let { File(clipFilesDir, it).delete() }
        }
        if (notify) listener?.onClipsChanged()
    }

    fun clearOldClips(now: Boolean = false) = synchronized(this) {
        if (listener != null)
            return@synchronized // never clear when clipboard is visible
        if (!now && lastClearOldClips > SystemClock.elapsedRealtime() - 5 * 1000)
            return@synchronized

        lastClearOldClips = SystemClock.elapsedRealtime()
        val retentionTime = Settings.getValues()?.mClipboardHistoryRetentionTime ?: 121L
        if (retentionTime > 120) return@synchronized
        val minTime = System.currentTimeMillis() - retentionTime * 60 * 1000L
        clearExpiredBefore(minTime)
    }

    /** Public for deterministic device verification; normal callers use [clearOldClips]. */
    fun clearExpiredBefore(minTime: Long) = synchronized(this) {
        delete(cache.filter { it.timeStamp < minTime && !it.isPinned })
    }

    fun clearNonPinned() = synchronized(this) {
        val toRemove = cache.filter { !it.isPinned }
        if (toRemove.isEmpty())
            return@synchronized // nothing to remove
        delete(toRemove)
    }

    fun clear() = synchronized(this) {
        if (cache.isEmpty()) return@synchronized
        val entries = cache.toList()
        cache.clear()
        searchIndex.clear()
        db.writableDatabase.delete(TABLE, null, null)
        entries.forEach { entry ->
            entry.filename?.takeIf(ClipboardHistoryPolicy::isSafeLeafFilename)?.let { File(clipFilesDir, it).delete() }
        }
        listener?.onClipsRemoved(0, entries.size)
    }

    fun cleanupFiles(prefs: SharedPreferences) {
        if (!prefs.getBoolean(Settings.PREF_CLIPBOARD_USE_FILES, Defaults.PREF_CLIPBOARD_USE_FILES)) {
            delete(cache.filter { it.filename != null && !it.isPinned })
            return
        }

        val files = clipFilesDir.listFiles()?.toMutableList() ?: return
        val fnames = files.mapTo(HashSet()) { it.name }
        val entries = cache.filter { it.filename != null }
        val enames = entries.mapTo(HashSet()) { it.filename }

        val filesToRemove = files.filter { it.name !in enames }
        val entriesToRemove = entries.filter { it.filename!! !in fnames }
        if (filesToRemove.isEmpty() && entriesToRemove.isEmpty()) {
            deleteIfSizeExceeded(prefs)
            return
        }

        Log.w(TAG, "deleting ${filesToRemove.size} files and ${entriesToRemove.size} clipboard entries")
        filesToRemove.forEach { it.delete() }
        delete(entriesToRemove)

        deleteIfSizeExceeded(prefs)
    }

    companion object {
        private const val TAG = "ClipboardDao"

        private const val TABLE = "CLIPBOARD"
        // it's possible timestamp is not unique, so we use a separate ID
        // ID is generated and returned on insert, see https://sqlite.org/rowidtable.html
        private const val COLUMN_ID = "ID"
        private const val COLUMN_TIMESTAMP = "TIMESTAMP"
        private const val COLUMN_PINNED = "PINNED"
        private const val COLUMN_TEXT = "TEXT" // we could enforce unique text, but that's only necessary if we can drop the cache (later)
        private const val COLUMN_FILE = "FILE" // path relative to files dir
        private const val COLUMN_MIME_TYPE = "MIME_TYPE" // for files, actually a list of mime types according to clipboard description
        const val CREATE_TABLE = """
            CREATE TABLE $TABLE (
                $COLUMN_ID INTEGER PRIMARY KEY,
                $COLUMN_TIMESTAMP INTEGER NOT NULL,
                $COLUMN_PINNED TINYINT NOT NULL,
                $COLUMN_TEXT TEXT
            )
        """

        const val ADD_FILE_COLUMN = "ALTER TABLE $TABLE ADD COLUMN $COLUMN_FILE TEXT"
        const val ADD_MIME_TYPE_COLUMN = "ALTER TABLE $TABLE ADD COLUMN $COLUMN_MIME_TYPE TEXT"

        private var instance: ClipboardDao? = null
        lateinit var clipFilesDir: File
            private set

        /** Returns the instance or creates a new one. Returns null if instance can't be created (e.g. no access to db due to device being locked) */
        fun getInstance(context: Context): ClipboardDao? {
            if (instance == null)
                try {
                    val privateContext = CredentialEncryptedStorage.contextOrNull(context) ?: return null
                    clipFilesDir = File(privateContext.filesDir, "clipboard")
                    clipFilesDir.mkdirs()
                    instance = ClipboardDao(Database.getInstance(context))
                    instance?.cleanupFiles(context.prefs())
                } catch (e: Throwable) {
                    Log.e(TAG, "can't create ClipboardDao", e)
                }
            return instance
        }

        private fun ClipDescription.getMimeTypes(): List<String> {
            val types = mutableListOf<String>()
            for (i in 0..<mimeTypeCount) {
                types.add(getMimeType(i))
            }
            if (types.isEmpty())
                types.add("*/*")
            return types
        }
    }
}

class ClipboardContentProvider : FileProvider() {
    override fun attachInfo(context: Context, info: ProviderInfo) {
        // The application context is credential protected and this provider is unavailable until
        // unlock. Direct-Boot assets opt into device-protected storage elsewhere.
        super.attachInfo(context, info)
    }
}
