// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.database

import helium314.keyboard.latin.ClipboardHistoryEntry
import java.util.Locale

/** Hard limits and in-memory search rules for credential-encrypted clipboard history. */
object ClipboardHistoryPolicy {
    /** Prevent one hostile or accidental clipboard item from consuming unbounded private storage. */
    const val MAX_TEXT_CHARS = 100_000

    /** Pinned and unpinned entries combined. */
    const val MAX_TOTAL_ENTRIES = 200

    /** Recent unpinned history is deliberately smaller than the absolute limit. */
    const val MAX_UNPINNED_ENTRIES = 100

    /** Search text never leaves memory, and is bounded before normalization. */
    const val MAX_QUERY_CHARS = 256

    const val MAX_FILENAME_CHARS = 255
    const val MAX_MIME_TYPES = 16
    const val MAX_MIME_TYPE_CHARS = 255

    fun acceptsText(text: String): Boolean = text.isNotEmpty() && text.length <= MAX_TEXT_CHARS

    fun acceptsAttachment(filename: String, label: String?, mimeTypes: List<String>?): Boolean =
        isSafeLeafFilename(filename) &&
            (label == null || label.length <= MAX_TEXT_CHARS) &&
            mimeTypes != null && mimeTypes.isNotEmpty() && mimeTypes.size <= MAX_MIME_TYPES &&
            mimeTypes.all { it.isNotEmpty() && it.length <= MAX_MIME_TYPE_CHARS && '§' !in it }

    fun isSafeLeafFilename(filename: String): Boolean =
        filename.isNotEmpty() && filename.length <= MAX_FILENAME_CHARS &&
            filename != "." && filename != ".." &&
            '/' !in filename && '\\' !in filename

    fun requiredUnpinnedEvictions(
        totalEntries: Int,
        unpinnedEntries: Int,
        incomingPinned: Boolean,
    ): Int = maxOf(
        (totalEntries + 1 - MAX_TOTAL_ENTRIES).coerceAtLeast(0),
        (unpinnedEntries + (if (incomingPinned) 0 else 1) - MAX_UNPINNED_ENTRIES).coerceAtLeast(0),
    )

    fun normalizeQuery(query: String): List<String> = query
        .take(MAX_QUERY_CHARS)
        .trim()
        .lowercase(Locale.ROOT)
        .split(Regex("\\s+"))
        .filter(String::isNotEmpty)

    fun searchDocument(entry: ClipboardHistoryEntry): String = buildString {
        entry.text?.let { append(it) }
        entry.filename?.let { append('\n').append(it) }
        entry.mimeTypes?.forEach { append('\n').append(it) }
    }.lowercase(Locale.ROOT)

    fun matches(searchDocument: String, queryTokens: List<String>): Boolean =
        queryTokens.all(searchDocument::contains)
}
