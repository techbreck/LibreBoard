// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.personal

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import androidx.core.database.sqlite.transaction
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.CommitObservation
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.PersonalStore
import helium314.keyboard.latin.engine.RejectionObservation
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.normalizeCandidate
import helium314.keyboard.latin.engine.geometric.LexiconWord
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.security.MessageDigest
import kotlin.math.exp
import kotlin.math.ln1p

class SqlitePersonalStore private constructor(context: Context) :
    SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION), PersonalStore {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(CREATE_UNIGRAM)
        db.execSQL(CREATE_NGRAM)
        db.execSQL(CREATE_REJECTION)
        db.execSQL("CREATE INDEX ngram_lookup ON ngram(language, prefix, order_n)")
        db.execSQL("CREATE INDEX unigram_prefix ON unigram(language, word)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion == 1 && newVersion == 2) {
            db.execSQL("ALTER TABLE unigram ADD COLUMN surface TEXT")
            db.execSQL("UPDATE unigram SET surface = word WHERE surface IS NULL")
            return
        }
        throw IllegalStateException("No personal-store migration exists from $oldVersion to $newVersion")
    }

    override fun observeCommit(observation: CommitObservation) {
        val committedSurfaces = observation.tokens.filter { it.isNotBlank() }.takeLast(MAX_PHRASE_TOKENS)
        val committedTokens = committedSurfaces.map(::normalizeCandidate)
        val precedingTokens = observation.precedingTokens.map(::normalizeCandidate).filter { it.isNotBlank() }
            .takeLast(MAX_PHRASE_TOKENS - 1)
        if (committedTokens.isEmpty() || observation.languageTag.isBlank()) return
        writableDatabase.transaction {
            committedTokens.zip(committedSurfaces).forEach { (word, surface) ->
                incrementUnigram(this, word, surface, observation.languageTag, observation.timestampMillis)
            }
            val sequence = precedingTokens + committedTokens
            val firstCommitted = precedingTokens.size
            for (end in firstCommitted + 1..sequence.size) {
                for (order in 2..minOf(MAX_PHRASE_TOKENS, end)) {
                    val window = sequence.subList(end - order, end)
                    incrementNgram(
                        this,
                        prefix = window.dropLast(1).joinToString(TOKEN_SEPARATOR),
                        continuation = window.last(),
                        language = observation.languageTag,
                        order = order,
                        timestamp = observation.timestampMillis,
                    )
                }
            }
            if (observation.wasManualSelection && observation.correctionRaw != null && observation.contextFingerprint != null) {
                recordManualAcceptance(
                    this,
                    normalizeCandidate(observation.correctionRaw),
                    committedTokens.last(),
                    observation.languageTag,
                    observation.contextFingerprint,
                )
            }
            deleteExpiredRejections(this, observation.timestampMillis)
        }
    }

    /** Undo the counts written for one confirmed commit after an immediate correction revert. */
    fun compensateCommit(tokens: List<String>, precedingTokens: List<String>, languageTag: String) {
        val committed = tokens.map(::normalizeCandidate).filter { it.isNotBlank() }.takeLast(MAX_PHRASE_TOKENS)
        val preceding = precedingTokens.map(::normalizeCandidate).filter { it.isNotBlank() }
            .takeLast(MAX_PHRASE_TOKENS - 1)
        if (committed.isEmpty() || languageTag.isBlank()) return
        writableDatabase.transaction {
            committed.forEach { word ->
                decrementOrDelete(this, "unigram", "word = ? AND language = ?", arrayOf(word, languageTag))
            }
            val sequence = preceding + committed
            val firstCommitted = preceding.size
            for (end in firstCommitted + 1..sequence.size) {
                for (order in 2..minOf(MAX_PHRASE_TOKENS, end)) {
                    val window = sequence.subList(end - order, end)
                    decrementOrDelete(
                        this,
                        "ngram",
                        "prefix = ? AND continuation = ? AND language = ? AND order_n = ?",
                        arrayOf(
                            window.dropLast(1).joinToString(TOKEN_SEPARATOR), window.last(), languageTag,
                            order.toString(),
                        ),
                    )
                }
            }
        }
    }

    override fun observeRejection(observation: RejectionObservation) {
        val values = ContentValues().apply {
            put("raw", normalizeCandidate(observation.raw))
            put("replacement", normalizeCandidate(observation.replacement))
            put("language", observation.languageTag)
            put("context_hash", observation.contextFingerprint)
            put("rejected_at", observation.timestampMillis)
            put("manual_accepts", 0)
        }
        writableDatabase.insertWithOnConflict("rejection", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    override fun suggest(request: TypingRequest, deadline: Deadline): List<Candidate> {
        if (!request.fieldPolicy.allowsPersistence || deadline.expired) return emptyList()
        val result = linkedMapOf<Pair<String, String>, Candidate>()
        val now = System.currentTimeMillis()
        val raw = normalizeCandidate(request.rawText)
        val db = readableDatabase

        if (raw.isNotEmpty()) {
            val prefix = escapeLike(raw) + "%"
            request.enabledLanguages.forEach { language ->
                if (deadline.expired) return@forEach
                // Fetch the exact personal word independently so a large prefix family can never
                // push the user's explicit entry outside the bounded completion query.
                db.query(
                    "unigram",
                    arrayOf("word", "surface", "count", "last_used"),
                    "language = ? AND word = ?",
                    arrayOf(language, raw),
                    null,
                    null,
                    null,
                    "1",
                ).use { cursor ->
                    if (cursor.moveToFirst() && !deadline.expired) {
                        val word = cursor.getString(0)
                        val decayed = decayedCount(cursor.getLong(2), cursor.getLong(3), now)
                        result[word to language] = Candidate(
                            surface = cursor.getString(1) ?: word,
                            languageTag = language,
                            sources = setOf(CandidateSource.PERSONAL),
                            components = ScoreComponents(personal = ln1p(decayed)),
                            exactPersonalMatch = true,
                        )
                    }
                }
                if (deadline.expired) return@forEach
                db.query(
                    "unigram",
                    arrayOf("word", "surface", "count", "last_used"),
                    "language = ? AND word LIKE ? ESCAPE '\\'",
                    arrayOf(language, prefix),
                    null,
                    null,
                    "count DESC, last_used DESC",
                    "32",
                ).use { cursor ->
                    while (cursor.moveToNext() && !deadline.expired) {
                        val word = cursor.getString(0)
                        val surface = cursor.getString(1) ?: word
                        val decayed = decayedCount(cursor.getLong(2), cursor.getLong(3), now)
                        val candidate = Candidate(
                            surface = surface,
                            languageTag = language,
                            sources = setOf(CandidateSource.PERSONAL),
                            components = ScoreComponents(personal = ln1p(decayed)),
                            exactPersonalMatch = word == raw,
                            rejectionPenalty = rejectionPenalty(db, raw, word, language, request.precedingContext),
                        )
                        result[word to language] = candidate
                    }
                }
            }
        }

        val contextTokens = tokenize(request.precedingContext).takeLast(MAX_PHRASE_TOKENS - 1)
        if (contextTokens.isNotEmpty()) {
            request.enabledLanguages.forEach { language ->
                for (prefixLength in contextTokens.size downTo 1) {
                    if (deadline.expired) break
                    val prefix = contextTokens.takeLast(prefixLength).joinToString(TOKEN_SEPARATOR)
                    db.query(
                        "ngram",
                        arrayOf("continuation", "count", "last_used", "order_n"),
                        "language = ? AND prefix = ? AND order_n = ?",
                        arrayOf(language, prefix, (prefixLength + 1).toString()),
                        null,
                        null,
                        "count DESC, last_used DESC",
                        "16",
                    ).use { cursor ->
                        while (cursor.moveToNext() && !deadline.expired) {
                            val continuation = cursor.getString(0)
                            if (raw.isNotEmpty() && !continuation.startsWith(raw)) continue
                            val decayed = decayedCount(cursor.getLong(1), cursor.getLong(2), now)
                            val existing = result[continuation to language]
                            result[continuation to language] = (existing ?: Candidate(
                                surface = preferredSurface(db, continuation, language),
                                languageTag = language,
                                sources = emptySet(),
                            )).copy(
                                sources = existing?.sources.orEmpty() + setOf(CandidateSource.PERSONAL, CandidateSource.PERSONAL_PHRASE),
                                components = (existing?.components ?: ScoreComponents()).copy(personal = ln1p(decayed) + prefixLength),
                                rejectionPenalty = rejectionPenalty(db, raw, continuation, language, request.precedingContext),
                            )
                        }
                    }
                    if (result.size >= 16) break
                }
            }
        }
        return result.values.sortedWith(
            compareByDescending<Candidate> { it.exactPersonalMatch }
                .thenByDescending { it.components.personal },
        ).take(32)
    }

    /**
     * Returns only learned unigram surfaces for swipe decoding. Length filtering is deliberately
     * performed in Kotlin because SQLite counts punctuation that the CTC alphabet omits. The SQL
     * query remains bounded, so a large personal vocabulary cannot monopolize the decoder deadline.
     */
    fun swipeLexicon(
        languageTags: List<String>,
        approximateLength: Int,
        maximumWords: Int,
        deadline: Deadline,
    ): List<LexiconWord> {
        if (maximumWords <= 0 || deadline.expired) return emptyList()
        val minimumLength = (approximateLength - SWIPE_LENGTH_TOLERANCE_BELOW).coerceAtLeast(1)
        val now = System.currentTimeMillis()
        val result = ArrayList<LexiconWord>(minOf(maximumWords, 64))
        val seen = hashSetOf<Pair<String, String>>()
        languageTags.filter(String::isNotBlank).distinct().forEach { language ->
            if (deadline.expired || result.size >= maximumWords) return@forEach
            val remaining = maximumWords - result.size
            readableDatabase.query(
                "unigram",
                arrayOf("word", "surface", "count", "last_used"),
                "language = ?",
                arrayOf(language),
                null,
                null,
                "count DESC, last_used DESC",
                minOf(MAX_PERSONAL_SWIPE_QUERY_ROWS, remaining * 2).toString(),
            ).use { cursor ->
                while (cursor.moveToNext() && !deadline.expired && result.size < maximumWords) {
                    val normalized = cursor.getString(0)
                    if (swipeEmissionLength(normalized) !in minimumLength..(
                            approximateLength + SWIPE_LENGTH_TOLERANCE_ABOVE
                        ).coerceAtMost(MAX_PERSONAL_WORD_LENGTH)
                    ) continue
                    val key = normalized to language
                    if (!seen.add(key)) continue
                    val decayed = decayedCount(cursor.getLong(2), cursor.getLong(3), now)
                    result += LexiconWord(
                        word = cursor.getString(1) ?: normalized,
                        languageTag = language,
                        frequency = decayed.coerceIn(0.0, Int.MAX_VALUE.toDouble()).toInt(),
                        personal = true,
                    )
                }
            }
        }
        return result
    }

    fun isCorrectionSuppressed(raw: String, replacement: String, languageTag: String): Boolean {
        val normalizedRaw = normalizeCandidate(raw)
        val normalizedReplacement = normalizeCandidate(replacement)
        if (normalizedRaw.isBlank() || normalizedReplacement.isBlank() || normalizedRaw == normalizedReplacement) return false
        val cutoff = System.currentTimeMillis() - REJECTION_RETENTION_MILLIS
        readableDatabase.query(
            "rejection",
            arrayOf("1"),
            "raw = ? AND replacement = ? AND language = ? AND rejected_at >= ? AND manual_accepts < ?",
            arrayOf(
                normalizedRaw, normalizedReplacement, languageTag, cutoff.toString(),
                REQUIRED_MANUAL_ACCEPTS.toString(),
            ),
            null, null, null, "1",
        ).use { return it.moveToFirst() }
    }

    override fun export(): ByteArray {
        val db = readableDatabase
        val unigrams = mutableListOf<ExportUnigram>()
        db.query("unigram", arrayOf("word", "language", "count", "last_used", "surface"), null, null, null, null,
            "language, word").use { c ->
            while (c.moveToNext()) unigrams += ExportUnigram(
                c.getString(0), c.getString(1), c.getLong(2), c.getLong(3), c.getString(4) ?: c.getString(0),
            )
        }
        val ngrams = mutableListOf<ExportNgram>()
        db.query("ngram", arrayOf("prefix", "continuation", "language", "order_n", "count", "last_used"),
            null, null, null, null, "language, order_n, prefix, continuation").use { c ->
            while (c.moveToNext()) ngrams += ExportNgram(
                c.getString(0), c.getString(1), c.getString(2), c.getInt(3), c.getLong(4), c.getLong(5),
            )
        }
        val rejections = mutableListOf<ExportRejection>()
        db.query("rejection", arrayOf("raw", "replacement", "language", "context_hash", "rejected_at", "manual_accepts"),
            null, null, null, null, "language, raw, replacement, context_hash").use { c ->
            while (c.moveToNext()) rejections += ExportRejection(
                c.getString(0), c.getString(1), c.getString(2), c.getString(3), c.getLong(4), c.getInt(5),
            )
        }
        return JSON.encodeToString(
            PersonalExport(EXPORT_SCHEMA_VERSION, unigrams, ngrams, rejections),
        ).encodeToByteArray()
    }

    /** Replace learned data from a validated, versioned manual-backup payload in one transaction. */
    fun restore(payload: ByteArray) {
        require(payload.size <= MAX_EXPORT_BYTES) { "Personal backup is too large" }
        val restored = JSON.decodeFromString<PersonalExport>(payload.decodeToString())
        require(restored.schemaVersion in 1..EXPORT_SCHEMA_VERSION) { "Unsupported personal backup schema" }
        require(restored.unigrams.size + restored.ngrams.size + restored.rejections.size <= MAX_EXPORT_ROWS) {
            "Personal backup has too many rows"
        }
        restored.validate()
        writableDatabase.transaction {
            delete("unigram", null, null)
            delete("ngram", null, null)
            delete("rejection", null, null)
            restored.unigrams.forEach { row ->
                insertOrThrow("unigram", null, ContentValues().apply {
                    put("word", row.word); put("language", row.language)
                    put("count", row.count); put("last_used", row.lastUsed); put("surface", row.surface)
                })
            }
            restored.ngrams.forEach { row ->
                insertOrThrow("ngram", null, ContentValues().apply {
                    put("prefix", row.prefix); put("continuation", row.continuation)
                    put("language", row.language); put("order_n", row.order)
                    put("count", row.count); put("last_used", row.lastUsed)
                })
            }
            restored.rejections.forEach { row ->
                insertOrThrow("rejection", null, ContentValues().apply {
                    put("raw", row.raw); put("replacement", row.replacement)
                    put("language", row.language); put("context_hash", row.contextHash)
                    put("rejected_at", row.rejectedAt); put("manual_accepts", row.manualAccepts)
                })
            }
        }
    }

    override fun wipe() {
        writableDatabase.transaction {
            delete("unigram", null, null)
            delete("ngram", null, null)
            delete("rejection", null, null)
        }
    }

    private fun incrementUnigram(
        db: SQLiteDatabase,
        word: String,
        surface: String,
        language: String,
        timestamp: Long,
    ) {
        val updated = db.execUpdate(
            "UPDATE unigram SET count = count + 1, last_used = ?, surface = ? WHERE word = ? AND language = ?",
            arrayOf(timestamp, surface, word, language),
        )
        if (updated == 0) {
            db.insertOrThrow("unigram", null, ContentValues().apply {
                put("word", word); put("surface", surface); put("language", language)
                put("count", 1); put("last_used", timestamp)
            })
        }
    }

    private fun preferredSurface(db: SQLiteDatabase, word: String, language: String): String =
        db.query(
            "unigram", arrayOf("surface"), "word = ? AND language = ?", arrayOf(word, language),
            null, null, null, "1",
        ).use { if (it.moveToFirst()) it.getString(0) ?: word else word }

    private fun incrementNgram(
        db: SQLiteDatabase,
        prefix: String,
        continuation: String,
        language: String,
        order: Int,
        timestamp: Long,
    ) {
        val updated = db.execUpdate(
            "UPDATE ngram SET count = count + 1, last_used = ? WHERE prefix = ? AND continuation = ? AND language = ? AND order_n = ?",
            arrayOf(timestamp, prefix, continuation, language, order),
        )
        if (updated == 0) {
            db.insertOrThrow("ngram", null, ContentValues().apply {
                put("prefix", prefix); put("continuation", continuation); put("language", language)
                put("order_n", order); put("count", 1); put("last_used", timestamp)
            })
        }
    }

    private fun decrementOrDelete(
        db: SQLiteDatabase,
        table: String,
        selection: String,
        arguments: Array<String>,
    ) {
        val count = db.query(table, arrayOf("count"), selection, arguments, null, null, null, "1").use {
            if (it.moveToFirst()) it.getLong(0) else return
        }
        if (count <= 1) db.delete(table, selection, arguments)
        else db.execSQL("UPDATE $table SET count = count - 1 WHERE $selection", arguments)
    }

    private fun recordManualAcceptance(
        db: SQLiteDatabase,
        raw: String,
        replacement: String,
        language: String,
        contextHash: String,
    ) {
        db.execSQL(
            "UPDATE rejection SET manual_accepts = manual_accepts + 1 WHERE raw = ? AND replacement = ? AND language = ? AND context_hash = ?",
            arrayOf(raw, replacement, language, contextHash),
        )
    }

    private fun rejectionPenalty(
        db: SQLiteDatabase,
        raw: String,
        replacement: String,
        language: String,
        context: String,
    ): Double {
        if (raw.isBlank() || raw == replacement) return 0.0
        val cutoff = System.currentTimeMillis() - REJECTION_RETENTION_MILLIS
        val fingerprint = fingerprintContext(context)
        db.query(
            "rejection", arrayOf("manual_accepts"),
            "raw = ? AND replacement = ? AND language = ? AND context_hash = ? AND rejected_at >= ?",
            arrayOf(raw, replacement, language, fingerprint, cutoff.toString()),
            null, null, null, "1",
        ).use { return if (it.moveToFirst() && it.getInt(0) < REQUIRED_MANUAL_ACCEPTS) 2.0 else 0.0 }
    }

    private fun deleteExpiredRejections(db: SQLiteDatabase, now: Long) {
        db.delete("rejection", "rejected_at < ?", arrayOf((now - REJECTION_RETENTION_MILLIS).toString()))
    }

    private fun SQLiteDatabase.execUpdate(sql: String, arguments: Array<Any>): Int {
        compileStatement(sql).use { statement ->
            arguments.forEachIndexed { index, value ->
                when (value) {
                    is Long -> statement.bindLong(index + 1, value)
                    is Int -> statement.bindLong(index + 1, value.toLong())
                    else -> statement.bindString(index + 1, value.toString())
                }
            }
            return statement.executeUpdateDelete()
        }
    }

    private fun decayedCount(count: Long, lastUsed: Long, now: Long): Double {
        val age = (now - lastUsed).coerceAtLeast(0)
        return count * exp(-age.toDouble() / DECAY_TIME_CONSTANT_MILLIS)
    }

    private fun escapeLike(value: String): String = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    companion object {
        private const val DATABASE_NAME = "libreboard_personal.db"
        private const val DATABASE_VERSION = 2
        private const val EXPORT_SCHEMA_VERSION = 2
        private const val MAX_EXPORT_BYTES = 16 * 1024 * 1024
        private const val MAX_EXPORT_ROWS = 100_000
        private const val MAX_PHRASE_TOKENS = 4
        private const val MAX_PERSONAL_SWIPE_QUERY_ROWS = 512
        private const val MAX_PERSONAL_WORD_LENGTH = 64
        private const val SWIPE_LENGTH_TOLERANCE_BELOW = 3
        private const val SWIPE_LENGTH_TOLERANCE_ABOVE = 4
        private const val REQUIRED_MANUAL_ACCEPTS = 2
        private const val REJECTION_RETENTION_MILLIS = 30L * 24 * 60 * 60 * 1000
        private const val DECAY_TIME_CONSTANT_MILLIS = 90.0 * 24 * 60 * 60 * 1000
        private const val TOKEN_SEPARATOR = "\u001f"
        private val JSON = Json { encodeDefaults = true }

        private const val CREATE_UNIGRAM = """
            CREATE TABLE unigram (
                word TEXT NOT NULL,
                surface TEXT NOT NULL,
                language TEXT NOT NULL,
                count INTEGER NOT NULL,
                last_used INTEGER NOT NULL,
                PRIMARY KEY (word, language)
            )
        """
        private const val CREATE_NGRAM = """
            CREATE TABLE ngram (
                prefix TEXT NOT NULL,
                continuation TEXT NOT NULL,
                language TEXT NOT NULL,
                order_n INTEGER NOT NULL CHECK (order_n BETWEEN 2 AND 4),
                count INTEGER NOT NULL,
                last_used INTEGER NOT NULL,
                PRIMARY KEY (prefix, continuation, language, order_n)
            )
        """
        private const val CREATE_REJECTION = """
            CREATE TABLE rejection (
                raw TEXT NOT NULL,
                replacement TEXT NOT NULL,
                language TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                rejected_at INTEGER NOT NULL,
                manual_accepts INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (raw, replacement, language, context_hash)
            )
        """

        fun openOrNull(context: Context): SqlitePersonalStore? =
            CredentialEncryptedStorage.contextOrNull(context)?.let(::SqlitePersonalStore)

        fun tokenize(text: String): List<String> = text
            .let(::tokenizeSurfaces)
            .map(::normalizeCandidate)

        fun tokenizeSurfaces(text: String): List<String> = text
            .split(Regex("[^\\p{L}\\p{N}'’]+"))
            .filter(String::isNotBlank)

        private fun swipeEmissionLength(word: String): Int {
            var count = 0
            var index = 0
            while (index < word.length) {
                val codePoint = word.codePointAt(index)
                if (codePoint != '\''.code && codePoint != 0x2019 && codePoint != '-'.code) count++
                index += Character.charCount(codePoint)
            }
            return count
        }

        /** Stores only a one-way fingerprint, never the surrounding prose. */
        fun fingerprintContext(context: String): String {
            val bounded = tokenize(context).takeLast(8).joinToString(TOKEN_SEPARATOR)
            return MessageDigest.getInstance("SHA-256").digest(bounded.encodeToByteArray())
                .joinToString("") { "%02x".format(it) }
        }
    }
}

private fun PersonalExport.validate() {
    fun validToken(value: String, maxLength: Int = 256) = value.isNotBlank() && value.length <= maxLength
    require(unigrams.all {
        validToken(it.word) && validToken(it.surface) && normalizeCandidate(it.surface) == it.word &&
            validToken(it.language, 64) && it.count >= 0 && it.lastUsed >= 0
    })
    require(ngrams.all {
        validToken(it.prefix, 1024) && validToken(it.continuation) && validToken(it.language, 64) &&
            it.order in 2..4 && it.count >= 0 && it.lastUsed >= 0
    })
    require(rejections.all {
        validToken(it.raw) && validToken(it.replacement) && validToken(it.language, 64) &&
            it.contextHash.matches(Regex("[0-9a-f]{64}")) && it.rejectedAt >= 0 && it.manualAccepts >= 0
    })
}

@Serializable
private data class PersonalExport(
    val schemaVersion: Int,
    val unigrams: List<ExportUnigram>,
    val ngrams: List<ExportNgram>,
    val rejections: List<ExportRejection>,
)

@Serializable private data class ExportUnigram(
    val word: String,
    val language: String,
    val count: Long,
    val lastUsed: Long,
    val surface: String = word,
)
@Serializable private data class ExportNgram(
    val prefix: String,
    val continuation: String,
    val language: String,
    val order: Int,
    val count: Long,
    val lastUsed: Long,
)
@Serializable private data class ExportRejection(
    val raw: String,
    val replacement: String,
    val language: String,
    val contextHash: String,
    val rejectedAt: Long,
    val manualAccepts: Int,
)
