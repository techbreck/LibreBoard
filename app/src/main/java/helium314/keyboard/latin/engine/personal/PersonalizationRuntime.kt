// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.personal

import android.content.Context
import android.view.inputmethod.EditorInfo
import helium314.keyboard.latin.engine.CommitObservation
import helium314.keyboard.latin.engine.FieldPolicyResolver
import helium314.keyboard.latin.engine.RejectionObservation
import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.concurrent.ConcurrentHashMap

/** Process-local facade used by the retained Java input logic. */
object PersonalizationRuntime {
    @Volatile private var store: SqlitePersonalStore? = null
    private val sessionRejections = ConcurrentHashMap.newKeySet<String>()
    private val sessionLock = Any()
    private val sessionCommittedTokens = ArrayDeque<String>()

    @JvmStatic
    fun observeCommit(
        context: Context,
        editorInfo: EditorInfo?,
        incognito: Boolean,
        committedWord: String,
        languageTag: String,
        manualSelection: Boolean,
        correctionRaw: String? = null,
    ) {
        if (incognito || !FieldPolicyResolver.resolve(editorInfo).allowsPersistence) return
        val committedTokens = SqlitePersonalStore.tokenize(committedWord).takeLast(4)
        if (committedTokens.isEmpty()) return
        val personalStore = open(context) ?: return
        val precedingTokens = synchronized(sessionLock) { sessionCommittedTokens.takeLast(3) }
        val contextFingerprint = SqlitePersonalStore.fingerprintContext(precedingTokens.joinToString(" "))
        personalStore.observeCommit(CommitObservation(
            tokens = committedTokens,
            languageTag = languageTag,
            timestampMillis = System.currentTimeMillis(),
            wasManualSelection = manualSelection,
            correctionRaw = correctionRaw,
            contextFingerprint = contextFingerprint,
            precedingTokens = precedingTokens,
        ))
        synchronized(sessionLock) {
            committedTokens.forEach(sessionCommittedTokens::addLast)
            while (sessionCommittedTokens.size > MAX_SESSION_TOKENS) sessionCommittedTokens.removeFirst()
        }
    }

    @JvmStatic
    fun observeRejection(
        context: Context,
        editorInfo: EditorInfo?,
        incognito: Boolean,
        raw: String,
        replacement: String,
        languageTag: String,
    ) {
        if (incognito || !FieldPolicyResolver.resolve(editorInfo).allowsPersistence) return
        val normalizedRaw = normalizeCandidate(raw)
        val normalizedReplacement = normalizeCandidate(replacement)
        if (normalizedRaw.isBlank() || normalizedReplacement.isBlank() || normalizedRaw == normalizedReplacement) return
        val replacementTokens = SqlitePersonalStore.tokenize(replacement).takeLast(4)
        val sessionSnapshot = synchronized(sessionLock) { sessionCommittedTokens.toList() }
        val isLatestSessionCommit = replacementTokens.isNotEmpty() && sessionSnapshot.endsWith(replacementTokens)
        // Rejection equivalence is derived only from tokens this IME observed committing. Never
        // recover context from the editor during a rejection, even if policy resolution changes
        // between input events. An unreconciled rejection gets the stable empty-context hash.
        val precedingTokens = if (isLatestSessionCommit) {
            sessionSnapshot.dropLast(replacementTokens.size).takeLast(3)
        } else emptyList()
        val contextHash = SqlitePersonalStore.fingerprintContext(precedingTokens.joinToString(" "))
        sessionRejections += rejectionKey(normalizedRaw, normalizedReplacement, languageTag)
        open(context)?.let { personalStore ->
            if (isLatestSessionCommit) {
                personalStore.compensateCommit(replacementTokens, precedingTokens, languageTag)
                synchronized(sessionLock) { repeat(replacementTokens.size) { sessionCommittedTokens.removeLastOrNull() } }
            }
            personalStore.observeRejection(RejectionObservation(
                normalizedRaw,
                normalizedReplacement,
                languageTag,
                contextHash,
                System.currentTimeMillis(),
            ))
        }
    }

    @JvmStatic
    fun isRejectedThisSession(raw: String, replacement: String, languageTag: String): Boolean =
        rejectionKey(normalizeCandidate(raw), normalizeCandidate(replacement), languageTag) in sessionRejections

    @JvmStatic
    fun isCorrectionSuppressed(
        context: Context,
        raw: String,
        replacement: String,
        languageTag: String,
    ): Boolean = isRejectedThisSession(raw, replacement, languageTag) ||
        (open(context)?.isCorrectionSuppressed(raw, replacement, languageTag) == true)

    @JvmStatic
    fun clearSession() {
        sessionRejections.clear()
        synchronized(sessionLock) { sessionCommittedTokens.clear() }
    }

    @JvmStatic
    fun wipe(context: Context) {
        open(context)?.wipe()
        clearSession()
    }

    @JvmStatic
    fun export(context: Context): ByteArray? = open(context)?.export()

    @JvmStatic
    fun restore(context: Context, payload: ByteArray) {
        val personalStore = open(context)
            ?: throw IllegalStateException("Credential-encrypted storage is unavailable")
        personalStore.restore(payload)
        clearSession()
    }

    private fun open(context: Context): SqlitePersonalStore? {
        store?.let { return it }
        return synchronized(this) {
            store ?: SqlitePersonalStore.openOrNull(context)?.also { store = it }
        }
    }

    private fun rejectionKey(raw: String, replacement: String, languageTag: String) =
        "$languageTag\u001f$raw\u001f$replacement"

    private fun <T> List<T>.endsWith(suffix: List<T>): Boolean =
        size >= suffix.size && subList(size - suffix.size, size) == suffix

    private const val MAX_SESSION_TOKENS = 8
}
