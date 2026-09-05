// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.personal

import android.content.Context
import android.view.inputmethod.EditorInfo
import helium314.keyboard.latin.engine.CommitObservation
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.FieldClass
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.FieldPolicyResolver
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.RejectionObservation
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
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
        val committedSurfaces = SqlitePersonalStore.tokenizeSurfaces(committedWord).takeLast(4)
        if (committedSurfaces.isEmpty()) return
        val committedTokens = committedSurfaces.map(::normalizeCandidate)
        val personalStore = open(context) ?: return
        val precedingTokens = synchronized(sessionLock) { sessionCommittedTokens.takeLast(3) }
        val contextFingerprint = SqlitePersonalStore.fingerprintContext(precedingTokens.joinToString(" "))
        personalStore.observeCommit(CommitObservation(
            tokens = committedSurfaces,
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

    /**
     * Read personal candidates only for a policy that permits both suggestions and persistence.
     * The supplied context must already be a bounded, policy-approved snapshot; this facade never
     * reaches back into the editor or accepts an application identifier.
     */
    @JvmStatic
    fun suggest(
        context: Context,
        rawText: String,
        precedingContext: String,
        enabledLanguageTags: List<String>,
        fieldPolicy: FieldPolicy,
        incognito: Boolean,
        inputStyle: InputStyle,
        sequenceId: Long,
    ): List<Candidate> {
        if (incognito || !fieldPolicy.allowsSuggestions || !fieldPolicy.allowsPersistence) return emptyList()
        val languages = enabledLanguageTags.filter(String::isNotBlank).distinct().take(MAX_PERSONAL_LANGUAGES)
        if (languages.isEmpty()) return emptyList()
        val request = TypingRequest.bounded(
            rawText = rawText,
            precedingContext = precedingContext,
            geometry = EMPTY_GEOMETRY,
            enabledLanguages = languages,
            wordLock = WordLock.Unlocked,
            fieldPolicy = fieldPolicy,
            fieldClass = FieldClass.PLAIN,
            inputStyle = inputStyle,
            sequenceId = sequenceId,
        )
        return open(context)?.suggest(request, Deadline.afterMillis(PERSONAL_QUERY_BUDGET_MILLIS)).orEmpty()
    }

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
    private const val MAX_PERSONAL_LANGUAGES = 8
    private const val PERSONAL_QUERY_BUDGET_MILLIS = 15L
    private val EMPTY_GEOMETRY = KeyGeometry(1f, 1f, emptyList())
}
