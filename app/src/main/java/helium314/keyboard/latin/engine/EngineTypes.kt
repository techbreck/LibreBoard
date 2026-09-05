// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import java.text.Normalizer
import kotlin.math.max

const val MAX_CONTEXT_CHARS = 256
const val MAX_CONTEXT_TOKENS = 8
const val MAX_CONTEXT_WORDPIECES = 32
const val MAX_CANDIDATES = 32

enum class InputStyle { TAP, SWIPE, PREDICTION }

enum class FieldClass { PLAIN, SHORT_MESSAGE, SEARCH, CODE_OR_TERMINAL, RESTRICTED }

data class TouchPoint(val x: Float, val y: Float, val timeMillis: Long = 0)

data class KeySlot(
    val id: Int,
    val label: String,
    val centerX: Float,
    val centerY: Float,
    val width: Float,
    val height: Float,
    val enabled: Boolean = true,
)

data class KeyGeometry(
    val width: Float,
    val height: Float,
    val keys: List<KeySlot>,
) {
    init {
        require(width > 0 && height > 0) { "Keyboard geometry must have positive dimensions" }
        require(keys.size <= 64) { "The v1 swipe ABI supports at most 64 key slots" }
    }

    fun keyFor(character: Char): KeySlot? = keys.firstOrNull {
        it.enabled && it.label.codePointCount(0, it.label.length) == 1 &&
            it.label.equals(character.toString(), ignoreCase = true)
    }
}

sealed interface WordLock {
    data object Unlocked : WordLock
    data class Automatic(val languageTag: String) : WordLock
    data class Manual(val languageTag: String) : WordLock
}

data class TypingRequest(
    val rawText: String,
    val path: List<TouchPoint>,
    val precedingContext: String,
    val geometry: KeyGeometry,
    val enabledLanguages: List<String>,
    val wordLock: WordLock,
    val fieldPolicy: FieldPolicy,
    val fieldClass: FieldClass,
    val inputStyle: InputStyle,
    val sequenceId: Long,
) {
    init {
        require(enabledLanguages.isNotEmpty()) { "At least one language must be enabled" }
        require(precedingContext.length <= MAX_CONTEXT_CHARS) { "Context must be bounded before dispatch" }
        require(path.size <= 1024) { "Raw pointer history must be bounded" }
    }

    companion object {
        fun bounded(
            rawText: String,
            path: List<TouchPoint> = emptyList(),
            precedingContext: CharSequence? = null,
            geometry: KeyGeometry,
            enabledLanguages: List<String>,
            wordLock: WordLock = WordLock.Unlocked,
            fieldPolicy: FieldPolicy,
            fieldClass: FieldClass = FieldClass.PLAIN,
            inputStyle: InputStyle,
            sequenceId: Long,
        ) = TypingRequest(
            rawText = rawText,
            path = path.takeLast(1024),
            precedingContext = precedingContext?.toString().orEmpty().takeLast(MAX_CONTEXT_CHARS),
            geometry = geometry,
            enabledLanguages = enabledLanguages.distinct(),
            wordLock = wordLock,
            fieldPolicy = fieldPolicy,
            fieldClass = fieldClass,
            inputStyle = inputStyle,
            sequenceId = sequenceId,
        )
    }
}

enum class CandidateSource {
    RAW,
    STATIC_DICTIONARY,
    SPATIAL,
    PERSONAL,
    PERSONAL_PHRASE,
    CONTRACTION,
    COMPOUND,
    SPLIT_JOIN,
    HOMOPHONE,
    CTC_SWIPE,
    GEOMETRIC_SWIPE,
    NEXT_WORD,
}

enum class EditOperation { INSERT, DELETE, SUBSTITUTE, TRANSPOSE, SPLIT, JOIN, APOSTROPHE }

data class ScoreComponents(
    val spatial: Double? = null,
    val staticFrequency: Double? = null,
    val personal: Double? = null,
    val context: Double? = null,
    val language: Double? = null,
)

data class Candidate(
    val surface: String,
    val normalized: String = normalizeCandidate(surface),
    val languageTag: String,
    val sources: Set<CandidateSource>,
    val components: ScoreComponents = ScoreComponents(),
    val edits: List<EditOperation> = emptyList(),
    val exactPersonalMatch: Boolean = false,
    val rejectionPenalty: Double = 0.0,
    val totalScore: Double = Double.NEGATIVE_INFINITY,
    val calibratedProbability: Double = 0.0,
)

enum class EngineAvailability { AVAILABLE, DISABLED, UNAVAILABLE, TIMEOUT, INCOMPATIBLE, CIRCUIT_OPEN }

data class SuggestionBatch(
    val sequenceId: Long,
    val candidates: List<Candidate>,
    val autoCorrection: Candidate?,
    val neuralAvailability: EngineAvailability,
    val completedAtNanos: Long = System.nanoTime(),
) {
    init {
        require(candidates.size <= MAX_CANDIDATES)
        require(autoCorrection == null || autoCorrection in candidates)
    }
}

fun normalizeCandidate(value: String): String = Normalizer.normalize(value, Normalizer.Form.NFKC).lowercase()

class Deadline private constructor(
    private val expiresAtNanos: Long,
    private val nanoTime: () -> Long,
) {
    val expired: Boolean get() = nanoTime() >= expiresAtNanos
    val remainingMillis: Long get() = max(0L, (expiresAtNanos - nanoTime()) / 1_000_000L)

    companion object {
        fun afterMillis(durationMillis: Long, nanoTime: () -> Long = System::nanoTime): Deadline {
            require(durationMillis >= 0)
            return Deadline(nanoTime() + durationMillis * 1_000_000L, nanoTime)
        }
    }
}
