// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

data class LanguageEvidence(
    val probabilities: Map<String, Double>,
    val typedCharacterCount: Int,
)

class LanguageLockController(private val lockMargin: Double = 0.25) {
    private var hardLanguage: String? = null
    private var current: WordLock = WordLock.Unlocked

    fun current(): WordLock = current

    fun selectManually(languageTag: String) {
        hardLanguage = languageTag
        current = WordLock.Manual(languageTag)
    }

    fun releaseManualSelection() {
        hardLanguage = null
        current = WordLock.Unlocked
    }

    fun beginWord(): WordLock {
        current = hardLanguage?.let { WordLock.Manual(it) } ?: WordLock.Unlocked
        return current
    }

    fun observe(evidence: LanguageEvidence): WordLock {
        if (current is WordLock.Manual || current is WordLock.Automatic) return current
        if (evidence.probabilities.isEmpty()) return current
        val ordered = evidence.probabilities.entries.sortedByDescending { it.value }
        val lead = ordered.first()
        val runnerUp = ordered.getOrNull(1)?.value ?: 0.0
        if (evidence.typedCharacterCount >= 3 || lead.value - runnerUp >= lockMargin) {
            current = WordLock.Automatic(lead.key)
        }
        return current
    }

    fun endWord() {
        current = hardLanguage?.let { WordLock.Manual(it) } ?: WordLock.Unlocked
    }
}
