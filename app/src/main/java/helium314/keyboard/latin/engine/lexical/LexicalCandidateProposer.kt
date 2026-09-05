// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.lexical

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EditOperation
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.Locale

/**
 * Produces a deliberately small set of explainable lexical transformations. Dictionary checks
 * are language-specific so a multilingual slate can never manufacture a cross-language phrase.
 */
class LexicalCandidateProposer(
    private val isValidWord: (word: String, languageTag: String) -> Boolean,
) {
    fun propose(
        rawText: String,
        enabledLanguageTags: List<String>,
        wordLock: WordLock,
        deadline: Deadline,
    ): List<Candidate> {
        if (rawText.isBlank() || deadline.expired) return emptyList()
        if (rawText.codePointCount(0, rawText.length) > MAX_RAW_CODEPOINTS) return emptyList()
        val languages = allowedLanguages(enabledLanguageTags, wordLock)
        if (languages.isEmpty()) return emptyList()

        val candidates = ArrayList<Candidate>(languages.size * 2)
        for (languageTag in languages) {
            if (deadline.expired) break
            contraction(rawText, languageTag)?.let(candidates::add)
            if (deadline.expired) break
            if (' ' in rawText) {
                joined(rawText, languageTag)?.let(candidates::add)
            } else if (primaryLanguage(languageTag) == GERMAN_LANGUAGE) {
                germanCompound(rawText, languageTag, deadline)?.let(candidates::add)
            } else {
                missedSpace(rawText, languageTag, deadline)?.let(candidates::add)
            }
        }
        return candidates
            .distinctBy { it.normalized to it.languageTag }
            .take(MAX_PROPOSALS)
    }

    private fun contraction(rawText: String, languageTag: String): Candidate? {
        if (primaryLanguage(languageTag) != ENGLISH_LANGUAGE || ' ' in rawText) return null
        val canonical = ENGLISH_CONTRACTIONS[normalizeCandidate(rawText)] ?: return null
        val surface = applyCasing(rawText, canonical)
        if (!isValidWord(surface, languageTag) && !isValidWord(canonical, languageTag)) return null
        return Candidate(
            surface = surface,
            languageTag = languageTag,
            sources = setOf(CandidateSource.CONTRACTION),
            components = ScoreComponents(staticFrequency = GENERATED_EVIDENCE),
            edits = listOf(EditOperation.APOSTROPHE),
        )
    }

    private fun missedSpace(rawText: String, languageTag: String, deadline: Deadline): Candidate? {
        if (rawText.length < MIN_SPLIT_LENGTH || rawText.any { !it.isLetter() }) return null
        var bestSurface: String? = null
        var bestImbalance = Int.MAX_VALUE
        for (index in 1 until rawText.length) {
            if (deadline.expired) break
            if (Character.isLowSurrogate(rawText[index])) continue
            val left = rawText.substring(0, index)
            val right = rawText.substring(index)
            if (!validFragment(left, languageTag) || !validFragment(right, languageTag)) continue
            // Prefer balanced phrases and then the leftmost split for deterministic behavior.
            val imbalance = kotlin.math.abs(left.length - right.length)
            if (imbalance < bestImbalance) {
                bestSurface = "$left $right"
                bestImbalance = imbalance
            }
        }
        val surface = bestSurface ?: return null
        return Candidate(
            surface = surface,
            languageTag = languageTag,
            sources = setOf(CandidateSource.SPLIT_JOIN),
            components = ScoreComponents(staticFrequency = GENERATED_EVIDENCE),
            edits = listOf(EditOperation.SPLIT),
        )
    }

    private fun joined(rawText: String, languageTag: String): Candidate? {
        if (rawText.count { it == ' ' } != 1 || rawText != rawText.trim()) return null
        val parts = rawText.split(' ')
        if (parts.size != 2 || parts.any { it.isEmpty() || it.any { character -> !character.isLetter() } }) return null
        val joined = parts.joinToString(separator = "")
        if (!isValidWord(joined, languageTag)) return null
        return Candidate(
            surface = joined,
            languageTag = languageTag,
            sources = setOf(CandidateSource.SPLIT_JOIN),
            components = ScoreComponents(staticFrequency = GENERATED_EVIDENCE),
            edits = listOf(EditOperation.JOIN),
        )
    }

    private fun germanCompound(rawText: String, languageTag: String, deadline: Deadline): Candidate? {
        if (rawText.length < MIN_COMPOUND_LENGTH || rawText.any { !it.isLetter() }) return null
        for (index in MIN_COMPOUND_PART_LENGTH..rawText.length - MIN_COMPOUND_PART_LENGTH) {
            if (deadline.expired) return null
            if (Character.isLowSurrogate(rawText[index])) continue
            val left = rawText.substring(0, index)
            val right = rawText.substring(index)
            if (isValidWord(left, languageTag) && isValidWord(right, languageTag)) {
                // The candidate intentionally keeps the original surface. This reinforces the
                // unsplit form without ever offering an automatic German split.
                return Candidate(
                    surface = rawText,
                    languageTag = languageTag,
                    sources = setOf(CandidateSource.COMPOUND),
                    components = ScoreComponents(staticFrequency = GENERATED_EVIDENCE),
                )
            }
        }
        return null
    }

    private fun validFragment(fragment: String, languageTag: String): Boolean =
        fragment.length >= MIN_FRAGMENT_LENGTH && isValidWord(fragment, languageTag)

    private fun allowedLanguages(enabled: List<String>, lock: WordLock): List<String> {
        val canonical = enabled.map(Locale::forLanguageTag)
            .filter { it.language.isNotBlank() }
            .map(Locale::toLanguageTag)
            .distinct()
        val locked = when (lock) {
            WordLock.Unlocked -> null
            is WordLock.Automatic -> Locale.forLanguageTag(lock.languageTag).toLanguageTag()
            is WordLock.Manual -> Locale.forLanguageTag(lock.languageTag).toLanguageTag()
        }
        return locked?.let { language -> canonical.filter { it.equals(language, ignoreCase = true) } } ?: canonical
    }

    private fun primaryLanguage(languageTag: String): String = Locale.forLanguageTag(languageTag).language

    private fun applyCasing(raw: String, canonical: String): String = when {
        raw.all(Char::isUpperCase) -> canonical.uppercase(Locale.ENGLISH)
        raw.firstOrNull()?.isUpperCase() == true -> canonical.replaceFirstChar { it.titlecase(Locale.ENGLISH) }
        else -> canonical
    }

    private companion object {
        const val ENGLISH_LANGUAGE = "en"
        const val GERMAN_LANGUAGE = "de"
        const val MAX_RAW_CODEPOINTS = 64
        const val MAX_PROPOSALS = 4
        const val MIN_FRAGMENT_LENGTH = 2
        const val MIN_SPLIT_LENGTH = MIN_FRAGMENT_LENGTH * 2
        const val MIN_COMPOUND_PART_LENGTH = 3
        const val MIN_COMPOUND_LENGTH = MIN_COMPOUND_PART_LENGTH * 2
        const val GENERATED_EVIDENCE = 1.0

        val ENGLISH_CONTRACTIONS = mapOf(
            "arent" to "aren't",
            "cant" to "can't",
            "couldnt" to "couldn't",
            "didnt" to "didn't",
            "doesnt" to "doesn't",
            "dont" to "don't",
            "hadnt" to "hadn't",
            "hasnt" to "hasn't",
            "havent" to "haven't",
            "hed" to "he'd",
            "hell" to "he'll",
            "hes" to "he's",
            "id" to "I'd",
            "ill" to "I'll",
            "im" to "I'm",
            "isnt" to "isn't",
            "ive" to "I've",
            "mustnt" to "mustn't",
            "shouldnt" to "shouldn't",
            "wasnt" to "wasn't",
            "werent" to "weren't",
            "wont" to "won't",
            "wouldnt" to "wouldn't",
            "youre" to "you're",
            "youve" to "you've",
        )
    }
}
