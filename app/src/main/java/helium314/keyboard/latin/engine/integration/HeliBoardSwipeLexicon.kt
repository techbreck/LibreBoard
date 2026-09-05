// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.latin.DictionaryFacilitator
import helium314.keyboard.latin.engine.FieldPolicy
import helium314.keyboard.latin.engine.geometric.LexiconWord
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import helium314.keyboard.latin.engine.normalizeCandidate
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.settings.Settings
import java.util.Locale

internal data class SwipeLexiconRuntimePolicy(
    val usePersonalizedWords: Boolean,
    val blockPossiblyOffensive: Boolean,
    val fieldPolicy: FieldPolicy,
    val incognito: Boolean,
)

/**
 * Data-only bridge from retained AOSP dictionaries and LibreBoard's CE personal store into both
 * open swipe decoders. Static results are cached by immutable dictionary revision and request
 * shape; personal words remain live and are read only when the current privacy policy permits it.
 */
internal class HeliBoardSwipeLexicon(
    private val facilitator: DictionaryFacilitator,
    private val policyProvider: () -> SwipeLexiconRuntimePolicy = {
        val values = Settings.getValues()
        SwipeLexiconRuntimePolicy(
            usePersonalizedWords = values.mUsePersonalizedDicts,
            blockPossiblyOffensive = values.mBlockPotentiallyOffensive,
            fieldPolicy = values.mInputAttributes.mFieldPolicy,
            incognito = values.mIncognitoModeEnabled,
        )
    },
    private val personalProvider: (List<String>, Int, Int, SwipeLexiconRuntimePolicy) -> List<LexiconWord> =
        { languages, length, maximum, policy ->
            PersonalizationRuntime.swipeLexicon(
                Settings.getCurrentContext(),
                languages,
                length,
                maximum,
                policy.fieldPolicy,
                policy.incognito,
            )
        },
    private val personalRevisionProvider: () -> Long = PersonalizationRuntime::swipeLexiconRevision,
) : SwipeLexicon {
    @Volatile private var cache: Cache? = null

    override fun words(languageTags: List<String>, approximateLength: Int): Sequence<LexiconWord> {
        val languages = languageTags.asSequence()
            .map(::canonicalLanguageTag)
            .filter(String::isNotBlank)
            .distinct()
            .toList()
        if (languages.isEmpty()) return emptySequence()
        val policy = policyProvider()
        val personalAllowed = policy.usePersonalizedWords && policy.fieldPolicy.allowsSuggestions &&
            policy.fieldPolicy.allowsPersistence && !policy.incognito
        val key = CacheKey(
            facilitator.swipeLexiconRevision,
            if (personalAllowed) personalRevisionProvider() else -1L,
            languages,
            approximateLength,
            policy.blockPossiblyOffensive,
            personalAllowed,
        )
        cache?.takeIf { it.key == key }?.let { return it.words.asSequence() }
        val words = synchronized(this) {
            cache?.takeIf { it.key == key }?.let { return@synchronized it.words }
            val staticWords = facilitator.getSwipeLexiconWords(
                languages,
                approximateLength,
                MAX_STATIC_WORDS,
                policy.blockPossiblyOffensive,
            )
            val personalWords = if (personalAllowed) {
                personalProvider(languages, approximateLength, MAX_PERSONAL_WORDS, policy)
            } else emptyList()
            (personalWords.asSequence() + staticWords.asSequence())
                .map { word ->
                    val language = canonicalLanguageTag(word.languageTag)
                    if (language == word.languageTag) word else word.copy(languageTag = language)
                }
                .distinctBy { normalizeCandidate(it.word) to canonicalLanguageTag(it.languageTag) }
                .toList()
                .also { cache = Cache(key, it) }
        }
        return words.asSequence()
    }

    private data class CacheKey(
        val staticRevision: Long,
        val personalRevision: Long,
        val languages: List<String>,
        val approximateLength: Int,
        val blockPossiblyOffensive: Boolean,
        val personalAllowed: Boolean,
    )

    private data class Cache(val key: CacheKey, val words: List<LexiconWord>)

    private companion object {
        const val MAX_STATIC_WORDS = 100_000
        const val MAX_PERSONAL_WORDS = 256

        fun canonicalLanguageTag(tag: String): String =
            Locale.forLanguageTag(tag).takeUnless { it == Locale.ROOT }?.toLanguageTag().orEmpty()
    }
}
