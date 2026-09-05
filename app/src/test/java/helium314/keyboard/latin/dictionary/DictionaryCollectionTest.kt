// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.dictionary

import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Locale

class DictionaryCollectionTest {
    @Test
    fun unigramEnumerationIsGloballyBoundedAndSkipsEmojiDictionary() {
        val first = EnumeratingDictionary(Dictionary.TYPE_MAIN, listOf("alpha", "beta"))
        val emoji = EnumeratingDictionary(Dictionary.TYPE_EMOJI, listOf("should-not-appear"))
        val second = EnumeratingDictionary(Dictionary.TYPE_MAIN, listOf("gamma", "delta"))
        val collection = DictionaryCollection(
            Dictionary.TYPE_MAIN,
            Locale.ENGLISH,
            listOf(first, emoji, second),
            floatArrayOf(1f, 1f, 1f),
        )
        val visited = mutableListOf<String>()

        val count = collection.visitUnigrams(3) { word, _, _, _ -> visited += word }

        assertEquals(3, count)
        assertEquals(listOf("alpha", "beta", "gamma"), visited)
        assertEquals(0, emoji.lastMaximum)
        assertEquals(1, second.lastMaximum)
    }

    private class EnumeratingDictionary(type: String, private val words: List<String>) :
        Dictionary(type, Locale.ENGLISH) {
        var lastMaximum = 0

        override fun visitUnigrams(maximumWords: Int, visitor: UnigramVisitor): Int {
            lastMaximum = maximumWords
            words.take(maximumWords).forEach { visitor.visit(it, 100, false, false) }
            return minOf(words.size, maximumWords)
        }

        override fun getSuggestions(
            composedData: ComposedData?,
            ngramContext: NgramContext?,
            proximityInfoHandle: Long,
            settingsValuesForSuggestion: SettingsValuesForSuggestion?,
            sessionId: Int,
            weightForLocale: Float,
            inOutWeightOfLangModelVsSpatialModel: FloatArray?,
        ): ArrayList<SuggestedWordInfo> = arrayListOf()

        override fun isInDictionary(word: String?): Boolean = false
    }
}
