/*
 * SPDX-License-Identifier: GPL-3.0-only
 */
package helium314.keyboard.latin.suggestions

import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.dictionary.Dictionary
import kotlin.test.Test
import kotlin.test.assertEquals

class RawSuggestionLayoutTest {
    @Test
    fun `raw tap input stays left while the best candidate stays centered`() {
        assertEquals(
            0,
            SuggestionStripLayoutHelper.getPositionInSuggestionStrip(
                SuggestedWords.INDEX_OF_TYPED_WORD,
                false,
                true,
                true,
                1,
                0,
            ),
        )
        assertEquals(
            1,
            SuggestionStripLayoutHelper.getPositionInSuggestionStrip(
                SuggestedWords.INDEX_OF_AUTO_CORRECTION,
                false,
                true,
                true,
                1,
                0,
            ),
        )
    }

    @Test
    fun `visible word count includes literal tap input`() {
        val raw = word("thsi", SuggestedWordInfo.KIND_TYPED)
        val correction = word("this", SuggestedWordInfo.KIND_CORRECTION)
        val words = SuggestedWords(
            arrayListOf(raw, correction),
            null,
            raw,
            false,
            false,
            false,
            SuggestedWords.INPUT_STYLE_TYPING,
            1,
        )

        assertEquals(2, words.wordCountToShow)
    }

    private fun word(value: String, kind: Int) = SuggestedWordInfo(
        value,
        "",
        1,
        kind,
        Dictionary.DICTIONARY_USER_TYPED,
        SuggestedWordInfo.NOT_AN_INDEX,
        SuggestedWordInfo.NOT_A_CONFIDENCE,
    )
}
