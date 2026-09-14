// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin

import androidx.core.content.edit
import android.text.InputType
import android.view.inputmethod.EditorInfo
import helium314.keyboard.ShadowBinaryDictionaryUtils
import helium314.keyboard.ShadowInputMethodManager2
import helium314.keyboard.ShadowLocaleManagerCompat
import helium314.keyboard.event.Event
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.Key
import helium314.keyboard.keyboard.KeyboardElement
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_SHORTCUT
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo.KIND_WHITELIST
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.common.StringUtils
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.engine.AutoCorrectionAggressiveness
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.runtime.LiveTypingEngine
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.utils.SuggestionResults
import helium314.keyboard.latin.utils.prefs
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import org.robolectric.shadows.ShadowLog
import java.util.*
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

@Suppress("NonAsciiCharacters")
@RunWith(RobolectricTestRunner::class)
@Config(shadows = [
    ShadowLocaleManagerCompat::class,
    ShadowInputMethodManager2::class,
    ShadowBinaryDictionaryUtils::class,
    ShadowFacilitator::class,
])
class SuggestTest {
    private val latinIME = Robolectric.setupService(LatinIME::class.java)
    private val suggest get() = latinIME.mInputLogic.mSuggest

    private val confidenceModest = 0.24f
    private val confidenceAggressive = 0.65f
    private val confidenceVeryAggressive = 0.9f

    init {
        ShadowLog.setupLogging()
        ShadowLog.stream = System.out
    }

    @BeforeTest fun reset() {
        latinIME.prefs().edit { clear() }
        // Service construction has no active editor and now fails closed by design. Suggestion
        // tests explicitly install a normal text-field policy before changing correction prefs.
        Settings.getInstance().loadSettings(
            latinIME,
            Locale.ENGLISH,
            InputAttributes(EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }, false, latinIME.packageName),
        )
        currentTypingLocale = Locale.ENGLISH
        tapTypingSuggestions = suggestionResults(emptyList())
        glideTypingSuggestions = suggestionResults(emptyList())
        nextWordSuggestions = suggestionResults(emptyList())
    }

    @Test fun `'on' to 'in' if 'in' was used before in this context`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "on",
            listOf(suggestion("on", 1800000, locale), suggestion("in", 600000, locale)),
            suggestion("in", 240, locale),
            null, // never typed "on" in this context
            locale,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because first suggestion score is too low
    }

    @Test fun `live engine settings are bounded and invalid modes fail to balanced`() {
        latinIME.prefs().edit {
            putFloat(Settings.PREF_NEURAL_STRENGTH, 500f)
            putString(Settings.PREF_AUTO_CORRECTION_AGGRESSIVENESS, "not-a-mode")
        }
        Settings.getInstance().loadSettings(
            latinIME,
            Locale.ENGLISH,
            InputAttributes(EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }, false, latinIME.packageName),
        )
        assertEquals(100, Settings.getValues().mNeuralStrength)
        assertEquals(AutoCorrectionAggressiveness.BALANCED, Settings.getValues().mAutoCorrectionAggressiveness)

        latinIME.prefs().edit {
            putFloat(Settings.PREF_NEURAL_STRENGTH, -50f)
            putString(Settings.PREF_AUTO_CORRECTION_AGGRESSIVENESS, "AGGRESSIVE")
        }
        Settings.getInstance().loadSettings(
            latinIME,
            Locale.ENGLISH,
            InputAttributes(EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }, false, latinIME.packageName),
        )
        assertEquals(0, Settings.getValues().mNeuralStrength)
        assertEquals(AutoCorrectionAggressiveness.AGGRESSIVE, Settings.getValues().mAutoCorrectionAggressiveness)
    }

    @Test fun `calibrated engine recommendation bypasses only legacy threshold`() {
        val result = shouldBeAutoCorrected(
            "thsi",
            listOf(suggestion("this", 1, Locale.ENGLISH)),
            null,
            null,
            Locale.ENGLISH,
            confidenceModest,
            engineAutoCorrectionNormalized = "this",
        )

        assert(result.last())
    }

    @Test fun `calibrated joint evidence may correct a valid word despite legacy score gate`() {
        val result = shouldBeAutoCorrected(
            "their",
            listOf(
                suggestion("there", 1, Locale.ENGLISH),
                suggestion("their", 1_500_000, Locale.ENGLISH),
            ),
            null,
            null,
            Locale.ENGLISH,
            confidenceModest,
            engineAutoCorrectionNormalized = "there",
        )

        assert(result.last())
    }

    @Test fun `calibrated engine recommendation cannot bypass digit veto`() {
        val composer = WordComposer()
        StringUtils.toCodePointArray("th1s").forEach {
            val event = Event.createEventForCodePointFromAlreadyTypedText(
                it,
                Constants.NOT_A_COORDINATE,
                Constants.NOT_A_COORDINATE,
            )
            composer.applyProcessedEvent(composer.processEvent(event))
        }
        val result = shouldBeAutoCorrected(
            "th1s",
            listOf(suggestion("this", 1, Locale.ENGLISH)),
            null,
            null,
            Locale.ENGLISH,
            confidenceModest,
            engineAutoCorrectionNormalized = "this",
            wordComposer = composer,
        )

        assert(!result.last())
    }

    @Test fun `'ill' to 'I'll' if 'ill' not used before in this context, and I'll is whitelisted`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            null,
            null,
            locale,
            confidenceModest
        )
        assert(result.last()) // should be corrected
        // correction because both empty scores are 0, which should be fine (next check is comparing empty scores)
    }

    @Test fun `not 'ill' to 'I'll' if only 'ill' was used before in this context`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            null,
            suggestion("ill", 200, locale),
            locale,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because first empty score not high enough
    }

    @Test fun `'ill' to 'I'll' if both have same ngram score`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            suggestion("I'll", 200, locale),
            suggestion("ill", 200, locale),
            locale,
            confidenceModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `no 'ill' to 'I'll' if 'ill' has somewhat better ngram score`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "ill",
            listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale)),
            suggestion("I'll", 200, locale),
            suggestion("ill", 211, locale),
            locale,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `no English 'I' for Polish 'i' when typing in Polish`() {
        val result = shouldBeAutoCorrected(
            "i",
            listOf(suggestion("I", Int.MAX_VALUE, Locale.ENGLISH), suggestion("i", 1500000, Locale("pl"))),
            null,
            null,
            Locale("pl"),
            confidenceVeryAggressive
        )
        assert(!result.last()) // should not be corrected
        // not even checking at modest and aggressive thresholds, this is a locale thing
        // if very aggressive, still no correction because locale matches with typed word only
    }

    @Test fun `English 'I' instead of Polish 'i' when typing in English`() {
        val result = shouldBeAutoCorrected(
            "i",
            listOf(suggestion("I", Int.MAX_VALUE, Locale.ENGLISH), suggestion("i", 1500000, Locale("pl"))),
            null,
            null,
            Locale.ENGLISH,
            confidenceModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `no English 'in' instead of French 'un' when typing in French`() {
        val result = shouldBeAutoCorrected(
            "un",
            listOf(suggestion("in", Int.MAX_VALUE, Locale.ENGLISH), suggestion("un", 1500000, Locale.FRENCH)),
            null,
            null,
            Locale.FRENCH,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because of locale matching
    }

    @Test fun `no 'né' instead of 'ne'`() {
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, Locale.FRENCH), suggestion("né", 1900000-1, Locale.FRENCH)),
            null,
            null,
            Locale.FRENCH,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
        // not corrected because score is lower
    }

    @Test fun `'né' instead of 'ne' if 'né' in ngram context`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, locale), suggestion("né", 1900000-1, locale)),
            suggestion("né", 200, locale),
            null,
            locale,
            confidenceModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `'né' instead of 'ne' if 'né' has clearly better score in ngram context`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, locale), suggestion("né", 1900000-1, locale)),
            suggestion("né", 215, locale),
            suggestion("ne", 200, locale),
            locale,
            confidenceModest
        )
        assert(result.last()) // should be corrected
    }

    @Test fun `no 'né' instead of 'ne' if both with same score in ngram context`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "ne",
            listOf(suggestion("ne", 1900000, locale), suggestion("né", 1900000-1, locale)),
            suggestion("né", 200, locale),
            suggestion("ne", 200, locale),
            locale,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `no 'ne' instead of 'né'`() {
        val locale = Locale.FRENCH
        val result = shouldBeAutoCorrected(
            "né",
            listOf(suggestion("ne", 600000, locale), suggestion("né", 1600000, locale)),
            suggestion("né", 200, locale),
            suggestion("ne", 200, locale),
            locale,
            confidenceModest
        )
        assert(!result.last()) // should not be corrected
        // not even allowed to check because of low score for ne
    }

    @Test fun `shortcuts might be autocorrected by default`() {
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 700000, locale, true)),
            null,
            null,
            locale,
            confidenceAggressive
        )
        assert(result.last()) // should be corrected

        val result2 = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 300000, locale, true)),
            null,
            null,
            locale,
            confidenceModest
        )
        assert(!result2.last()) // should not be corrected
    }

    @Test fun `shortcuts are not autocorrected when setting is off`() {
        val prefs = latinIME.prefs()
        prefs.edit { putBoolean(Settings.PREF_AUTOCORRECT_SHORTCUTS, false) }
        val locale = Locale.ENGLISH
        val result = shouldBeAutoCorrected(
            "gd",
            listOf(suggestion("good", 12000000, locale, true)),
            null,
            null,
            locale,
            confidenceAggressive
        )
        assert(!result.last()) // should not be corrected
    }

    @Test fun `typed word is first suggestion`() { // first suggestion will not be shown to the user
        tapTypingSuggestions = suggestionResults(listOf(suggestion("hello", 123, currentTypingLocale)))
        val results = getSuggestedWords(false, "henlo", CapsMode.OFF)
        assertEquals(listOf("henlo", "hello"), results.mSuggestedWordInfoList.map { it.mWord })

        tapTypingSuggestions = suggestionResults(listOf(suggestion("hello", 123, currentTypingLocale)))
        val results2 = getSuggestedWords(false, "hello", CapsMode.OFF)
        assertEquals(listOf("hello"), results2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `typed word is second suggestion if autocorrect is pending`() {
        enableAutocorrect(confidenceVeryAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("hello", 650000, currentTypingLocale), // 600000 is the limit
            suggestion("hell", 620000, currentTypingLocale),
            suggestion("hem", 100000, currentTypingLocale),
        ))
        val results = getSuggestedWords(false, "henlo", CapsMode.OFF)
        assert(results.mWillAutoCorrect)
        assertEquals(5, results.mSuggestedWordInfoList.size)
        assertEquals("henlo", results.mSuggestedWordInfoList[0].mWord) // typed word, not shown
        assertEquals("hello", results.mSuggestedWordInfoList[1].mWord) // autocorrection is first suggestion
        assertEquals("henlo", results.mSuggestedWordInfoList[2].mWord) // if autocorrection is pending, typed word is next suggestion
        assertEquals("hell", results.mSuggestedWordInfoList[3].mWord)
        assertEquals("hem", results.mSuggestedWordInfoList[4].mWord)
    }

    @Test fun `CenterSuggestionTextToEnter has typed text or autocorrect as first suggestion`() {
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("surge", 1000),
            suggestion("sure", 935),
            suggestion("siege", 925),
        ))
        val results = getSuggestedWords(false, "suge", CapsMode.OFF)
        assertEquals(listOf("suge", "surge", "sure", "siege"), results.mSuggestedWordInfoList.map { it.mWord })

        latinIME.prefs().edit { putBoolean(Settings.PREF_CENTER_SUGGESTION_TEXT_TO_ENTER, true) }
        // first suggestion duplicated because the first element in the list is not shown to the user
        val results2 = getSuggestedWords(false, "suge", CapsMode.OFF)
        assertEquals(listOf("suge", "suge", "surge", "sure", "siege"), results2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `empty typed word gives next word suggestions`() {
        nextWordSuggestions = suggestionResults(listOf(
            suggestion("another", 24, currentTypingLocale),
            suggestion("next", 123, currentTypingLocale),
        ))
        val results = getSuggestedWords(false, "", CapsMode.OFF)
        assertEquals(listOf("next", "another"), results.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `single letter suggestions are suppressed dependent on score of other suggestions`() {
        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("a", 100),
            suggestion("next", 95), // more than 94% -> moved to top
        ))
        val results = getSuggestedWords(true, "", CapsMode.OFF)
        assertEquals(listOf("next", "a"), results.mSuggestedWordInfoList.map { it.mWord })
        assertEquals(2, results.mSuggestedWordInfoList.size)
        assertEquals(93, results.mSuggestedWordInfoList[1].mScore) // score is 93% of original

        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("a", 100),
            suggestion("next", 90),
        ))
        val results2 = getSuggestedWords(true, "", CapsMode.OFF)
        assertEquals(listOf("a", "next"), results2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `prefer next word suggestions`() {
        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("ab", 1000),
            suggestion("next", 935),
            suggestion("something", 925), // not used because less than 93%
        ))
        nextWordSuggestions = suggestionResults(listOf(
            suggestion("next", 180),
            suggestion("something", 175),
            suggestion("ab", 160), // not used because less than 170
        ))
        val results = getSuggestedWords(true, "", CapsMode.OFF)
        assertEquals(listOf("next", "ab", "something"), results.mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf(935, 1000, 925), results.mSuggestedWordInfoList.map { it.mScore })

        nextWordSuggestions = suggestionResults(listOf(
            suggestion("next", 180),
            suggestion("something", 175),
            suggestion("ab", 172), // now it's used
        ))
        assertEquals("ab", getSuggestedWords(true, "", CapsMode.OFF).mSuggestedWordInfoList[0].mWord)
    }

    @Test fun `single quotes at end are attached to suggestions`() {
        // the single quote is a word connector, so it's part of the typed word if at the end (or in the middle, but we're not interested in that)
        // the suggestions are for the normal word without ', and then the ' is attached
        // this has weird effects, because there are suggestions where the ' does not fit
        //  e.g. for emoji suggestions and languages where ' isn't used in words
        // todo: improve the situation on this!
        //  disable for shortcuts?
        //  maybe also could be removed from word connectors for a bunch of languages

        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("ab", 1000),
            suggestion("next", 935),
            suggestion("something", 925),
            suggestion("🎄", 924),
        ))
        val results = getSuggestedWords(false, "someword'", CapsMode.OFF)
        assertEquals(listOf("someword'", "ab'", "next'", "something'", "🎄'"), results.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `tap typing suggestions depend on typed word capitalization, ignoring automatic caps modes`() {
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        listOf(CapsMode.OFF, CapsMode.AUTO, CapsMode.AUTO_LOCKED).forEach { mode ->
            // not capitalized -> original suggestions
            assertEquals(listOf("but", "buy", "bit", "😢", "Butter"),
                getSuggestedWords(false, "but", mode).mSuggestedWordInfoList.map { it.mWord })

            // capitalized -> capitalized suggestions
            assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
                getSuggestedWords(false, "But", mode).mSuggestedWordInfoList.map { it.mWord })

            // more than one uppercase character, but not all -> original suggestions
            assertEquals(listOf("BuT", "but", "buy", "bit", "😢", "Butter"),
                getSuggestedWords(false, "BuT", mode).mSuggestedWordInfoList.map { it.mWord })

            // full uppercase typed word -> uppercase suggestions
            assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
                getSuggestedWords(false, "BUT", mode).mSuggestedWordInfoList.map { it.mWord })
        }
    }

    @Test fun `next word and glide typing suggestions use automatic caps modes`() {
        glideTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))
        assertEquals(listOf("but", "buy", "bit", "😢", "Butter"),
            getSuggestedWords(true, "", CapsMode.OFF).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
            getSuggestedWords(true, "", CapsMode.AUTO).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(true, "", CapsMode.AUTO_LOCKED).mSuggestedWordInfoList.map { it.mWord })

        nextWordSuggestions = glideTypingSuggestions
        assertEquals(listOf("but", "buy", "bit", "😢", "Butter"),
            getSuggestedWords(false, "", CapsMode.OFF).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
            getSuggestedWords(false, "", CapsMode.AUTO).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(false, "", CapsMode.AUTO_LOCKED).mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `no autocorrect if more than one uppercase character in typed word, but not all uppercase`() {
        enableAutocorrect(confidenceVeryAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 650000),
            suggestion("bit", 620000),
            suggestion("buy", 500000),
            suggestion("😢", 100000),
            suggestion("Butter", 80000),
        ))

        val result = getSuggestedWords(false, "bur", CapsMode.OFF)
        assert(result.mWillAutoCorrect)
        assertEquals(listOf("bur", "but", "bur", "bit", "buy", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        val result2 = getSuggestedWords(false, "Bur", CapsMode.OFF)
        assert(result2.mWillAutoCorrect)
        assertEquals(listOf("Bur", "But", "Bur", "Bit", "Buy", "😢", "Butter"), result2.mSuggestedWordInfoList.map { it.mWord })

        val result3 = getSuggestedWords(false, "BUr", CapsMode.OFF)
        assert(!result3.mWillAutoCorrect)
        assertEquals(listOf("BUr", "but", "bit", "buy", "😢", "Butter"), result3.mSuggestedWordInfoList.map { it.mWord })

        val result4 = getSuggestedWords(false, "BUR", CapsMode.OFF)
        assert(result4.mWillAutoCorrect)
        assertEquals(listOf("BUR", "BUT", "BUR", "BIT", "BUY", "😢", "BUTTER"), result4.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `suggestions use manual caps modes`() {
        // added in https://github.com/HeliBorg/HeliBoard/pull/1807 / cb0eae695f0cbd061e5bbcc416d6e14d18d869d8
        // manual caps mode can be set any time, even if there are already suggestions
        // on the phone this will give different suggestions because there is different proximityInfo (coming from keyboard, which is different when shifted)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "but", CapsMode.MANUAL)
        assert(!result.mWillAutoCorrect)
        assertEquals(listOf("but", "But", "Buy", "Bit", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        assertEquals(listOf("but", "BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(false, "but", CapsMode.MANUAL_LOCKED).mSuggestedWordInfoList.map { it.mWord })

        // same for glide typing
        glideTypingSuggestions = tapTypingSuggestions
        val result2 = getSuggestedWords(true, "", CapsMode.MANUAL)
        assert(!result2.mWillAutoCorrect)
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"), result2.mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(true, "", CapsMode.MANUAL_LOCKED).mSuggestedWordInfoList.map { it.mWord })

        // same for next word suggestions
        nextWordSuggestions = tapTypingSuggestions
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"),
            getSuggestedWords(false, "", CapsMode.MANUAL).mSuggestedWordInfoList.map { it.mWord })
        assertEquals(listOf("BUT", "BUY", "BIT", "😢", "BUTTER"),
            getSuggestedWords(false, "", CapsMode.MANUAL_LOCKED).mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `autocorrect will "correct" capitalization with manual caps modes`() {
        enableAutocorrect(confidenceAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "but", CapsMode.MANUAL)
        assert(result.mWillAutoCorrect)
        assertEquals(listOf("but", "But", "Buy", "Bit", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        val result2 = getSuggestedWords(false, "but", CapsMode.MANUAL_LOCKED)
        assert(result2.mWillAutoCorrect)
        assertEquals(listOf("but", "BUT", "BUY", "BIT", "😢", "BUTTER"), result2.mSuggestedWordInfoList.map { it.mWord })

        glideTypingSuggestions = tapTypingSuggestions
        val result3 = getSuggestedWords(true, "", CapsMode.MANUAL)
        // mWillAutoCorrect is true on phone, because after glide typing is done, the result is set as typed word
        // when pressing shift to capitalize it, autocorrect will only work if the typed word changes when capitalized. this does not work without typed word
        //assert(result3.mWillAutoCorrect)
        assertEquals("but", result3.typedWordInfo.mWord)
        assertEquals(listOf("But", "Buy", "Bit", "😢", "Butter"), result3.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `no caps mode autocorrect if word has many uppercase letters`() {
        // bug report in https://github.com/HeliBorg/HeliBoard/issues/2162
        enableAutocorrect(confidenceAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 100),
            suggestion("buy", 95),
            suggestion("bit", 90),
            suggestion("😢", 85),
            suggestion("Butter", 80),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "buT", CapsMode.MANUAL)
        assert(!result.mWillAutoCorrect)
        assertEquals(listOf("buT", "BuT", "But", "Buy", "Bit", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        val result2 = getSuggestedWords(false, "buT", CapsMode.MANUAL_LOCKED)
        assert(!result2.mWillAutoCorrect)
        assertEquals(listOf("buT", "BUT", "BUY", "BIT", "😢", "BUTTER"), result2.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `normal autocorrect works with manual caps modes`() {
        enableAutocorrect(confidenceAggressive)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 650000),
            suggestion("bit", 620000),
            suggestion("buy", 500000),
            suggestion("😢", 100000),
            suggestion("Butter", 80000),
        ))

        // shift -> capitalized suggestions
        val result = getSuggestedWords(false, "bur", CapsMode.MANUAL)
        assert(result.mWillAutoCorrect)
        assertEquals(listOf("bur", "But", "Bur", "Bit", "Buy", "😢", "Butter"), result.mSuggestedWordInfoList.map { it.mWord })

        // caps lock -> uppercase suggestions
        val result2 = getSuggestedWords(false, "bur", CapsMode.MANUAL_LOCKED)
        assert(result2.mWillAutoCorrect)
        assertEquals(listOf("bur", "BUT", "BUR", "BIT", "BUY", "😢", "BUTTER"), result2.mSuggestedWordInfoList.map { it.mWord })

        // score too low will correct to capitalized typed word (except if using very aggressive threshold)
        tapTypingSuggestions = suggestionResults(listOf(
            suggestion("but", 6500),
            suggestion("bit", 6200),
            suggestion("buy", 5000),
            suggestion("😢", 1000),
            suggestion("Butter", 800),
        ))
        val result3 = getSuggestedWords(false, "bur", CapsMode.MANUAL)
        assert(result3.mWillAutoCorrect)
        assertEquals(listOf("bur", "Bur", "But", "Bit", "Buy", "😢", "Butter"), result3.mSuggestedWordInfoList.map { it.mWord })

        val result4 = getSuggestedWords(false, "bur", CapsMode.MANUAL_LOCKED)
        assert(result4.mWillAutoCorrect)
        assertEquals(listOf("bur", "BUR", "BUT", "BIT", "BUY", "😢", "BUTTER"), result4.mSuggestedWordInfoList.map { it.mWord })
    }

    @Test fun `personal candidates enter the live slate without taking over autocorrect`() {
        val editorInfo = EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }
        PersonalizationRuntime.wipe(latinIME)
        try {
            PersonalizationRuntime.observeCommit(
                latinIME, editorInfo, false, "LibreBoard", currentTypingLocale.toLanguageTag(), false,
            )
            tapTypingSuggestions = suggestionResults(listOf(suggestion("libretto", 900_000)))
            val completion = getSuggestedWords(false, "libre", CapsMode.OFF)
            assertEquals("libre", completion.getWord(0))
            assertEquals("libretto", completion.getWord(1))
            assert("LibreBoard" in completion.mSuggestedWordInfoList.map { it.mWord })

            enableAutocorrect(confidenceVeryAggressive)
            tapTypingSuggestions = suggestionResults(listOf(suggestion("whiteboard", 1_500_000)))
            val exactPersonal = getSuggestedWords(false, "libreboard", CapsMode.OFF)
            assert(!exactPersonal.mWillAutoCorrect)
            assertEquals("libreboard", exactPersonal.getWord(0))
            assert("LibreBoard" in exactPersonal.mSuggestedWordInfoList.map { it.mWord })

            val exactPersonalWithShift = getSuggestedWords(false, "libreboard", CapsMode.MANUAL)
            assert(!exactPersonalWithShift.mWillAutoCorrect)
            assert("LibreBoard" in exactPersonalWithShift.mSuggestedWordInfoList.map { it.mWord })
        } finally {
            PersonalizationRuntime.wipe(latinIME)
        }
    }

    @Test fun `live CTC candidate enters batch suggestions and remains a single typed commit`() {
        LiveTypingEngine.swipeDecoder.clear()
        LiveTypingEngine.swipeDecoder.install(SwipeDecoder { _, _ ->
            SwipeDecodeResult(
                EngineAvailability.AVAILABLE,
                listOf(Candidate(
                    "cat",
                    languageTag = currentTypingLocale.toLanguageTag(),
                    sources = setOf(CandidateSource.CTC_SWIPE),
                    components = ScoreComponents(spatial = -0.1, staticFrequency = 4.0),
                )),
            )
        }, AutoCloseable {})
        try {
            val result = getSuggestedWordsForGestureWithPath("cat")

            assertEquals(listOf("cat"), result.mSuggestedWordInfoList.map { it.mWord })
            assertEquals("cat", result.typedWordInfo.mWord)
            assert(!result.mWillAutoCorrect)
        } finally {
            LiveTypingEngine.swipeDecoder.clear()
        }
    }

    @Test fun `missing CTC model retains classic nearest-key batch fallback`() {
        LiveTypingEngine.swipeDecoder.clear()
        tapTypingSuggestions = suggestionResults(listOf(suggestion("cat", 900_000)))
        glideTypingSuggestions = suggestionResults(emptyList())

        val result = getSuggestedWordsForGestureWithPath("cat")

        assertEquals("cat", result.mSuggestedWordInfoList.first().mWord)
        assertEquals("cat", result.typedWordInfo.mWord)
    }

    private fun getSuggestedWords(gesture: Boolean, typedWord: String, capsMode: CapsMode): SuggestedWords {
        val wc = WordComposer()
        if (gesture) wc.setBatchInputPointers(InputPointers(1))
        if (typedWord.isNotEmpty()) {
            StringUtils.toCodePointArray(typedWord).forEach {
                val e = Event.createEventForCodePointFromAlreadyTypedText(it, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE)
                wc.applyProcessedEvent(wc.processEvent(e))
            }
        }
        // we ignore the special treatment of CapsMode.AUTO here (depends on settings, input type, and text before cursor)
        if (gesture)
            wc.setCapitalizedModeAtStartComposingTime(capsMode) // done in inputlogic when entering batch mode
        wc.adviseCapitalizedModeBeforeFetchingSuggestions(capsMode) // done in inputLogic before getSuggestedWords

        val params = KeyboardParams()
        val elementId = when (capsMode) {
            CapsMode.MANUAL_LOCKED -> KeyboardElement.ALPHABET_SHIFT_LOCKED
            CapsMode.MANUAL -> KeyboardElement.ALPHABET_MANUAL_SHIFTED
            CapsMode.AUTO -> KeyboardElement.ALPHABET_AUTOMATIC_SHIFTED
            else -> KeyboardElement.ALPHABET // off
        }
        params.GRID_HEIGHT = 1
        params.GRID_WIDTH = 1
        params.mId = KeyboardLayoutSet.getFakeKeyboardId(elementId)

        suggest.clearNextWordSuggestionsCache()
        return suggest.getSuggestedWords(
            wc, NgramContext.EMPTY_PREV_WORDS_INFO, Keyboard(params), Settings.getValues().mSettingsValuesForSuggestion,
            Settings.getValues().mAutoCorrectEnabled, 0, 0
        )
    }

    private fun getSuggestedWordsForGestureWithPath(trace: String): SuggestedWords {
        val params = KeyboardParams().apply {
            GRID_HEIGHT = 1
            GRID_WIDTH = trace.length
            mId = KeyboardLayoutSet.getFakeKeyboardId(KeyboardElement.ALPHABET)
            mOccupiedWidth = trace.length * 100
            mOccupiedHeight = 100
            mBaseWidth = mOccupiedWidth
            mBaseHeight = mOccupiedHeight
            mMostCommonKeyWidth = 100
            mMostCommonKeyHeight = 100
        }
        val pointers = InputPointers(trace.length)
        trace.forEachIndexed { index, character ->
            params.onAddKey(Key(
                character.toString(), null, character.code, null, null,
                0, 0, index * 100, 0, 100, 100, 0, 0,
            ))
            pointers.addPointer(index * 100 + 50, 50, 0, index * 10)
        }
        val composer = WordComposer().apply {
            setBatchInputPointers(pointers)
            setCapitalizedModeAtStartComposingTime(CapsMode.OFF)
            adviseCapitalizedModeBeforeFetchingSuggestions(CapsMode.OFF)
        }
        suggest.clearNextWordSuggestionsCache()
        return suggest.getSuggestedWords(
            composer,
            NgramContext.EMPTY_PREV_WORDS_INFO,
            Keyboard(params),
            Settings.getValues().mSettingsValuesForSuggestion,
            Settings.getValues().mAutoCorrectEnabled,
            0,
            33,
        )
    }

    // The Phase 0 measurement harness scores what the editor actually receives. These lock the two
    // entry points it reads, because scoring rank one of the strip instead silently reported the
    // typed word back to itself for every measured row and hid every correction.

    @Test fun `commit decision keeps the typed word when production will not auto-correct`() {
        val locale = Locale.ENGLISH
        val suggestedWords = suggestedWordsOf(
            typedWord = "targte",
            words = listOf("targte", "target"),
            willAutoCorrect = false,
            locale = locale,
        )
        // "target" is on the strip one tap away, but nothing replaces the typed text.
        assertEquals("target", suggestedWords.getWord(SuggestedWords.INDEX_OF_AUTO_CORRECTION))
        assertEquals(
            Suggest.CommitDecision("targte", false),
            Suggest.commitDecisionOf(suggestedWords, "targte"),
        )
    }

    @Test fun `commit decision takes the correction slot when production will auto-correct`() {
        val suggestedWords = suggestedWordsOf(
            typedWord = "targte",
            words = listOf("targte", "target"),
            willAutoCorrect = true,
            locale = Locale.ENGLISH,
        )
        assertEquals(
            Suggest.CommitDecision("target", true),
            Suggest.commitDecisionOf(suggestedWords, "targte"),
        )
    }

    @Test fun `commit decision falls back to the typed word without a correction slot`() {
        val suggestedWords = suggestedWordsOf(
            typedWord = "targte",
            words = listOf("targte"),
            willAutoCorrect = true,
            locale = Locale.ENGLISH,
        )
        assertEquals(
            Suggest.CommitDecision("targte", false),
            Suggest.commitDecisionOf(suggestedWords, "targte"),
        )
    }

    @Test fun `classic commit decision corrects a whitelisted replacement`() {
        val locale = Locale.ENGLISH
        assertEquals(
            Suggest.CommitDecision("I'll", true),
            classicCommitDecision("ill",
                listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale))),
        )
    }

    @Test fun `classic commit decision keeps a word the ngram context prefers`() {
        val locale = Locale.ENGLISH
        nextWordSuggestions = suggestionResults(listOf(suggestion("ill", 200, locale)))
        assertEquals(
            Suggest.CommitDecision("ill", false),
            classicCommitDecision("ill",
                listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale))),
        )
    }

    @Test fun `classic commit decision keeps the typed word without any suggestion`() {
        assertEquals(
            Suggest.CommitDecision("ill", false),
            classicCommitDecision("ill", emptyList()),
        )
    }

    @Test fun `a field that does not request auto-correction disables the threshold entirely`() {
        // SettingsValues sets the auto-correction threshold to Float.MAX_VALUE when auto-correction
        // is off, so nothing but whitelist entries can ever be committed. A measurement fixture
        // built on a bare TYPE_CLASS_TEXT field therefore reports a keyboard that cannot correct,
        // which is why the Phase 0 harness requests TYPE_TEXT_FLAG_AUTO_CORRECT and asserts this.
        fun loadWith(inputType: Int): Boolean {
            Settings.getInstance().loadSettings(
                latinIME,
                Locale.ENGLISH,
                InputAttributes(
                    EditorInfo().apply { this.inputType = inputType }, false, latinIME.packageName),
            )
            return Settings.getValues().mAutoCorrectEnabled
        }

        assert(!loadWith(InputType.TYPE_CLASS_TEXT))
        assertEquals(Float.MAX_VALUE, Settings.getValues().mAutoCorrectionThreshold)

        assert(loadWith(InputType.TYPE_CLASS_TEXT
            or InputType.TYPE_TEXT_VARIATION_NORMAL
            or InputType.TYPE_TEXT_FLAG_AUTO_CORRECT))
        assert(Settings.getValues().mAutoCorrectionThreshold < 1f)
    }

    @Test fun `a resumed composition is never auto-corrected`() {
        // WordComposer.setComposingWord marks the composition resumed, which production treats as
        // "the user tapped back into an existing word" and never auto-corrects. A measurement
        // harness that builds rows that way observes 0% correction on every system; rows must be
        // replayed as key events instead.
        val locale = Locale.ENGLISH
        val suggestions = listOf(suggestion("I'll", Int.MAX_VALUE, locale), suggestion("ill", 1500000, locale))
        val fresh = shouldBeAutoCorrected("ill", suggestions, null, null, locale, confidenceModest)
        assert(fresh.last())

        val resumed = WordComposer().apply {
            setComposingWord(StringUtils.toCodePointArray("ill"), IntArray(6) { Constants.NOT_A_COORDINATE })
        }
        assert(resumed.isResumed)
        val result = shouldBeAutoCorrected(
            "ill", suggestions, null, null, locale, confidenceModest, wordComposer = resumed)
        assert(!result.last())
    }

    private fun suggestedWordsOf(
        typedWord: String, words: List<String>, willAutoCorrect: Boolean, locale: Locale,
    ): SuggestedWords {
        val infos = ArrayList(words.map { suggestion(it, 1_000_000, locale) })
        return SuggestedWords(
            infos, null, suggestion(typedWord, SuggestedWordInfo.MAX_SCORE, locale),
            !willAutoCorrect, willAutoCorrect, false, SuggestedWords.INPUT_STYLE_TYPING, 0,
        )
    }

    private fun classicCommitDecision(
        typedWord: String,
        suggestions: List<SuggestedWordInfo>,
        autoCorrectThreshold: Float = confidenceModest,
    ): Suggest.CommitDecision {
        enableAutocorrect(autoCorrectThreshold)
        val composer = WordComposer()
        StringUtils.toCodePointArray(typedWord).forEach {
            val event = Event.createEventForCodePointFromAlreadyTypedText(
                it, Constants.NOT_A_COORDINATE, Constants.NOT_A_COORDINATE)
            composer.applyProcessedEvent(composer.processEvent(event))
        }
        val params = KeyboardParams().apply {
            GRID_HEIGHT = 1
            GRID_WIDTH = 1
            mId = KeyboardLayoutSet.getFakeKeyboardId(KeyboardElement.ALPHABET)
        }
        suggest.clearNextWordSuggestionsCache()
        return suggest.classicCommitDecision(
            composer, NgramContext.EMPTY_PREV_WORDS_INFO, Keyboard(params),
            Settings.getValues().mSettingsValuesForSuggestion,
            SuggestedWords.INPUT_STYLE_TYPING, suggestionResults(suggestions),
        )
    }

    private fun enableAutocorrect(autoCorrectThreshold: Float) {
        latinIME.prefs().edit {
            putBoolean(Settings.PREF_AUTO_CORRECTION, true)
            putFloat(Settings.PREF_AUTO_CORRECT_CONFIDENCE, autoCorrectThreshold)
            putBoolean(Settings.PREF_MORE_AUTO_CORRECTION, true)
        }
        suggest.setAutoCorrectionThreshold(Settings.getValues().mAutoCorrectionThreshold)
    }

    private fun shouldBeAutoCorrected(word: String, // typed word
                              suggestions: List<SuggestedWordInfo>, // suggestions ordered by score, including suggestion for typed word if in dictionary
                              firstSuggestionForEmpty: SuggestedWordInfo?, // first suggestion if typed word would be empty (null if none)
                              typedWordSuggestionForEmpty: SuggestedWordInfo?, // suggestion for actually typed word if typed word would be empty (null if none)
                              typingLocale: Locale, // used for checking whether suggestion locale is the same, relevant e.g. for English i -> I shortcut, but we want Polish i
                              autoCorrectThreshold: Float,
                              engineAutoCorrectionNormalized: String? = null,
                              wordComposer: WordComposer = WordComposer.getComposerForTest(false),
    ): List<Boolean> {
        enableAutocorrect(autoCorrectThreshold)
        currentTypingLocale = typingLocale
        val suggestionsContainer = ArrayList<SuggestedWordInfo>().apply { addAll(suggestions) }
        val suggestionResults = SuggestionResults(suggestions.size, false, false)
        suggestions.forEach { suggestionResults.add(it) }

        // store the original SuggestedWordInfo for typed word, as it will be removed
        // we may want to re-add it in case auto-correction happens, so that the original word can at least be selected
        val typedWordFirstOccurrenceWordInfo: SuggestedWordInfo? = suggestionsContainer.firstOrNull { it.mWord == word }

        val firstOccurrenceOfTypedWordInSuggestions =
            SuggestedWordInfo.removeDupsAndTypedWord(word, suggestionsContainer)

        return suggest.shouldBeAutoCorrected(
            StringUtils.getTrailingSingleQuotesCount(word),
            word,
            suggestionsContainer.firstOrNull(), // todo: get from suggestions? mostly it's just removing the typed word, right?
            { firstSuggestionForEmpty to typedWordSuggestionForEmpty },
            true, // doesn't make sense otherwise
            wordComposer,
            suggestionResults,
            firstOccurrenceOfTypedWordInSuggestions,
            typedWordFirstOccurrenceWordInfo,
            engineAutoCorrectionNormalized,
        ).toList()
    }
}

private var currentTypingLocale = Locale.ENGLISH
private var tapTypingSuggestions = suggestionResults(emptyList())
private var glideTypingSuggestions = suggestionResults(emptyList())
private var nextWordSuggestions = suggestionResults(emptyList())

fun suggestion(word: String, score: Int, locale: Locale = currentTypingLocale, shortcut: Boolean = false) =
    SuggestedWordInfo(
        word,
        "", // irrelevant

        // typically IntMax for whitelisted, 1.5M for exact match, 600k for close match
        // when previous word context is empty, scores are usually 200+ if word is known and somewhat often used, 0 if unknown
        score,
        if (score == Int.MAX_VALUE) KIND_WHITELIST
            else if (shortcut) KIND_SHORTCUT // whitelist & shortcut only counts a whitelist
            else KIND_FLAG_APPROPRIATE_FOR_AUTO_CORRECTION, // shortcuts seem to never have this flag
        TestDict(locale),
        0, // irrelevant
        0 // irrelevant?
    )

fun suggestionResults(suggestions: List<SuggestedWordInfo>, isBeginningOfSentence: Boolean = false) =
    SuggestionResults(18, isBeginningOfSentence, false).apply { addAll(suggestions) }

@Implements(DictionaryFacilitatorImpl::class)
class ShadowFacilitator {
    @Implementation
    fun getCurrentLocale(): Locale = currentTypingLocale
    @Implementation
    fun getMainLocale(): Locale = currentTypingLocale
    @Implementation
    fun getActiveLocales(): List<Locale> = listOf(currentTypingLocale)
    @Implementation
    fun hasAtLeastOneInitializedMainDictionary() = true // otherwise no autocorrect
    @Implementation
    // what is relevant for facilitator?
    // only detecting whether we want next word suggestions (typed word empty and not batch mode)
    fun getSuggestionResults(composedData: ComposedData, ngramContext: NgramContext, keyboard: Keyboard,
                             settingsValuesForSuggestion: SettingsValuesForSuggestion, sessionId: Int, inputStyle: Int
    ): SuggestionResults = when {
        composedData.mIsBatchMode -> glideTypingSuggestions
        composedData.mTypedWord.isEmpty() -> nextWordSuggestions
        else -> tapTypingSuggestions
    }
}

private class TestDict(locale: Locale) : Dictionary(TYPE_MAIN, locale) {
    override fun getSuggestions(
        composedData: ComposedData?,
        ngramContext: NgramContext?,
        proximityInfoHandle: Long,
        settingsValuesForSuggestion: SettingsValuesForSuggestion?,
        sessionId: Int,
        weightForLocale: Float,
        inOutWeightOfLangModelVsSpatialModel: FloatArray?
    ): ArrayList<SuggestedWordInfo> {
        return ArrayList()
    }

    override fun isInDictionary(word: String?): Boolean = false
}
