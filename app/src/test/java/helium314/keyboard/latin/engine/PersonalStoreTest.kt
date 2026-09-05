// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Context
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.engine.personal.SqlitePersonalStore
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class PersonalStoreTest {
    private lateinit var store: SqlitePersonalStore

    @Before
    fun setUp() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        store = requireNotNull(SqlitePersonalStore.openOrNull(context))
        store.wipe()
    }

    @After
    fun tearDown() = store.wipe()

    @Test
    fun versionedExportRoundTripsWithoutRawContext() {
        store.observeCommit(CommitObservation(
            tokens = listOf("private", "phrase"),
            languageTag = "en-US",
            timestampMillis = 1_000,
            wasManualSelection = false,
            contextFingerprint = SqlitePersonalStore.fingerprintContext("private"),
        ))
        store.observeRejection(RejectionObservation(
            raw = "teh",
            replacement = "the",
            languageTag = "en-US",
            contextFingerprint = SqlitePersonalStore.fingerprintContext("a private paragraph"),
            timestampMillis = 2_000,
        ))
        val exported = store.export()
        assertTrue("raw context must not be exported", !exported.decodeToString().contains("paragraph"))

        store.wipe()
        store.restore(exported)
        assertArrayEquals(exported, store.export())
    }

    @Test
    fun malformedRestoreLeavesExistingRowsUntouched() {
        store.observeCommit(CommitObservation(listOf("kept"), "en-US", 1_000, false))
        val before = store.export()
        assertFailsWith<IllegalArgumentException> { store.restore("{}".encodeToByteArray()) }
        assertArrayEquals(before, store.export())
    }

    @Test
    fun versionOneBackupRestoresWithNormalizedSurface() {
        val legacy = """{"schemaVersion":1,"unigrams":[{"word":"legacy","language":"en-US","count":2,"lastUsed":1000}],"ngrams":[],"rejections":[]}"""
        store.restore(legacy.encodeToByteArray())

        val upgraded = store.export().decodeToString()
        assertTrue(upgraded.contains("\"schemaVersion\":2"))
        assertTrue(upgraded.contains("\"surface\":\"legacy\""))
    }

    @Test
    fun precedingConfirmedTokensFormNgramsWithoutRelearningTheirUnigrams() {
        store.observeCommit(CommitObservation(listOf("first"), "en-US", 1_000, false))
        store.observeCommit(CommitObservation(
            tokens = listOf("second"),
            languageTag = "en-US",
            timestampMillis = 2_000,
            wasManualSelection = false,
            precedingTokens = listOf("first"),
        ))
        val exported = store.export().decodeToString()
        assertTrue(exported.contains("\"word\":\"first\",\"language\":\"en-US\",\"count\":1"))
        assertTrue(exported.contains("\"prefix\":\"first\",\"continuation\":\"second\""))
    }

    @Test
    fun immediateRejectionCompensatesTheLatestSessionCommit() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val editorInfo = EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }
        PersonalizationRuntime.wipe(context)
        PersonalizationRuntime.observeCommit(context, editorInfo, false, "first", "en-US", false)
        PersonalizationRuntime.observeCommit(context, editorInfo, false, "second", "en-US", false, "sekond")
        PersonalizationRuntime.observeRejection(
            context, editorInfo, false, "sekond", "second", "en-US",
        )

        val exported = requireNotNull(PersonalizationRuntime.export(context)).decodeToString()
        assertFalse(exported.contains("\"word\":\"second\""))
        assertFalse(exported.contains("\"continuation\":\"second\""))
        assertTrue(exported.contains("\"raw\":\"sekond\",\"replacement\":\"second\""))
    }

    @Test
    fun rejectedCorrectionRequiresTwoEquivalentManualAcceptances() {
        val contextHash = SqlitePersonalStore.fingerprintContext("a stable context")
        store.observeRejection(RejectionObservation("teh", "the", "en-US", contextHash, System.currentTimeMillis()))
        assertTrue(store.isCorrectionSuppressed("teh", "the", "en-US"))

        repeat(2) {
            store.observeCommit(CommitObservation(
                tokens = listOf("the"),
                languageTag = "en-US",
                timestampMillis = System.currentTimeMillis(),
                wasManualSelection = true,
                correctionRaw = "teh",
                contextFingerprint = contextHash,
            ))
            if (it == 0) assertTrue(store.isCorrectionSuppressed("teh", "the", "en-US"))
        }
        assertFalse(store.isCorrectionSuppressed("teh", "the", "en-US"))
    }

    @Test
    fun runtimeReturnsPreferredSurfaceOnlyInNormalFields() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val editorInfo = EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }
        PersonalizationRuntime.wipe(context)
        PersonalizationRuntime.observeCommit(context, editorInfo, false, "LibreBoard", "en-US", false)

        val normal = PersonalizationRuntime.suggest(
            context = context,
            rawText = "libre",
            precedingContext = "",
            enabledLanguageTags = listOf("en-US"),
            fieldPolicy = FieldPolicy.NORMAL,
            incognito = false,
            inputStyle = InputStyle.TAP,
            sequenceId = 1,
        )
        assertEquals("LibreBoard", normal.single().surface)
        assertTrue(normal.single().exactPersonalMatch.not())

        val sensitive = PersonalizationRuntime.suggest(
            context = context,
            rawText = "libre",
            precedingContext = "should never be read",
            enabledLanguageTags = listOf("en-US"),
            fieldPolicy = FieldPolicy.SENSITIVE,
            incognito = false,
            inputStyle = InputStyle.TAP,
            sequenceId = 2,
        )
        assertTrue(sensitive.isEmpty())
    }

    @Test
    fun exactPersonalWordSurvivesABusyPrefixFamily() {
        store.observeCommit(CommitObservation(listOf("Libre"), "en-US", 1_000, false))
        repeat(40) { index ->
            repeat(2) {
                store.observeCommit(CommitObservation(listOf("libre$index"), "en-US", 2_000L + index, false))
            }
        }
        val request = TypingRequest.bounded(
            rawText = "libre",
            geometry = KeyGeometry(1f, 1f, emptyList()),
            enabledLanguages = listOf("en-US"),
            fieldPolicy = FieldPolicy.NORMAL,
            inputStyle = InputStyle.TAP,
            sequenceId = 3,
        )

        val candidates = store.suggest(request, Deadline.afterMillis(1_000))
        assertEquals("Libre", candidates.first().surface)
        assertTrue(candidates.first().exactPersonalMatch)
    }

    @Test
    fun swipeLexiconIsLanguageAndLengthBoundedAndPreservesSurface() {
        store.observeCommit(CommitObservation(listOf("LibreBoard"), "en-US", System.currentTimeMillis(), false))
        store.observeCommit(CommitObservation(listOf("Datenschutz"), "de-DE", System.currentTimeMillis(), false))
        store.observeCommit(CommitObservation(listOf("cat"), "en-US", System.currentTimeMillis(), false))

        val words = store.swipeLexicon(
            languageTags = listOf("en-US"),
            approximateLength = 10,
            maximumWords = 8,
            deadline = Deadline.afterMillis(1_000),
        )

        assertEquals(listOf("LibreBoard"), words.map { it.word })
        assertTrue(words.single().personal)
    }

    @Test
    fun runtimeNeverExposesPersonalSwipeWordsInIncognitoOrRestrictedFields() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val editorInfo = EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }
        PersonalizationRuntime.wipe(context)
        PersonalizationRuntime.observeCommit(context, editorInfo, false, "LibreBoard", "en-US", false)

        val incognito = PersonalizationRuntime.swipeLexicon(
            context, listOf("en-US"), 10, 8, FieldPolicy.NORMAL, incognito = true,
        )
        val restricted = PersonalizationRuntime.swipeLexicon(
            context, listOf("en-US"), 10, 8, FieldPolicy.SENSITIVE, incognito = false,
        )

        assertTrue(incognito.isEmpty())
        assertTrue(restricted.isEmpty())
    }
}
