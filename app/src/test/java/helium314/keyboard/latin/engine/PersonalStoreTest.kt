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
}
