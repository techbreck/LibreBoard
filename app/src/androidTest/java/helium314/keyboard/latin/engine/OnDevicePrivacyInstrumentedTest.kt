// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.Manifest
import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import androidx.test.runner.AndroidJUnit4

/**
 * Privacy invariants that need a real Android package manager and credential-encrypted SQLite.
 *
 * This suite is necessary device evidence, but it does not replace the physical GrapheneOS release
 * matrix in docs/grapheneos.md.
 */
@RunWith(AndroidJUnit4::class)
class OnDevicePrivacyInstrumentedTest {
    private lateinit var context: Context

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        assertFalse("instrumentation must use credential-encrypted app storage", context.isDeviceProtectedStorage)
        PersonalizationRuntime.wipe(context)
    }

    @After
    fun tearDown() {
        PersonalizationRuntime.wipe(context)
    }

    @Suppress("DEPRECATION")
    @Test
    fun installedPackageHasNoNetworkOrBackupAndImeIsDirectBootAware() {
        val packageInfo = context.packageManager.getPackageInfo(
            context.packageName,
            PackageManager.GET_PERMISSIONS or PackageManager.GET_SERVICES,
        )
        val requestedPermissions = packageInfo.requestedPermissions.orEmpty().toSet()
        assertFalse(Manifest.permission.INTERNET in requestedPermissions)
        assertFalse(Manifest.permission.ACCESS_NETWORK_STATE in requestedPermissions)

        val applicationInfo = packageInfo.applicationInfo
        assertNotNull(applicationInfo)
        assertEquals(0, applicationInfo!!.flags and ApplicationInfo.FLAG_ALLOW_BACKUP)
        assertEquals(0, applicationInfo.flags and ApplicationInfo.FLAG_USES_CLEARTEXT_TRAFFIC)

        val imeService = packageInfo.services.orEmpty().single {
            it.name == "helium314.keyboard.latin.LatinIME"
        }
        assertEquals(Manifest.permission.BIND_INPUT_METHOD, imeService.permission)
        assertTrue("IME service must be available before first unlock", imeService.directBootAware)
    }

    @Test
    fun fieldPolicyMatrixFailsClosedOnAndroidEditorInfo() {
        assertEquals(FieldPolicy.SENSITIVE, FieldPolicyResolver.resolve(null))
        assertFullyRestricted(editor(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD))
        assertFullyRestricted(editor(InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD))
        assertFullyRestricted(editor(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS))

        val noLearning = editor(InputType.TYPE_CLASS_TEXT).apply {
            imeOptions = EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING
        }
        assertFullyRestricted(noLearning)

        val noSuggestions = editor(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS)
        assertEquals(FieldPolicy.NO_SUGGESTIONS, FieldPolicyResolver.resolve(noSuggestions))
        assertEquals(FieldPolicy.NORMAL, FieldPolicyResolver.resolve(noSuggestions, allowNoSuggestionsOverride = true))

        val passwordWithOverride = editor(
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD or
                InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS,
        )
        assertEquals(
            FieldPolicy.SENSITIVE,
            FieldPolicyResolver.resolve(passwordWithOverride, allowNoSuggestionsOverride = true),
        )

        val terminal = editor(InputType.TYPE_CLASS_TEXT).apply { privateImeOptions = "org.libreboard.terminal" }
        assertEquals(FieldPolicy.TERMINAL, FieldPolicyResolver.resolve(terminal))
        with(FieldPolicy.TERMINAL) {
            assertFalse(allowsContextRead)
            assertTrue(allowsSuggestions)
            assertFalse(allowsComposing)
            assertFalse(allowsAutoCorrection)
            assertFalse(allowsClipboardCapture)
            assertFalse(allowsPersistence)
        }
    }

    @Test
    fun personalStorePersistsNormalCommitButNotIncognitoOrSensitiveInput() {
        val normalEditor = editor(InputType.TYPE_CLASS_TEXT)
        PersonalizationRuntime.observeCommit(
            context,
            normalEditor,
            incognito = false,
            committedWord = "Breck",
            languageTag = "en-US",
            manualSelection = true,
        )

        val candidates = PersonalizationRuntime.suggest(
            context = context,
            rawText = "bre",
            precedingContext = "",
            enabledLanguageTags = listOf("en-US"),
            fieldPolicy = FieldPolicy.NORMAL,
            incognito = false,
            inputStyle = InputStyle.TAP,
            sequenceId = 1,
        )
        assertTrue(candidates.any { it.surface == "Breck" && it.exactPersonalMatch.not() })

        PersonalizationRuntime.observeCommit(
            context,
            normalEditor,
            incognito = true,
            committedWord = "NeverPersistIncognito",
            languageTag = "en-US",
            manualSelection = false,
        )
        PersonalizationRuntime.observeCommit(
            context,
            editor(InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD),
            incognito = false,
            committedWord = "NeverPersistSensitive",
            languageTag = "en-US",
            manualSelection = false,
        )

        assertTrue(
            PersonalizationRuntime.suggest(
                context, "bre", "", listOf("en-US"), FieldPolicy.SENSITIVE,
                incognito = false, InputStyle.TAP, sequenceId = 2,
            ).isEmpty(),
        )
        assertTrue(
            PersonalizationRuntime.suggest(
                context, "bre", "", listOf("en-US"), FieldPolicy.NORMAL,
                incognito = true, InputStyle.TAP, sequenceId = 3,
            ).isEmpty(),
        )

        val exported = requireNotNull(PersonalizationRuntime.export(context)).decodeToString()
        assertTrue(exported.contains("Breck"))
        assertFalse(exported.contains("NeverPersistIncognito"))
        assertFalse(exported.contains("NeverPersistSensitive"))
    }

    private fun assertFullyRestricted(editorInfo: EditorInfo) {
        with(FieldPolicyResolver.resolve(editorInfo)) {
            assertFalse(allowsContextRead)
            assertFalse(allowsSuggestions)
            assertFalse(allowsAutoCorrection)
            assertFalse(allowsClipboardCapture)
            assertFalse(allowsPersistence)
        }
    }

    private fun editor(inputType: Int) = EditorInfo().apply { this.inputType = inputType }
}
