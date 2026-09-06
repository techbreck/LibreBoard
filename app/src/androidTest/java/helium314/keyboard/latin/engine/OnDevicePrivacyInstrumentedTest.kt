// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.Manifest
import android.app.ActivityManager
import android.content.ClipDescription
import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.PersistableBundle
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import helium314.keyboard.compat.ClipboardManagerCompat
import helium314.keyboard.latin.database.ClipboardDao
import helium314.keyboard.latin.database.ClipboardHistoryPolicy
import helium314.keyboard.keyboard.clipboard.ClipboardSearchActivity
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.engine.personal.LearnedDataWiper
import helium314.keyboard.latin.engine.runtime.NeuralDevicePolicy
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
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
        ClipboardDao.getInstance(context)?.clear()
    }

    @After
    fun tearDown() {
        PersonalizationRuntime.wipe(context)
        ClipboardDao.getInstance(context)?.clear()
        context.filesDir.resolve("models/active/fixture.onnx").delete()
        context.filesDir.resolve("models/active").delete()
        context.filesDir.resolve("models").delete()
        context.filesDir.resolve("clipboard/fixture").delete()
    }

    @Suppress("DEPRECATION")
    @Test
    fun installedPackageHasNoNetworkOrBackupAndImeIsDirectBootAware() {
        val packageInfo = context.packageManager.getPackageInfo(
            context.packageName,
            PackageManager.GET_PERMISSIONS or PackageManager.GET_SERVICES or PackageManager.GET_ACTIVITIES,
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

        val clipboardSearch = packageInfo.activities.orEmpty().single {
            it.name == ClipboardSearchActivity::class.java.name
        }
        assertFalse("clipboard search must not be externally launchable", clipboardSearch.exported)
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


        val clipboardSearch = editor(InputType.TYPE_CLASS_TEXT).apply {
            privateImeOptions = "$PRIVATE_IME_OPTION_CLIPBOARD_SEARCH.240,"
        }
        assertEquals(FieldPolicy.NO_LEARNING, FieldPolicyResolver.resolve(clipboardSearch))
        assertFullyRestricted(clipboardSearch)
    }

    @Test
    fun platformSensitiveClipboardMarkerVetoesHistoryCapture() {
        val description = ClipDescription("private", arrayOf("text/plain")).apply {
            extras = PersistableBundle().apply {
                putBoolean("android.content.extra.IS_SENSITIVE", true)
            }
        }
        val markedSensitive = ClipboardManagerCompat.getClipSensitivity(description) == true

        assertTrue(markedSensitive)
        assertFalse(ClipboardHistoryPolicy.allowsCapture(FieldPolicy.NORMAL, false, markedSensitive))
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

    @Test
    fun privateStoresStayOutOfDeviceProtectedStorageWhenGivenThatContext() {
        val deviceProtectedContext = context.createDeviceProtectedStorageContext()
        assertTrue(deviceProtectedContext.isDeviceProtectedStorage)
        val privateContext = requireNotNull(
            CredentialEncryptedStorage.contextOrNull(deviceProtectedContext),
        )
        assertFalse(
            "unlocked private storage must resolve back to credential encryption",
            privateContext.isDeviceProtectedStorage,
        )

        requireNotNull(ClipboardDao.getInstance(deviceProtectedContext)).addClip(
            System.currentTimeMillis(),
            pinned = false,
            text = "credential encrypted clipboard",
        )
        PersonalizationRuntime.observeCommit(
            deviceProtectedContext,
            editor(InputType.TYPE_CLASS_TEXT),
            incognito = false,
            committedWord = "CredentialEncryptedPersonalWord",
            languageTag = "en-US",
            manualSelection = true,
        )

        assertTrue(context.getDatabasePath("libreboard_private.db").exists())
        assertTrue(context.getDatabasePath("libreboard_personal.db").exists())
        assertFalse(deviceProtectedContext.getDatabasePath("libreboard_private.db").exists())
        assertFalse(deviceProtectedContext.getDatabasePath("libreboard_personal.db").exists())
    }

    @Test
    fun requestedLowRamRunUsesPlatformFlagAndDisablesOnlyContextModel() {
        assumeTrue(
            "run with libreboardRequireLowRam=true on the dedicated low-RAM emulator",
            InstrumentationRegistry.getArguments().getString("libreboardRequireLowRam") == "true",
        )
        val activityManager = requireNotNull(context.getSystemService(ActivityManager::class.java))

        assertTrue("ActivityManager must identify the release-evidence target as low RAM", activityManager.isLowRamDevice)
        assertFalse(NeuralDevicePolicy.allowsModel(ModelKind.CONTEXT_RESCORER, context))
        assertTrue(NeuralDevicePolicy.allowsModel(ModelKind.SWIPE_CTC, context))
    }

    @Test
    fun clipboardDatabaseIsBoundedSearchableAndSafeAgainstStaleIds() {
        val dao = requireNotNull(ClipboardDao.getInstance(context))
        val now = System.currentTimeMillis()
        dao.addClip(now, pinned = true, text = "GrapheneOS release checklist")
        dao.addClip(now + 1, pinned = false, text = "German compound validation")

        assertEquals("GrapheneOS release checklist", dao.search("graph RELEASE").single().text)
        assertTrue(dao.search("missing").isEmpty())
        assertFalse(dao.addClip(now + 2, false, "x".repeat(ClipboardHistoryPolicy.MAX_TEXT_CHARS + 1)))
        assertEquals(2, dao.count())

        dao.addClip(now - 20_000, pinned = false, text = "expired-unpinned")
        dao.addClip(now - 20_000, pinned = true, text = "expired-but-pinned")
        dao.clearExpiredBefore(now - 10_000)
        assertTrue(dao.search("expired-unpinned").isEmpty())
        assertEquals("expired-but-pinned", dao.search("expired-but-pinned").single().text)

        repeat(ClipboardHistoryPolicy.MAX_UNPINNED_ENTRIES + 5) { index ->
            dao.addClip(now + 10 + index, pinned = false, text = "bounded-$index")
        }
        assertEquals(ClipboardHistoryPolicy.MAX_UNPINNED_ENTRIES, dao.getAll().count { !it.isPinned })
        assertTrue(dao.getAll().size <= ClipboardHistoryPolicy.MAX_TOTAL_ENTRIES)
        assertTrue(dao.search("bounded-0").isEmpty())
        assertEquals("bounded-104", dao.search("bounded-104").single().text)

        val removableId = dao.search("bounded-104").single().id
        dao.deleteClip(removableId)
        assertEquals(null, dao.get(removableId))
        dao.deleteClip(removableId) // stale UI events are idempotent

        dao.clearNonPinned()
        assertEquals(
            setOf("GrapheneOS release checklist", "expired-but-pinned"),
            dao.getAll().mapNotNull { it.text }.toSet(),
        )
    }

    @Test
    fun explicitLearnedDataWipeClearsCeRowsAndLegacyFilesWithoutTouchingModelsOrClipboard() {
        PersonalizationRuntime.observeCommit(
            context,
            editor(InputType.TYPE_CLASS_TEXT),
            incognito = false,
            committedWord = "DeleteMeOnDevice",
            languageTag = "en-US",
            manualSelection = true,
        )
        val blacklist = context.filesDir.resolve("blacklists/en-US.txt").apply {
            parentFile!!.mkdirs()
            writeText("rejected\n")
        }
        val model = context.filesDir.resolve("models/active/fixture.onnx").apply {
            parentFile!!.mkdirs()
            writeText("model")
        }
        val clipboardFile = context.filesDir.resolve("clipboard/fixture").apply {
            parentFile!!.mkdirs()
            writeText("clip")
        }

        LearnedDataWiper.wipe(context)

        val exported = requireNotNull(PersonalizationRuntime.export(context)).decodeToString()
        assertFalse(exported.contains("DeleteMeOnDevice"))
        assertFalse(blacklist.exists())
        assertTrue(model.exists())
        assertTrue(clipboardFile.exists())
        model.parentFile?.parentFile?.deleteRecursively()
        clipboardFile.parentFile?.deleteRecursively()
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
