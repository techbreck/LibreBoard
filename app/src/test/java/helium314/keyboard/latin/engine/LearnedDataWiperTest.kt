// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Context
import android.view.inputmethod.EditorInfo
import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.latin.engine.personal.LearnedDataWiper
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class LearnedDataWiperTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val protectedContext = context.createDeviceProtectedStorageContext()

    @Before
    fun setUp() {
        PersonalizationRuntime.wipe(context)
        cleanupFixtureFiles()
    }

    @After
    fun tearDown() {
        PersonalizationRuntime.wipe(context)
        cleanupFixtureFiles()
    }

    @Test
    fun explicitWipeRemovesEveryLearnedStoreButPreservesUserOwnedData() {
        PersonalizationRuntime.observeCommit(
            context,
            EditorInfo(),
            false,
            "LibreSecret",
            "en-US",
            true,
        )
        PersonalizationRuntime.observeRejection(
            context,
            EditorInfo(),
            false,
            "teh",
            "the",
            "en-US",
        )
        learnedTargets().forEach { file ->
            file.parentFile!!.mkdirs()
            file.writeText("private")
        }
        val model = context.filesDir.resolve("models/active/model.onnx").apply {
            parentFile!!.mkdirs()
            writeText("public-model")
        }
        val clipboard = context.filesDir.resolve("clipboard/attachment").apply {
            parentFile!!.mkdirs()
            writeText("clipboard")
        }
        val explicitDictionary = context.filesDir.resolve("dicts/en-US/custom_user.dict").apply {
            parentFile!!.mkdirs()
            writeText("dictionary")
        }
        val stagedModel = context.filesDir.parentFile!!
            .resolve(context.filesDir.name + ".restore-123/models/active/model.onnx").apply {
                parentFile!!.mkdirs()
                writeText("staged-model")
            }

        LearnedDataWiper.wipe(context)

        val exported = PersonalizationRuntime.export(context)!!.decodeToString()
        assertFalse(exported.contains("libresecret", ignoreCase = true))
        assertFalse(PersonalizationRuntime.isRejectedThisSession("teh", "the", "en-US"))
        learnedTargets().forEach { assertFalse(it.exists()) }
        assertTrue(model.exists())
        assertTrue(clipboard.exists())
        assertTrue(explicitDictionary.exists())
        assertTrue(stagedModel.exists())
    }

    @Test
    fun explicitWipeIsIdempotent() {
        LearnedDataWiper.wipe(context)
        LearnedDataWiper.wipe(context)
        assertTrue(PersonalizationRuntime.export(context)!!.decodeToString().contains("\"unigrams\":[]"))
    }

    private fun learnedTargets(): List<File> = listOf(
        context.filesDir.resolve("blacklists/en-US.txt"),
        context.filesDir.resolve("UserHistoryDictionary.fixture/private.body"),
        context.filesDir.resolve("personalization-adapters/adapter.bin"),
        context.filesDir.resolve("personalization-cache/cache.bin"),
        context.filesDir.parentFile!!
            .resolve(context.filesDir.name + ".restore-123/blacklists/en-US.txt"),
        protectedContext.filesDir.parentFile!!
            .resolve(protectedContext.filesDir.name + ".restore-123/UserHistoryDictionary.fixture/private.body"),
    )

    private fun cleanupFixtureFiles() {
        learnedTargets().forEach { it.delete() }
        context.filesDir.resolve("models/active/model.onnx").delete()
        context.filesDir.resolve("clipboard/attachment").delete()
        context.filesDir.resolve("dicts/en-US/custom_user.dict").delete()
        listOf(
            context.filesDir.resolve("blacklists"),
            context.filesDir.resolve("UserHistoryDictionary.fixture"),
            context.filesDir.resolve("personalization-adapters"),
            context.filesDir.resolve("personalization-cache"),
            context.filesDir.resolve("models/active"),
            context.filesDir.resolve("models"),
            context.filesDir.resolve("clipboard"),
            context.filesDir.resolve("dicts/en-US"),
            context.filesDir.resolve("dicts"),
        ).forEach { it.delete() }
        context.filesDir.parentFile!!.resolve(context.filesDir.name + ".restore-123").deleteRecursively()
        protectedContext.filesDir.parentFile!!
            .resolve(protectedContext.filesDir.name + ".restore-123").deleteRecursively()
    }
}
