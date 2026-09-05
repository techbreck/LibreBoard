// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.personal

import android.content.Context
import android.content.Intent
import android.os.UserManager
import helium314.keyboard.latin.personalization.PersonalizationHelper
import helium314.keyboard.latin.utils.DeviceProtectedUtils
import java.io.File

/** Complete, user-invoked deletion boundary for locally learned typing data. */
object LearnedDataWiper {
    const val ACTION_LEARNED_DATA_WIPED = "org.libreboard.keyboard.action.LEARNED_DATA_WIPED"

    private val learnedRootNames = setOf(
        "blacklists",
        "personalization-adapters",
        "personalization-cache",
    )

    /**
     * Removes active learned state and any interrupted restore copies that could retain it.
     * Static dictionaries, imported model weights, clipboard history, and explicit user
     * dictionaries are intentionally outside this deletion contract.
     */
    @JvmStatic
    @Synchronized
    fun wipe(context: Context) {
        val appContext = context.applicationContext
        val userManager = appContext.getSystemService(UserManager::class.java)
        require(userManager?.isUserUnlocked != false) {
            "Learned data can only be deleted after credential storage is unlocked"
        }

        val failures = mutableListOf<Throwable>()
        runCatching { PersonalizationRuntime.wipeRequired(appContext) }
            .onFailure(failures::add)
        runCatching { PersonalizationHelper.removeAllUserHistoryDictionaries(appContext) }
            .onFailure(failures::add)

        deletionTargets(appContext).forEach { target ->
            runCatching {
                require(!target.exists() || target.deleteRecursively()) {
                    "Could not delete learned-data path ${target.name}"
                }
            }.onFailure(failures::add)
        }

        // The receiver is registered only inside the IME process and is explicitly non-exported.
        // It clears live dictionary and prediction caches after their backing data is gone.
        appContext.sendBroadcast(
            Intent(ACTION_LEARNED_DATA_WIPED).setPackage(appContext.packageName),
        )

        if (failures.isNotEmpty()) {
            val failure = IllegalStateException("One or more learned-data stores could not be deleted")
            failures.forEach(failure::addSuppressed)
            throw failure
        }
    }

    internal fun deletionTargets(context: Context): List<File> {
        val filesRoot = context.filesDir
        val deviceRoot = DeviceProtectedUtils.getFilesDir(context)
        val activeLearnedFiles = filesRoot.listFiles().orEmpty().filter { file ->
            file.name in learnedRootNames || file.name.startsWith("UserHistoryDictionary")
        }
        val learnedFilesInRestoreCopies = listOf(filesRoot, deviceRoot).flatMap { root ->
            root.parentFile?.listFiles().orEmpty().filter { candidate ->
                candidate.isDirectory && candidate.name.startsWith(root.name + ".restore-")
            }.flatMap { stagedRoot ->
                stagedRoot.listFiles().orEmpty().filter { file ->
                    file.name in learnedRootNames || file.name.startsWith("UserHistoryDictionary")
                }
            }
        }
        return (activeLearnedFiles + learnedFilesInRestoreCopies)
            .distinctBy { it.canonicalPath }
    }
}
