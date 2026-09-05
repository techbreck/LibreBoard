// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.settings.preferences

import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.activity.compose.ManagedActivityResultLauncher
import androidx.activity.result.ActivityResult
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import helium314.keyboard.dictionarypack.DictionaryPackConstants
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.keyboard.emoji.SupportedEmojis
import helium314.keyboard.latin.R
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.DeviceProtectedUtils
import helium314.keyboard.latin.utils.ExecutorUtils
import helium314.keyboard.latin.utils.LayoutUtilsCustom
import helium314.keyboard.latin.utils.Log
import helium314.keyboard.latin.utils.SubtypeSettings
import helium314.keyboard.latin.utils.getActivity
import helium314.keyboard.latin.utils.prefs
import helium314.keyboard.latin.utils.protectedPrefs
import helium314.keyboard.settings.Setting
import helium314.keyboard.settings.SettingsActivity
import helium314.keyboard.settings.dialogs.ConfirmationDialog
import helium314.keyboard.settings.dialogs.InfoDialog
import helium314.keyboard.settings.filePicker
import kotlinx.serialization.json.Json
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import androidx.core.content.edit
import helium314.keyboard.latin.checkVersionUpgrade
import helium314.keyboard.latin.transferOldPinnedClips

@Composable
fun BackupRestorePreference(setting: Setting) {
    var showDialog by rememberSaveable { mutableStateOf(false) }
    val ctx = LocalContext.current
    var error: String? by rememberSaveable { mutableStateOf(null) }
    val backupLauncher = backupLauncher { error = it }
    val restoreLauncher = restoreLauncher { error = it }
    Preference(name = setting.title, onClick = { showDialog = true })
    if (showDialog) {
        ConfirmationDialog(
            onDismissRequest = { showDialog = false },
            title = { Text(stringResource(R.string.backup_restore_title)) },
            content = { Text(stringResource(R.string.backup_restore_message)) },
            confirmButtonText = stringResource(R.string.button_backup),
            neutralButtonText = stringResource(R.string.button_restore),
            onNeutral = {
                showDialog = false
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
                    .addCategory(Intent.CATEGORY_OPENABLE)
                    .setType("application/zip")
                restoreLauncher.launch(intent)
            },
            onConfirmed = {
                val currentDate = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Calendar.getInstance().time)
                val intent = Intent(Intent.ACTION_CREATE_DOCUMENT)
                    .addCategory(Intent.CATEGORY_OPENABLE)
                    .putExtra(
                        Intent.EXTRA_TITLE,
                        ctx.getString(R.string.english_ime_name)
                            .replace(" ", "_") + "_backup_$currentDate.zip"
                    )
                    .setType("application/zip")
                backupLauncher.launch(intent)
            }
        )
    }
    if (error != null) {
        InfoDialog(
            if (error!!.startsWith("b"))
                stringResource(R.string.backup_error, error!!.drop(1))
            else stringResource(R.string.restore_error, error!!.drop(1))
        ) { error = null }
    }
}

@Composable
private fun backupLauncher(onError: (String) -> Unit): ManagedActivityResultLauncher<Intent, ActivityResult> {
    val ctx = LocalContext.current
    return filePicker { uri ->
        ExecutorUtils.getBackgroundExecutor(ExecutorUtils.KEYBOARD).execute {
            try {
                val filesDir = ctx.filesDir
                val protectedFilesDir = DeviceProtectedUtils.getFilesDir(ctx)
                val files = backupFiles(filesDir).map { it.relativeTo(filesDir).invariantSeparatorsPath to it }
                val protectedFiles = backupFiles(protectedFilesDir).map {
                    "$DEVICE_PROTECTED_PREFIX${it.relativeTo(protectedFilesDir).invariantSeparatorsPath}" to it
                }
                val preferences = settingsToJsonBytes(ctx.prefs().all)
                val protectedPreferences = settingsToJsonBytes(ctx.protectedPrefs().all)
                val personalData = PersonalizationRuntime.export(ctx)
                require(preferences.size <= MAX_PREFERENCES_BYTES) { "settings payload is too large" }
                require(protectedPreferences.size <= MAX_PREFERENCES_BYTES) {
                    "protected settings payload is too large"
                }
                require(personalData == null || personalData.size <= MAX_PERSONAL_DATA_BYTES) {
                    "personalization payload is too large"
                }
                val fileEntries = files + protectedFiles
                fileEntries.forEach { (_, file) ->
                    require(file.length() <= BackupArchivePolicy.MAX_FILE_BYTES) {
                        "backup file ${file.name} is too large"
                    }
                }
                val entryCount = fileEntries.size + 3 + if (personalData == null) 0 else 1
                require(entryCount <= BackupArchivePolicy.MAX_ARCHIVE_ENTRIES) {
                    "backup has too many entries"
                }
                val totalBytes = fileEntries.sumOf { it.second.length() } + preferences.size +
                    protectedPreferences.size + BACKUP_MANIFEST.length + (personalData?.size ?: 0)
                require(totalBytes <= BackupArchivePolicy.MAX_TOTAL_UNCOMPRESSED_BYTES) {
                    "backup data exceeds the archive size limit"
                }

                val output = requireNotNull(ctx.contentResolver.openOutputStream(uri)) {
                    "could not open the selected backup destination"
                }
                output.use { stream ->
                    ZipOutputStream(stream).use { zip ->
                        fileEntries.sortedBy { it.first }.forEach { (name, file) ->
                            zip.putDeterministicEntry(name)
                            FileInputStream(file).buffered().use { it.copyTo(zip) }
                            zip.closeEntry()
                        }
                        zip.writeEntry(BACKUP_MANIFEST_FILE_NAME, BACKUP_MANIFEST.encodeToByteArray())
                        zip.writeEntry(PREFS_FILE_NAME, preferences)
                        zip.writeEntry(PROTECTED_PREFS_FILE_NAME, protectedPreferences)
                        personalData?.let { zip.writeEntry(PERSONAL_DATA_FILE_NAME, it) }
                    }
                }
            } catch (t: Throwable) {
                Handler(Looper.getMainLooper()).post { onError("b" + (t.message ?: "unknown error")) }
                Log.w("AdvancedScreen", "error during backup", t)
            }
        }
    }
}

@Composable
private fun restoreLauncher(onError: (String) -> Unit): ManagedActivityResultLauncher<Intent, ActivityResult> {
    val ctx = LocalContext.current
    return filePicker { uri ->
        ExecutorUtils.getBackgroundExecutor(ExecutorUtils.KEYBOARD).execute {
            var oldPrefs: Map<String, Any?>? = null
            var oldProtectedPrefs: Map<String, Any?>? = null
            var oldPersonalData: ByteArray? = null
            var transaction: BackupRestoreTransaction.Transaction? = null
            var personalDataChanged = false
            var restoreCommitted = false
            var success = false
            var failureMessage: String? = null
            try {
                oldPrefs = ctx.prefs().all.toMap()
                oldProtectedPrefs = ctx.protectedPrefs().all.toMap()
                oldPersonalData = PersonalizationRuntime.export(ctx)
                val activeFilesDir = ctx.filesDir
                val activeDeviceProtectedFilesDir = DeviceProtectedUtils.getFilesDir(ctx)
                val restoreTransaction = BackupRestoreTransaction.begin(
                    activeFilesDir,
                    activeDeviceProtectedFilesDir,
                    ctx.noBackupFilesDir,
                    ::isBackupFilePath,
                )
                transaction = restoreTransaction
                val stagedFilesDir = restoreTransaction.stagedCredential
                val stagedDeviceProtectedFilesDir = restoreTransaction.stagedDevice
                require(!stagedFilesDir.exists() && !stagedDeviceProtectedFilesDir.exists()) {
                    "restore staging path already exists"
                }
                restoreTransaction.stageSelectedFiles()
                var anyMatch = false
                var foundManifest = false
                var restoredPrefs: Map<String, Any?>? = null
                var restoredProtectedPrefs: Map<String, Any?>? = null
                var restoredPersonalData: ByteArray? = null
                val readBudget = BackupArchivePolicy.ReadBudget()
                val seenEntries = HashSet<String>()
                val input = requireNotNull(ctx.contentResolver.openInputStream(uri)) {
                    "could not open the selected backup"
                }
                input.use { inputStream ->
                    ZipInputStream(inputStream).use { zip ->
                        var entry: ZipEntry? = zip.nextEntry
                        LayoutUtilsCustom.onLayoutFileChanged()
                        Settings.getInstance().stopListener()
                        while (entry != null) {
                            readBudget.beginEntry()
                            require(!entry.isDirectory) { "directory entries are not supported" }
                            require(seenEntries.add(entry.name)) { "duplicate backup entry" }
                            if (entry.name.startsWith(DEVICE_PROTECTED_PREFIX)) {
                                val adjustedName = entry.name.substringAfter(DEVICE_PROTECTED_PREFIX)
                                if (backupFilePatterns.any { adjustedName.matches(it) }) {
                                    require(restoreEntryToDir(
                                        zip,
                                        activeDeviceProtectedFilesDir,
                                        adjustedName,
                                        readBudget,
                                    )) { "unsafe device-protected backup entry" }
                                    anyMatch = true
                                } else {
                                    readBudget.drain(zip)
                                }
                            } else if (backupFilePatterns.any { entry.name.matches(it) }) {
                                require(restoreEntryToDir(zip, activeFilesDir, entry.name, readBudget)) {
                                    "unsafe credential-protected backup entry"
                                }
                                anyMatch = true
                            } else if (entry.name == BACKUP_MANIFEST_FILE_NAME) {
                                require(readBudget.read(zip, 4096).decodeToString() == BACKUP_MANIFEST) {
                                    "unsupported backup format"
                                }
                                foundManifest = true
                            } else if (entry.name == PREFS_FILE_NAME) {
                                val prefLines = readBudget.read(zip, MAX_PREFERENCES_BYTES.toLong()).decodeToString().split("\n")
                                restoredPrefs = parseBackupSettings(prefLines)
                                    ?: throw IllegalArgumentException("invalid settings payload")
                                anyMatch = true
                            } else if (entry.name == PROTECTED_PREFS_FILE_NAME) {
                                val prefLines = readBudget.read(zip, MAX_PREFERENCES_BYTES.toLong()).decodeToString().split("\n")
                                restoredProtectedPrefs = parseBackupSettings(prefLines)
                                    ?: throw IllegalArgumentException("invalid protected settings payload")
                                anyMatch = true
                            } else if (entry.name == PERSONAL_DATA_FILE_NAME) {
                                restoredPersonalData = readBudget.read(zip, MAX_PERSONAL_DATA_BYTES.toLong())
                                anyMatch = true
                            } else {
                                // Legacy archives may contain files no longer restored. Consume them
                                // under the same per-entry and total expansion limits.
                                readBudget.drain(zip)
                            }
                            zip.closeEntry()
                            entry = zip.nextEntry
                        }
                    }
                }
                if (!anyMatch)
                    throw Exception("nothing to restore in the given file")
                // Legacy HeliBoard archives may omit the LibreBoard manifest, but only a
                // versioned LibreBoard archive may replace the new personal/rejection store.
                if (restoredPersonalData != null) {
                    require(foundManifest) { "personal data requires a versioned LibreBoard backup" }
                }
                restoreTransaction.markCommitted()
                restoredPrefs?.let {
                    require(replacePreferences(ctx.prefs(), it)) { "could not restore settings" }
                }
                restoredProtectedPrefs?.let {
                    require(replacePreferences(ctx.protectedPrefs(), it)) {
                        "could not restore protected settings"
                    }
                }
                if (restoredPersonalData != null) {
                    PersonalizationRuntime.restore(ctx, restoredPersonalData)
                    personalDataChanged = true
                }
                restoreCommitted = true
                restoreTransaction.cleanupCommitted()
                success = true
            } catch (t: Throwable) {
                val rollbackFailures = mutableListOf<Throwable>()
                if (!restoreCommitted) {
                    transaction?.let { restoreTransaction ->
                        runCatching { restoreTransaction.rollback() }.onFailure(rollbackFailures::add)
                    }
                    if (personalDataChanged) {
                        runCatching {
                            oldPersonalData?.let { PersonalizationRuntime.restore(ctx, it) }
                                ?: PersonalizationRuntime.wipeRequired(ctx)
                        }.onFailure(rollbackFailures::add)
                    }
                    oldPrefs?.let { snapshot ->
                        runCatching { require(replacePreferences(ctx.prefs(), snapshot)) }
                            .onFailure(rollbackFailures::add)
                    }
                    oldProtectedPrefs?.let { snapshot ->
                        runCatching { require(replacePreferences(ctx.protectedPrefs(), snapshot)) }
                            .onFailure(rollbackFailures::add)
                    }
                }
                failureMessage = buildString {
                    append(t.message ?: "unknown error")
                    if (rollbackFailures.isNotEmpty()) append("; rollback was incomplete")
                }
                Log.w("AdvancedScreen", "error during restore", t)
                rollbackFailures.forEach { Log.w("AdvancedScreen", "error rolling back restore", it) }
            } finally {
                Handler(Looper.getMainLooper()).post {
                    if (success) {
                        Toast.makeText(ctx, ctx.getString(R.string.backup_restored), Toast.LENGTH_LONG).show()
                    } else {
                        onError("r" + (failureMessage ?: "unknown error"))
                    }
                    finishRestore(ctx)
                }
            }
        }
    }
}

private fun finishRestore(ctx: Context) {
    checkVersionUpgrade(ctx)
    transferOldPinnedClips(ctx)
    Settings.getInstance().startListener()
    SubtypeSettings.reloadEnabledSubtypes(ctx)
    ctx.sendBroadcast(Intent(DictionaryPackConstants.NEW_DICTIONARY_INTENT_ACTION).setPackage(ctx.packageName))
    LayoutUtilsCustom.onLayoutFileChanged()
    LayoutUtilsCustom.removeMissingLayouts(ctx)
    (ctx.getActivity() as? SettingsActivity)?.prefChanged()
    SupportedEmojis.load(ctx)
    KeyboardSwitcher.getInstance().setThemeNeedsReload()
}

@Suppress("UNCHECKED_CAST") // it is checked... but whatever (except string set, because can't check for that))
private fun settingsToJsonStream(settings: Map<String?, Any?>, out: OutputStream) {
    val booleans = settings.filter { it.key is String && it.value is Boolean } as Map<String, Boolean>
    val ints = settings.filter { it.key is String && it.value is Int } as Map<String, Int>
    val longs = settings.filter { it.key is String && it.value is Long } as Map<String, Long>
    val floats = settings.filter { it.key is String && it.value is Float } as Map<String, Float>
    val strings = settings.filter { it.key is String && it.value is String } as Map<String, String>
    val stringSets = settings.filter { it.key is String && it.value is Set<*> } as Map<String, Set<String>>
    // now write
    out.write("boolean settings\n".toByteArray())
    out.write(Json.encodeToString(booleans).toByteArray())
    out.write("\nint settings\n".toByteArray())
    out.write(Json.encodeToString(ints).toByteArray())
    out.write("\nlong settings\n".toByteArray())
    out.write(Json.encodeToString(longs).toByteArray())
    out.write("\nfloat settings\n".toByteArray())
    out.write(Json.encodeToString(floats).toByteArray())
    out.write("\nstring settings\n".toByteArray())
    out.write(Json.encodeToString(strings).toByteArray())
    out.write("\nstring set settings\n".toByteArray())
    out.write(Json.encodeToString(stringSets).toByteArray())
}

internal fun parseBackupSettings(lines: List<String>): Map<String, Any?>? = runCatching {
    val normalized = if (lines.lastOrNull().isNullOrEmpty()) lines.dropLast(1) else lines
    val i = normalized.iterator()
    val values = LinkedHashMap<String, Any?>()
    SETTINGS_SECTION_HEADERS.forEach { expectedHeader ->
        require(i.hasNext() && i.next() == expectedHeader) { "missing or reordered settings section" }
        require(i.hasNext()) { "missing settings payload" }
        val decoded: Map<String, Any?> = when (expectedHeader) {
            "boolean settings" -> Json.decodeFromString<Map<String, Boolean>>(i.next())
            "int settings" -> Json.decodeFromString<Map<String, Int>>(i.next())
            "long settings" -> Json.decodeFromString<Map<String, Long>>(i.next())
            "float settings" -> Json.decodeFromString<Map<String, Float>>(i.next())
            "string settings" -> Json.decodeFromString<Map<String, String>>(i.next())
            "string set settings" -> Json.decodeFromString<Map<String, Set<String>>>(i.next())
            else -> error("unsupported settings section")
        }
        decoded.forEach { (key, value) ->
            require(values.put(key, value) == null) { "duplicate settings key" }
        }
    }
    require(!i.hasNext()) { "unexpected settings data" }
    values
}.getOrNull()

private fun backupFiles(root: File): List<File> {
    val canonicalRoot = root.canonicalFile
    return root.walkTopDown().filter(File::isFile).mapNotNull { file ->
        val relative = file.relativeTo(root).invariantSeparatorsPath
        if (!isBackupFilePath(relative)) return@mapNotNull null
        require(BackupArchivePolicy.safeTarget(canonicalRoot, relative) == file.canonicalFile) {
            "backup file escapes private storage"
        }
        file
    }.sortedBy { it.relativeTo(root).invariantSeparatorsPath }.toList()
}

private fun settingsToJsonBytes(settings: Map<String?, Any?>): ByteArray =
    ByteArrayOutputStream().use { output ->
        settingsToJsonStream(settings, output)
        output.toByteArray()
    }

private fun ZipOutputStream.putDeterministicEntry(name: String) {
    require(BackupArchivePolicy.safeTarget(File("/archive-root"), name) != null) {
        "unsafe backup entry name"
    }
    putNextEntry(ZipEntry(name).apply { time = 0L })
}

private fun ZipOutputStream.writeEntry(name: String, content: ByteArray) {
    putDeterministicEntry(name)
    write(content)
    closeEntry()
}

private val SETTINGS_SECTION_HEADERS = listOf(
    "boolean settings",
    "int settings",
    "long settings",
    "float settings",
    "string settings",
    "string set settings",
)

private fun replacePreferences(prefs: SharedPreferences, values: Map<String, Any?>): Boolean {
    val editor = prefs.edit().clear()
    values.forEach { (key, value) ->
        when (value) {
            is String -> editor.putString(key, value)
            is Int -> editor.putInt(key, value)
            is Long -> editor.putLong(key, value)
            is Float -> editor.putFloat(key, value)
            is Boolean -> editor.putBoolean(key, value)
            is Set<*> -> {
                require(value.all { it is String }) { "invalid string-set setting" }
                editor.putStringSet(key, value.filterIsInstance<String>().toSet())
            }
            else -> throw IllegalArgumentException("unsupported setting type")
        }
    }
    return editor.commit()
}

private fun restoreEntryToDir(
    zip: ZipInputStream,
    baseDir: File,
    entryName: String,
    budget: BackupArchivePolicy.ReadBudget,
): Boolean {
    val file = BackupArchivePolicy.safeTarget(baseDir, entryName) ?: return false
    budget.copyTo(zip, file)
    return true
}

internal fun isBackupFilePath(path: String): Boolean = backupFilePatterns.any { path.matches(it) }

private const val PREFS_FILE_NAME = "preferences.json"
private const val PROTECTED_PREFS_FILE_NAME = "protected_preferences.json"
private const val PERSONAL_DATA_FILE_NAME = "personalization.json"
private const val BACKUP_MANIFEST_FILE_NAME = "libreboard-backup.json"
private const val BACKUP_MANIFEST = "{\"schemaVersion\":1,\"clipboardIncluded\":false}"
private const val MAX_PREFERENCES_BYTES = 2 * 1024 * 1024
private const val MAX_PERSONAL_DATA_BYTES = 16 * 1024 * 1024
private const val DEVICE_PROTECTED_PREFIX = "unprotected/"

private val backupFilePatterns by lazy { listOf(
    "blacklists${File.separator}.*\\.txt".toRegex(),
    "layouts${File.separator}.*${LayoutUtilsCustom.CUSTOM_LAYOUT_PREFIX}+\\..{0,4}".toRegex(), // can't expect a period at the end, as this would break restoring older backups
    "dicts${File.separator}.*${File.separator}.*user\\.dict".toRegex(),
    "UserHistoryDictionary.*${File.separator}UserHistoryDictionary.*\\.(body|header)".toRegex(),
    "custom_background_image.*".toRegex(),
    "custom_font".toRegex(),
    "custom_emoji_font".toRegex(),
) }
