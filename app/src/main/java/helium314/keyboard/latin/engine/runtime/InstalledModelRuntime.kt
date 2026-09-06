// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.engine.ModelKind
import helium314.keyboard.latin.engine.ModelManifest
import helium314.keyboard.latin.engine.ModelRegistry
import helium314.keyboard.latin.engine.ModelValidationLimits
import helium314.keyboard.latin.engine.OfficialContextModelPackImporter
import helium314.keyboard.latin.engine.OfficialModelPackImportResult
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import helium314.keyboard.latin.engine.onnx.InstalledNeuralModelLoader
import helium314.keyboard.latin.engine.onnx.LoadedSwipeModel
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import helium314.keyboard.latin.utils.Log
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.security.KeyFactory
import java.security.PublicKey
import java.security.interfaces.RSAPublicKey
import java.security.spec.X509EncodedKeySpec
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.zip.ZipFile

internal sealed interface SignedModelImportResult {
    data class Installed(val modelKind: ModelKind) : SignedModelImportResult
    data object Locked : SignedModelImportResult
    data object Unavailable : SignedModelImportResult
    data object Rejected : SignedModelImportResult
}

/** Device capability gate shared by automatic discovery and explicit local imports. */
internal object NeuralDevicePolicy {
    fun allowsModel(modelKind: ModelKind, isLowRamDevice: Boolean): Boolean =
        modelKind != ModelKind.CONTEXT_RESCORER || !isLowRamDevice

    fun allowsModel(modelKind: ModelKind, context: Context): Boolean = allowsModel(
        modelKind,
        context.getSystemService(ActivityManager::class.java)?.isLowRamDevice == true,
    )
}

/** Fixed, auditable acceptance policy shared by the bundled swipe and official context models. */
internal object OfficialModelPolicy {
    val allowedOperators = setOf(
        "Add", "And", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape", "Div", "Equal",
        "Erf", "Expand", "Gather", "Gemm", "LayerNormalization", "LessOrEqual", "MatMul", "Mod",
        "Mul", "Not", "Pow", "ReduceL2", "ReduceMean", "ReduceSum", "Reshape", "Shape", "Sigmoid",
        "Slice", "Softmax", "Sqrt", "Squeeze", "Sub", "Transpose", "Unsqueeze", "Where",
        "com.microsoft::GatherBlockQuantized", "com.microsoft::MatMulNBits",
    )

    fun limits(kind: ModelKind, appVersionCode: Int) = ModelValidationLimits(
        maximumBytes = when (kind) {
            ModelKind.SWIPE_CTC -> 2_621_440L
            ModelKind.CONTEXT_RESCORER -> 25_165_824L
        },
        allowedOperators = allowedOperators,
        acceptedModelKinds = setOf(kind),
        appVersionCode = appVersionCode,
        maximumParameterCount = when (kind) {
            ModelKind.SWIPE_CTC -> 1_000_000L
            ModelKind.CONTEXT_RESCORER -> 40_000_000L
        },
    )

    fun parsePublicKey(encoded: ByteArray): PublicKey {
        require(encoded.isNotEmpty() && encoded.size <= MAXIMUM_PUBLIC_KEY_BYTES)
        val key = KeyFactory.getInstance("RSA").generatePublic(X509EncodedKeySpec(encoded))
        require(key is RSAPublicKey && key.modulus.bitLength() >= MINIMUM_RSA_BITS)
        require(key.publicExponent.toLong() == 65_537L)
        return key
    }

    fun readPublicKey(input: InputStream): PublicKey {
        val output = java.io.ByteArrayOutputStream(MAXIMUM_PUBLIC_KEY_BYTES)
        val buffer = ByteArray(1024)
        var total = 0
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            total += count
            require(total <= MAXIMUM_PUBLIC_KEY_BYTES)
            output.write(buffer, 0, count)
        }
        return parsePublicKey(output.toByteArray())
    }

    fun shouldActivateBundledSwipe(
        installedBundleVersion: Int,
        appVersionCode: Int,
        hasActiveModel: Boolean,
    ): Boolean = !hasActiveModel || installedBundleVersion != appVersionCode

    private const val MAXIMUM_PUBLIC_KEY_BYTES = 8 * 1024
    private const val MINIMUM_RSA_BITS = 3072
}

/**
 * Installs and opens accepted models off the UI thread after first unlock. Missing packs, keys,
 * native code, or incompatible models are ordinary fallback states and are retried at a bounded
 * cadence when a new editor starts.
 */
internal object InstalledModelRuntime {
    private val monitor = Any()
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "LibreBoardModelBootstrap").apply { isDaemon = true }
    }
    private var refreshRunning = false
    private var lastAttemptMillis = Long.MIN_VALUE
    private var swipeModel: LoadedSwipeModel? = null
    private var swipeLexicon: SwipeLexicon? = null
    private var contextReady = false

    fun installFromUri(
        context: Context,
        uri: Uri,
        onComplete: (SignedModelImportResult) -> Unit,
    ) {
        val applicationContext = context.applicationContext
        executor.execute {
            val result = installFromUriBlocking(applicationContext, uri)
            runCatching { onComplete(result) }
        }
    }

    fun ensureLoaded(context: Context, lexicon: SwipeLexicon) {
        if (!BuildConfig.LIBREBOARD_ONNX_RUNTIME_PACKAGED || !BuildConfig.LIBREBOARD_SIGNED_MODELS_PACKAGED) {
            return
        }
        synchronized(monitor) {
            swipeModel?.takeIf { swipeLexicon !== lexicon }?.let { model ->
                swipeLexicon = lexicon
                LiveTypingEngine.installSwipe(model, lexicon)
            }
            if ((swipeModel != null && contextReady) || refreshRunning) return
            val now = android.os.SystemClock.elapsedRealtime()
            if (lastAttemptMillis != Long.MIN_VALUE && now - lastAttemptMillis < RETRY_INTERVAL_MILLIS) return
            lastAttemptMillis = now
            refreshRunning = true
        }
        val applicationContext = context.applicationContext
        executor.execute {
            try {
                refresh(applicationContext, lexicon)
            } catch (failure: Throwable) {
                Log.e(TAG, "Could not initialize an installed neural model; fallback remains active", failure)
            } finally {
                synchronized(monitor) { refreshRunning = false }
            }
        }
    }

    private fun refresh(context: Context, lexicon: SwipeLexicon) {
        val privateContext = CredentialEncryptedStorage.contextOrNull(context) ?: return
        val publicKey = loadProjectKey(context) ?: run {
            Log.w(TAG, "Signed model public key is unavailable; fallback remains active")
            return
        }

        if (synchronized(monitor) { swipeModel == null }) {
            val loaded = runCatching {
                val registry = ModelRegistry(
                    privateContext,
                    OfficialModelPolicy.limits(ModelKind.SWIPE_CTC, BuildConfig.VERSION_CODE),
                    publicKey,
                )
                val preferences = privateContext.getSharedPreferences(MODEL_STATE_PREFERENCES, Context.MODE_PRIVATE)
                val active = registry.activeModel()
                var activatedBundledModel = false
                if (OfficialModelPolicy.shouldActivateBundledSwipe(
                        preferences.getInt(BUNDLED_SWIPE_VERSION, 0),
                        BuildConfig.VERSION_CODE,
                        active != null,
                    )) {
                    runCatching {
                        context.assets.open(SWIPE_ARCHIVE_ASSET).use(registry::activate)
                    }.onSuccess {
                        activatedBundledModel = true
                        preferences.edit().putInt(BUNDLED_SWIPE_VERSION, BuildConfig.VERSION_CODE).commit()
                    }
                }
                var opened = InstalledNeuralModelLoader.openSwipe(registry)
                if (opened.component == null && activatedBundledModel) {
                    registry.rollbackActive()
                    opened = InstalledNeuralModelLoader.openSwipe(registry)
                }
                opened
            }.getOrNull()
            loaded?.component?.let { model ->
                synchronized(monitor) {
                    swipeModel = model
                    swipeLexicon = lexicon
                }
                LiveTypingEngine.installSwipe(model, lexicon)
            }
        }

        if (!NeuralDevicePolicy.allowsModel(ModelKind.CONTEXT_RESCORER, context)) {
            LiveTypingEngine.clearContext()
            synchronized(monitor) { contextReady = true }
            return
        }
        if (!synchronized(monitor) { contextReady }) {
            val loaded = runCatching {
                val registry = ModelRegistry(
                    privateContext,
                    OfficialModelPolicy.limits(ModelKind.CONTEXT_RESCORER, BuildConfig.VERSION_CODE),
                    publicKey,
                )
                var active = registry.activeModel()
                if (active == null) {
                    val result = OfficialContextModelPackImporter(context, registry).installFromOfficialPack()
                    if (result is OfficialModelPackImportResult.Installed) active = registry.activeModel()
                }
                active?.let { InstalledNeuralModelLoader.openContext(registry) }
            }.getOrNull()
            loaded?.component?.let { model ->
                LiveTypingEngine.installContext(model)
                synchronized(monitor) { contextReady = true }
            }
        }
    }

    private fun installFromUriBlocking(context: Context, uri: Uri): SignedModelImportResult {
        if (!BuildConfig.LIBREBOARD_ONNX_RUNTIME_PACKAGED || !BuildConfig.LIBREBOARD_SIGNED_MODELS_PACKAGED) {
            return SignedModelImportResult.Unavailable
        }
        val privateContext = CredentialEncryptedStorage.contextOrNull(context)
            ?: return SignedModelImportResult.Locked
        val publicKey = loadProjectKey(context) ?: return SignedModelImportResult.Unavailable
        val staged = try {
            stageArchive(privateContext, context, uri)
        } catch (failure: Exception) {
            Log.w(TAG, "Could not stage a local model archive", failure)
            return SignedModelImportResult.Rejected
        } ?: return SignedModelImportResult.Unavailable

        var activatedRegistry: ModelRegistry? = null
        return try {
            val manifest = readManifest(staged)
            if (
                manifest.modelKind == ModelKind.CONTEXT_RESCORER &&
                !NeuralDevicePolicy.allowsModel(manifest.modelKind, context)
            ) {
                return SignedModelImportResult.Unavailable
            }
            val registry = ModelRegistry(
                privateContext,
                OfficialModelPolicy.limits(manifest.modelKind, BuildConfig.VERSION_CODE),
                publicKey,
            )
            staged.inputStream().buffered().use(registry::activate)
            activatedRegistry = registry
            val result = when (manifest.modelKind) {
                ModelKind.SWIPE_CTC -> installImportedSwipe(registry)
                ModelKind.CONTEXT_RESCORER -> installImportedContext(registry)
            }
            if (result is SignedModelImportResult.Installed && result.modelKind == ModelKind.SWIPE_CTC) {
                privateContext.getSharedPreferences(MODEL_STATE_PREFERENCES, Context.MODE_PRIVATE)
                    .edit()
                    .putInt(BUNDLED_SWIPE_VERSION, BuildConfig.VERSION_CODE)
                    .commit()
            }
            result
        } catch (failure: Exception) {
            activatedRegistry?.rollbackActive()
            Log.w(TAG, "Rejected a local model archive", failure)
            SignedModelImportResult.Rejected
        } finally {
            if (!staged.delete()) Log.w(TAG, "Could not delete the staged local model archive")
        }
    }

    private fun installImportedSwipe(registry: ModelRegistry): SignedModelImportResult {
        val loaded = InstalledNeuralModelLoader.openSwipe(registry).component
            ?: run {
                registry.rollbackActive()
                return SignedModelImportResult.Rejected
            }
        val lexicon = synchronized(monitor) {
            swipeModel = loaded
            swipeLexicon
        }
        if (lexicon != null) LiveTypingEngine.installSwipe(loaded, lexicon)
        return SignedModelImportResult.Installed(ModelKind.SWIPE_CTC)
    }

    private fun installImportedContext(registry: ModelRegistry): SignedModelImportResult {
        val loaded = InstalledNeuralModelLoader.openContext(registry).component
            ?: run {
                registry.rollbackActive()
                return SignedModelImportResult.Rejected
            }
        LiveTypingEngine.installContext(loaded)
        synchronized(monitor) { contextReady = true }
        return SignedModelImportResult.Installed(ModelKind.CONTEXT_RESCORER)
    }

    private fun stageArchive(privateContext: Context, sourceContext: Context, uri: Uri): File? {
        val staged = File.createTempFile("model-import-", ".lbmodel", privateContext.cacheDir)
        try {
            val input = sourceContext.contentResolver.openInputStream(uri) ?: run {
                staged.delete()
                return null
            }
            input.use { source ->
                FileOutputStream(staged).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var total = 0L
                    while (true) {
                        val count = source.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= MAXIMUM_MODEL_ARCHIVE_BYTES) { "Model archive is too large" }
                        output.write(buffer, 0, count)
                    }
                    require(total > 0) { "Model archive is empty" }
                    output.fd.sync()
                }
            }
            return staged
        } catch (failure: Exception) {
            staged.delete()
            throw failure
        }
    }

    private fun readManifest(archiveFile: File): ModelManifest = ZipFile(archiveFile).use { archive ->
        val entry = requireNotNull(archive.getEntry(MODEL_MANIFEST_ENTRY)) { "Missing model manifest" }
        require(!entry.isDirectory && entry.size in 1..MAXIMUM_MODEL_MANIFEST_BYTES.toLong()) {
            "Model manifest is empty or too large"
        }
        val bytes = archive.getInputStream(entry).use { input ->
            val output = java.io.ByteArrayOutputStream(entry.size.toInt())
            val buffer = ByteArray(8192)
            var total = 0
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                total += count
                require(total <= MAXIMUM_MODEL_MANIFEST_BYTES) { "Model manifest is too large" }
                output.write(buffer, 0, count)
            }
            output.toByteArray()
        }
        Json.decodeFromString(bytes.decodeToString())
    }

    private fun loadProjectKey(context: Context): PublicKey? = runCatching {
        context.assets.open(PUBLIC_KEY_ASSET).use(OfficialModelPolicy::readPublicKey)
    }.getOrNull()

    private const val TAG = "InstalledModelRuntime"
    private const val PUBLIC_KEY_ASSET = "models/libreboard-model-signing-public.der"
    private const val SWIPE_ARCHIVE_ASSET = "models/swipe-latin-v1.lbmodel"
    private const val MODEL_STATE_PREFERENCES = "installed-model-runtime"
    private const val BUNDLED_SWIPE_VERSION = "bundled-swipe-version"
    private const val MODEL_MANIFEST_ENTRY = "manifest.json"
    private const val MAXIMUM_MODEL_ARCHIVE_BYTES = 28L * 1024L * 1024L
    private const val MAXIMUM_MODEL_MANIFEST_BYTES = 256 * 1024
    private const val RETRY_INTERVAL_MILLIS = 60_000L
}
