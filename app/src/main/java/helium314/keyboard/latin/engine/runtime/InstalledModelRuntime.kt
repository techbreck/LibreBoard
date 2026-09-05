// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import android.app.ActivityManager
import android.content.Context
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.engine.ModelKind
import helium314.keyboard.latin.engine.ModelRegistry
import helium314.keyboard.latin.engine.ModelValidationLimits
import helium314.keyboard.latin.engine.OfficialContextModelPackImporter
import helium314.keyboard.latin.engine.OfficialModelPackImportResult
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import helium314.keyboard.latin.engine.onnx.InstalledNeuralModelLoader
import helium314.keyboard.latin.engine.onnx.LoadedSwipeModel
import helium314.keyboard.latin.privacy.CredentialEncryptedStorage
import helium314.keyboard.latin.utils.Log
import java.io.InputStream
import java.security.KeyFactory
import java.security.PublicKey
import java.security.interfaces.RSAPublicKey
import java.security.spec.X509EncodedKeySpec
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

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
        if (CredentialEncryptedStorage.contextOrNull(context) == null) return
        val publicKey = runCatching {
            context.assets.open(PUBLIC_KEY_ASSET).use {
                OfficialModelPolicy.readPublicKey(it)
            }
        }.getOrElse {
            Log.w(TAG, "Signed model public key is unavailable; fallback remains active")
            return
        }

        if (synchronized(monitor) { swipeModel == null }) {
            val loaded = runCatching {
                val registry = ModelRegistry(
                    context,
                    OfficialModelPolicy.limits(ModelKind.SWIPE_CTC, BuildConfig.VERSION_CODE),
                    publicKey,
                )
                val active = runCatching {
                    context.assets.open(SWIPE_ARCHIVE_ASSET).use(registry::activate)
                }.getOrElse { registry.activeModel() }
                active?.let { InstalledNeuralModelLoader.openSwipe(registry) }
            }.getOrNull()
            loaded?.component?.let { model ->
                synchronized(monitor) {
                    swipeModel = model
                    swipeLexicon = lexicon
                }
                LiveTypingEngine.installSwipe(model, lexicon)
            }
        }

        val lowRam = context.getSystemService(ActivityManager::class.java)?.isLowRamDevice == true
        if (lowRam) {
            LiveTypingEngine.clearContext()
            synchronized(monitor) { contextReady = true }
            return
        }
        if (!synchronized(monitor) { contextReady }) {
            val loaded = runCatching {
                val registry = ModelRegistry(
                    context,
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

    private const val TAG = "InstalledModelRuntime"
    private const val PUBLIC_KEY_ASSET = "models/libreboard-model-signing-public.der"
    private const val SWIPE_ARCHIVE_ASSET = "models/swipe-latin-v1.lbmodel"
    private const val RETRY_INTERVAL_MILLIS = 60_000L
}
