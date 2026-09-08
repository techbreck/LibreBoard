// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.os.Build
import android.os.Debug
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.engine.ctc.CtcSwipeFeatures
import helium314.keyboard.latin.engine.onnx.OnnxCtcInferenceSession
import helium314.keyboard.latin.engine.context.ContextModelBatch
import helium314.keyboard.latin.engine.onnx.LibreBoardOnnxContracts
import helium314.keyboard.latin.engine.onnx.OnnxContextInferenceSession
import helium314.keyboard.latin.engine.onnx.OnnxRuntimeSessionFactory
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Explicit local fixture injection only. Does not install a model or qualify any release gate. */
@RunWith(AndroidJUnit4::class)
class ContextRuntimeInstrumentedTest {
    @Test fun sourceBuiltAndroidRuntimeMatchesCheckedHostContextScores() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardRequireContextRuntime") == "true")
        assertTrue("build with the verified source-built runtime", BuildConfig.LIBREBOARD_ONNX_RUNTIME_PACKAGED)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "context-runtime-smoke")
        val fixtureFile = File(directory, "fixture.json")
        val model = File(directory, "context.onnx")
        assertTrue(fixtureFile.length() in 1..65_536)
        assertTrue(model.length() in 1..(24L * 1024 * 1024))
        fun sha256(file: File): String {
            val hash = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer = ByteArray(65_536)
                while (true) {
                    val size = stream.read(buffer)
                    if (size < 0) break
                    hash.update(buffer, 0, size)
                }
            }
            return hash.digest().joinToString("") { "%02x".format(it) }
        }
        val fixtureHash = sha256(fixtureFile)
        assertEquals("supply the independently generated fixture hash", args.getString("contextFixtureSha256"), fixtureHash)
        val fixture = JSONObject(fixtureFile.readText())
        assertEquals(1, fixture.getInt("schemaVersion"))
        assertEquals("context-synthetic-kernel-parity-v1", fixture.getString("protocol"))
        assertFalse(fixture.getBoolean("releaseEligible"))
        assertEquals(fixture.getString("modelSha256"), sha256(model))
        assertEquals(0.001, fixture.getDouble("absoluteTolerance"), 0.0)
        fun memorySnapshot(): JSONObject {
            val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
            val runtime = Runtime.getRuntime()
            return JSONObject().put("processPssKiB", memory.totalPss)
                .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
                .put("javaHeapUsedBytes", runtime.totalMemory() - runtime.freeMemory())
        }
        val combined = args.getString("libreboardRequireCombinedRuntime") == "true"
        val timingIterations = args.getString("contextTimingIterations")?.let {
            requireNotNull(it.toIntOrNull()) { "contextTimingIterations must be an integer" }
        } ?: 0
        require(timingIterations == 0 || timingIterations in 20..100) {
            "contextTimingIterations must be zero or between 20 and 100"
        }
        val swipeModel = File(directory, "swipe.onnx")
        val swipeHash = if (combined) {
            assertTrue(swipeModel.length() in 1..(3L * 1024 * 1024))
            sha256(swipeModel).also { assertEquals(args.getString("swipeModelSha256"), it) }
        } else null
        val memoryBeforeOpen = memorySnapshot()
        val sampler = if (combined) RuntimeMemorySampler() else null
        var swipeSession: OnnxCtcInferenceSession? = null
        var memoryAfterSwipeOpen: JSONObject? = null
        val results = JSONArray()
        var memoryAfterOpen: JSONObject? = null
        try {
            if (combined) {
                val swipeOpened = OnnxRuntimeSessionFactory.open(swipeModel, LibreBoardOnnxContracts.swipeCtc)
                assertEquals(EngineAvailability.AVAILABLE, swipeOpened.availability)
                swipeSession = OnnxCtcInferenceSession(requireNotNull(swipeOpened.session))
                memoryAfterSwipeOpen = memorySnapshot()
            }
            val opened = OnnxRuntimeSessionFactory.open(model, LibreBoardOnnxContracts.contextEnDe)
            assertEquals("the packaged kernels must load the context graph", EngineAvailability.AVAILABLE, opened.availability)
            memoryAfterOpen = memorySnapshot()
            OnnxContextInferenceSession(requireNotNull(opened.session)).use { session ->
                val cases = fixture.getJSONArray("cases")
                assertEquals(3, cases.length())
                for ((index, rows) in listOf(1, 8, 32).withIndex()) {
                    val case = cases.getJSONObject(index)
                    assertEquals(rows, case.getInt("rows"))
                    val expected = case.getJSONArray("expectedScores")
                    assertEquals(rows, expected.length())
                    val ids = LongArray(rows * 32)
                    val attention = LongArray(rows * 32)
                    val candidates = FloatArray(rows * 32)
                    repeat(rows) { row ->
                        val offset = row * 32
                        listOf(1L, 3L, 6L, 7L).forEachIndexed { column, id ->
                            ids[offset + column] = id
                            attention[offset + column] = 1
                        }
                        for (column in 24..25) {
                            ids[offset + column] = (column - 16).toLong()
                            attention[offset + column] = 1
                            candidates[offset + column] = 1f
                        }
                    }
                    val batch = ContextModelBatch(rows, 32, ids, attention, candidates, LongArray(rows))
                    val started = SystemClock.elapsedRealtimeNanos()
                    val output = session.infer(batch, Deadline.afterMillis(60_000))
                    val elapsed = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
                    assertEquals(EngineAvailability.AVAILABLE, output.availability)
                    val scores = requireNotNull(output.candidateLogLikelihoods)
                    assertEquals(rows, scores.size)
                    scores.forEachIndexed { row, score ->
                        assertTrue(score.isFinite())
                        val reference = expected.getDouble(row)
                        assertTrue(reference.isFinite())
                        assertEquals("Android/host score parity for batch $rows row $row", reference, score.toDouble(), 0.001)
                    }
                    val memoryAfterContextInference = memorySnapshot()
                    swipeSession?.let { swipe ->
                        // Fixed-size synthetic tensors exercise both loaded graphs without making
                        // a swipe quality, decoder-memory, or live-fusion claim.
                        val centers = FloatArray(128).also {
                            it[0] = 0.25f; it[1] = 0.5f; it[2] = 0.75f; it[3] = 0.5f
                        }
                        val mask = FloatArray(64).also { it[0] = 1f; it[1] = 1f }
                        val labels = MutableList<String?>(64) { null }.also { it[0] = "a"; it[1] = "b" }
                        val path = FloatArray(128) { if (it % 2 == 0) (it / 2) / 63f else 0.5f }
                        val swipeOutput = swipe.infer(CtcSwipeFeatures(path, centers, mask, labels), Deadline.afterMillis(60_000))
                        assertEquals(EngineAvailability.AVAILABLE, swipeOutput.availability)
                        val logits = requireNotNull(swipeOutput.logits)
                        assertEquals(32 * 65, logits.size)
                        assertTrue(logits.all(Float::isFinite))
                    }
                    val repeatedTiming = if (timingIterations > 0) {
                        fun checkedInference(): Double {
                            val start = SystemClock.elapsedRealtimeNanos()
                            val inference = session.infer(batch, Deadline.afterMillis(60_000))
                            val milliseconds = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
                            assertEquals(EngineAvailability.AVAILABLE, inference.availability)
                            val values = requireNotNull(inference.candidateLogLikelihoods)
                            assertEquals(rows, values.size)
                            values.forEachIndexed { row, value ->
                                assertTrue(value.isFinite())
                                assertEquals(expected.getDouble(row), value.toDouble(), 0.001)
                            }
                            return milliseconds
                        }
                        repeat(3) { checkedInference() }
                        val times = List(timingIterations) { checkedInference() }
                        val sorted = times.sorted()
                        fun percentile(value: Double): Double = sorted[
                            (kotlin.math.ceil(value * sorted.size).toInt() - 1).coerceIn(sorted.indices)
                        ]
                        JSONObject().put("warmupIterations", 3).put("measuredIterations", timingIterations)
                            .put("samplesMs", JSONArray(times)).put("p50Ms", percentile(0.50))
                            .put("p95Ms", percentile(0.95)).put("p99Ms", percentile(0.99))
                            .put("scope", "Repeated synthetic context kernel only; memory sampler and host contention may affect timings")
                    } else null
                    results.put(JSONObject().put("rows", rows).put("singleRunMs", elapsed)
                        .put("repeatedTiming", repeatedTiming ?: JSONObject.NULL)
                        .put("memoryAfterInference", memoryAfterContextInference)
                        .put("memoryAfterSwipeInference", if (combined) memorySnapshot() else JSONObject.NULL))
                }
            }
        } finally {
            try { swipeSession?.close() } finally { sampler?.close() }
        }
        File(directory, "android-report.json").writeText(JSONObject()
            .put("schemaVersion", 1).put("releaseEligible", false).put("syntheticKernelParityOnly", true)
            .put("fixtureSha256", fixtureHash).put("modelSha256", sha256(model))
            .put("combinedModelSnapshots", combined).put("swipeModelSha256", swipeHash ?: JSONObject.NULL)
            .put("memoryAfterSwipeOpen", memoryAfterSwipeOpen ?: JSONObject.NULL)
            .put("apkSha256", sha256(File(context.applicationInfo.sourceDir)))
            .put("buildFingerprint", Build.FINGERPRINT).put("supportedAbis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            .put("memoryBeforeOpen", memoryBeforeOpen).put("memoryAfterOpen", memoryAfterOpen)
            .put("memoryAfterClose", memorySnapshot())
            .put("sampledMemory", sampler?.report() ?: JSONObject.NULL)
            .put("memoryScope", "Whole-process snapshots, not isolated added peak memory or a release gate")
            .put("cases", results).toString() + "\n")
    }
}

/** Test-only sampling can miss short peaks and adds overhead; it is not release qualification. */
private class RuntimeMemorySampler : AutoCloseable {
    private val running = AtomicBoolean(true)
    private val samples = AtomicInteger()
    private val maximumPss = AtomicInteger()
    private val maximumNative = AtomicLong()
    private val maximumJava = AtomicLong()
    private val failure = AtomicReference<Throwable?>()
    private val worker = Thread({
        try {
            while (running.get()) {
                val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
                val runtime = Runtime.getRuntime()
                maximumPss.accumulateAndGet(memory.totalPss, ::maxOf)
                maximumNative.accumulateAndGet(Debug.getNativeHeapAllocatedSize(), ::maxOf)
                maximumJava.accumulateAndGet(runtime.totalMemory() - runtime.freeMemory(), ::maxOf)
                samples.incrementAndGet()
                Thread.sleep(5)
            }
        } catch (_: InterruptedException) {
            // Explicit close interrupts the sampling delay.
        } catch (error: Throwable) {
            failure.set(error)
        }
    }, "LibreBoardMemoryProbe").apply { isDaemon = true; start() }

    override fun close() {
        running.set(false)
        worker.interrupt()
        worker.join(2_000)
        check(!worker.isAlive) { "memory sampler did not stop" }
        check(failure.get() == null) { "memory sampler failed: ${failure.get()}" }
        check(samples.get() > 0) { "memory sampler did not collect a sample" }
    }

    fun report(): JSONObject = JSONObject()
        .put("samples", samples.get()).put("requestedDelayBetweenSamplesMs", 5)
        .put("maximumProcessPssKiB", maximumPss.get())
        .put("maximumNativeHeapAllocatedBytes", maximumNative.get())
        .put("maximumJavaHeapUsedBytes", maximumJava.get())
        .put("scope", "Observed whole-process maxima at different times; do not sum heap maxima or infer a release peak pass")
}
