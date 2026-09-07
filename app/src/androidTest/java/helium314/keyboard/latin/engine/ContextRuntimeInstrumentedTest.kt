// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.os.Build
import android.os.Debug
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.engine.context.ContextModelBatch
import helium314.keyboard.latin.engine.onnx.LibreBoardOnnxContracts
import helium314.keyboard.latin.engine.onnx.OnnxContextInferenceSession
import helium314.keyboard.latin.engine.onnx.OnnxRuntimeSessionFactory
import java.io.File
import java.security.MessageDigest
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
        val memoryBeforeOpen = memorySnapshot()
        val opened = OnnxRuntimeSessionFactory.open(model, LibreBoardOnnxContracts.contextEnDe)
        assertEquals("the packaged kernels must load the context graph", EngineAvailability.AVAILABLE, opened.availability)
        val memoryAfterOpen = memorySnapshot()
        val results = JSONArray()
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
                results.put(JSONObject().put("rows", rows).put("singleRunMs", elapsed)
                    .put("memoryAfterInference", memorySnapshot()))
            }
        }
        File(directory, "android-report.json").writeText(JSONObject()
            .put("schemaVersion", 1).put("releaseEligible", false).put("syntheticKernelParityOnly", true)
            .put("fixtureSha256", fixtureHash).put("modelSha256", sha256(model))
            .put("apkSha256", sha256(File(context.applicationInfo.sourceDir)))
            .put("buildFingerprint", Build.FINGERPRINT).put("supportedAbis", JSONArray(Build.SUPPORTED_ABIS.toList()))
            .put("memoryBeforeOpen", memoryBeforeOpen).put("memoryAfterOpen", memoryAfterOpen)
            .put("memoryAfterClose", memorySnapshot())
            .put("memoryScope", "Whole-process snapshots, not isolated added peak memory or a release gate")
            .put("cases", results).toString() + "\n")
    }
}
