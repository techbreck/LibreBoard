// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.os.Build
import android.os.SystemClock
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.latin.BuildConfig
import helium314.keyboard.latin.dictionary.Dictionary
import helium314.keyboard.latin.dictionary.ReadOnlyBinaryDictionary
import helium314.keyboard.latin.engine.ctc.CtcSwipeDecoder
import helium314.keyboard.latin.engine.ctc.CtcInferenceSession
import helium314.keyboard.latin.engine.geometric.LexiconWord
import helium314.keyboard.latin.engine.geometric.StaticSwipeLexiconCollector
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import helium314.keyboard.latin.engine.geometric.SwipeLexiconIndex
import helium314.keyboard.latin.engine.onnx.LibreBoardOnnxContracts
import helium314.keyboard.latin.engine.onnx.OnnxCtcInferenceSession
import helium314.keyboard.latin.engine.onnx.OnnxRuntimeSessionFactory
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Bounded held-out decoder diagnostic, not full fusion, editor latency or release evidence. */
@RunWith(AndroidJUnit4::class)
class SwipeRuntimeInstrumentedTest {
    @Test fun replayValidationPathsThroughAndroidCtcAndBundledVocabulary() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardRequireSwipeRuntime") == "true")
        assertTrue(BuildConfig.LIBREBOARD_ONNX_RUNTIME_PACKAGED)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "swipe-runtime-diagnostic")
        val fixtureFile = File(directory, "fixture.json")
        val model = File(directory, "swipe.onnx")
        assertTrue(fixtureFile.length() in 1..1_048_576)
        assertTrue(model.length() in 1..(3L * 1024 * 1024))
        fun sha256(file: File): String {
            val hash = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { stream ->
                val buffer = ByteArray(65536)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    hash.update(buffer, 0, count)
                }
            }
            return hash.digest().joinToString("") { "%02x".format(it) }
        }
        val fixtureHash = sha256(fixtureFile)
        assertEquals(args.getString("swipeFixtureSha256"), fixtureHash)
        val fixture = JSONObject(fixtureFile.readText())
        assertEquals(1, fixture.getInt("schemaVersion"))
        assertFalse(fixture.getBoolean("releaseEligible"))
        assertEquals("validation", fixture.getString("split"))
        assertEquals("android-ctc-native-dictionary-diagnostic-v1", fixture.getString("protocol"))
        assertEquals(fixture.getString("modelSha256"), sha256(model))
        val layout = fixture.getJSONObject("layout")
        val labels = layout.getJSONArray("keyLabels")
        val centers = layout.getJSONArray("keyCenters")
        val mask = layout.getJSONArray("keyMask")
        assertEquals(64, labels.length())
        assertEquals(128, centers.length())
        assertEquals(64, mask.length())
        val geometry = KeyGeometry(1000f, 1000f, (0 until 64).filter { mask.getInt(it) == 1 }.map { index ->
            KeySlot(index, labels.getString(index), centers.getDouble(index * 2).toFloat() * 1000,
                centers.getDouble(index * 2 + 1).toFloat() * 1000, 100f, 1000f / 3)
        })
        val dictionaryFile = File.createTempFile("swipe-native-", ".dict", context.cacheDir)
        try {
            context.assets.open("dicts/main_en-US.dict").use { input -> dictionaryFile.outputStream().use { input.copyTo(it) } }
            val dictionaryHash = sha256(dictionaryFile)
            val native = ReadOnlyBinaryDictionary(dictionaryFile.absolutePath, 0, dictionaryFile.length(), false, Locale.US, Dictionary.TYPE_MAIN)
            val collector = StaticSwipeLexiconCollector(100_000)
            try {
                assertTrue(native.isValidDictionary)
                native.visitUnigrams(1_000_000) { word, frequency, notWord, offensive ->
                    if (!notWord) collector.add(LexiconWord(word, "en-US", frequency, possiblyOffensive = offensive))
                }
            } finally { native.close() }
            val index = SwipeLexiconIndex.from(collector.words())
            assertEquals(100_000, index.size)
            var lookupNanos = 0L
            val lexicon = SwipeLexicon { languages, length ->
                val start = SystemClock.elapsedRealtimeNanos()
                val words = index.words(languages, length, 100_000, true)
                lookupNanos += SystemClock.elapsedRealtimeNanos() - start
                words.asSequence()
            }
            val opened = OnnxRuntimeSessionFactory.open(model, LibreBoardOnnxContracts.swipeCtc)
            assertEquals(EngineAvailability.AVAILABLE, opened.availability)
            val maximumRows = args.getString("swipeMaximumRows")?.toInt() ?: 100
            assertTrue(maximumRows in 1..100)
            val results = JSONArray()
            OnnxCtcInferenceSession(requireNotNull(opened.session)).use { session ->
                var inferenceNanos = 0L
                val timedInference = CtcInferenceSession { features, deadline ->
                    val start = SystemClock.elapsedRealtimeNanos()
                    session.infer(features, deadline).also {
                        inferenceNanos += SystemClock.elapsedRealtimeNanos() - start
                    }
                }
                val decoder = CtcSwipeDecoder(timedInference, lexicon)
                val cases = fixture.getJSONArray("cases")
                assertEquals(100, cases.length())
                for (row in 0 until maximumRows) {
                    val case = cases.getJSONObject(row)
                    val coordinates = case.getJSONArray("path")
                    assertEquals(128, coordinates.length())
                    val path = (0 until 64).map { point -> TouchPoint(
                        coordinates.getDouble(point * 2).toFloat() * 1000,
                        coordinates.getDouble(point * 2 + 1).toFloat() * 1000,
                    ) }
                    val request = TypingRequest("", path, "", geometry, listOf("en-US"), WordLock.Manual("en-US"),
                        FieldPolicy.NORMAL, FieldClass.PLAIN, InputStyle.SWIPE, row.toLong())
                    lookupNanos = 0
                    inferenceNanos = 0
                    val started = SystemClock.elapsedRealtimeNanos()
                    // Deliberately generous diagnostic deadline measures cost rather than hiding
                    // slow paths behind a production timeout. It must not be called a budget pass.
                    val output = decoder.decode(request, Deadline.afterMillis(60_000))
                    val elapsed = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
                    assertEquals("decoder availability for row $row", EngineAvailability.AVAILABLE, output.availability)
                    results.put(JSONObject().put("id", case.getString("id")).put("target", case.getString("target"))
                        .put("strata", case.getJSONArray("strata")).put("decoderMs", elapsed)
                        .put("inferenceMs", inferenceNanos / 1_000_000.0).put("vocabularyLookupMs", lookupNanos / 1_000_000.0)
                        .put("predictions", JSONArray(output.candidates.map { it.surface })))
                    if ((row + 1) % 10 == 0) android.util.Log.i("SwipeRuntimeDiagnostic", "completed ${row + 1}/100 paths")
                }
            }
            File(directory, "android-report.json").writeText(JSONObject()
                .put("schemaVersion", 1).put("releaseEligible", false).put("decoderOnly", true)
                .put("fixtureSha256", fixtureHash).put("modelSha256", sha256(model))
                .put("apkSha256", sha256(File(context.applicationInfo.sourceDir)))
                .put("dictionarySha256", dictionaryHash).put("buildFingerprint", Build.FINGERPRINT)
                .put("maximumRows", maximumRows)
                .put("diagnosticDeadlineMs", 60_000).put("rows", results).toString() + "\n")
        } finally { dictionaryFile.delete() }
    }
}
