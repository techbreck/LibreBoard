// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.os.Build
import android.os.Debug
import android.text.InputType
import android.view.inputmethod.EditorInfo
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.keyboard.KeyboardElement
import helium314.keyboard.keyboard.KeyboardId
import helium314.keyboard.keyboard.KeyboardLayoutSet
import helium314.keyboard.keyboard.internal.KeyboardBuilder
import helium314.keyboard.keyboard.internal.KeyboardParams
import helium314.keyboard.keyboard.internal.UniqueKeysCache
import helium314.keyboard.latin.DictionaryFacilitatorImpl
import helium314.keyboard.latin.InputAttributes
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.define.DecoderSpecificConstants
import helium314.keyboard.latin.Suggest
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.WordComposer
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.engine.context.BpeContextTokenizer
import helium314.keyboard.latin.engine.context.ContextCandidateRescorer
import helium314.keyboard.latin.engine.onnx.LibreBoardOnnxContracts
import helium314.keyboard.latin.engine.onnx.OnnxContextInferenceSession
import helium314.keyboard.latin.engine.onnx.OnnxRuntimeSessionFactory
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.engine.runtime.LiveTypingEngine
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.settings.SettingsValuesForSuggestion
import helium314.keyboard.latin.utils.SubtypeUtilsAdditional
import helium314.keyboard.latin.utils.prefs
import java.io.File
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Phase 0 tap-corpus measurement harness. Replays prepared corpus rows through the live engine on
 * a bound device/emulator run and emits schema-3 measurement JSONL. The output is evidence only
 * when evaluate_engine.py binds it to a declared environment; it is never a release claim alone.
 *
 * Instrumentation arguments:
 *   phase0CorpusFile   prepared corpus JSONL under filesDir (required)
 *   phase0OutputFile   measurement JSONL written under filesDir (required)
 *   phase0RunId        testRunId bound into every test row (required)
 *   phase0Environment  environmentKind: stock_android_hardware | grapheneos_hardware | low_ram_emulator
 *   phase0ModelDir     optional filesDir directory holding context.onnx + tokenizer.json
 *   phase0Limit        optional row cap for rehearsals
 */
@RunWith(AndroidJUnit4::class)
class Phase0MeasurementInstrumentedTest {

    private class Fixture(
        val context: android.content.Context,
        val facilitator: DictionaryFacilitatorImpl,
        val suggest: Suggest,
        val keyboard: Keyboard,
        val letterBounds: FloatArray,
        val settingsForSuggestion: SettingsValuesForSuggestion,
        val editorInfo: EditorInfo,
        val inputAttributes: InputAttributes,
    )

    private fun buildFixture(context: android.content.Context): Fixture {
        Settings.init(context)
        val editorInfo = EditorInfo().apply { inputType = InputType.TYPE_CLASS_TEXT }
        val inputAttributes = InputAttributes(editorInfo, false, context.packageName)
        Settings.getInstance().loadSettings(context, Locale.US, inputAttributes)

        val facilitator = DictionaryFacilitatorImpl()
        facilitator.resetDictionaries(
            context, Locale.US, false, false, false, true, "", null,
        )
        val deadline = System.nanoTime() + 30_000_000_000L
        while (!facilitator.hasAtLeastOneInitializedMainDictionary()) {
            check(System.nanoTime() < deadline) { "main dictionary did not initialize" }
            Thread.sleep(25)
        }

        val layoutParams = KeyboardLayoutSet.Params()
        for ((name, value) in arrayOf<Pair<String, Any>>(
            "editorInfo" to editorInfo,
            "subtype" to helium314.keyboard.latin.RichInputMethodSubtype.get(
                SubtypeUtilsAdditional.createEmojiCapableAdditionalSubtype(Locale.ENGLISH, "qwerty", true)),
            "keyboardWidth" to 1000,
            "keyboardHeight" to 400,
        )) {
            KeyboardLayoutSet.Params::class.java.getDeclaredField(name).apply { isAccessible = true }.set(layoutParams, value)
        }
        val builder = KeyboardBuilder<KeyboardParams>(context, KeyboardParams(UniqueKeysCache.NO_CACHE))
        builder.load(KeyboardId(KeyboardElement.ALPHABET, layoutParams))
        val keyboard = builder.build()

        var left = Float.MAX_VALUE; var top = Float.MAX_VALUE
        var right = Float.MIN_VALUE; var bottom = Float.MIN_VALUE
        for (key in keyboard.sortedKeys) {
            if (key.code <= 0 || !Character.isLetter(key.code)) continue
            left = minOf(left, key.x.toFloat()); top = minOf(top, key.y.toFloat())
            right = maxOf(right, (key.x + key.width).toFloat()); bottom = maxOf(bottom, (key.y + key.height).toFloat())
        }
        check(left < right && top < bottom) { "keyboard exposes no letter keys" }

        return Fixture(
            context, facilitator, Suggest(facilitator), keyboard,
            floatArrayOf(left, top, right - left, bottom - top),
            SettingsValuesForSuggestion(false, false), editorInfo, inputAttributes,
        )
    }

    private fun installContextModel(context: android.content.Context, directory: File) {
        val model = File(directory, "context.onnx")
        val tokenizer = File(directory, "tokenizer.json")
        check(model.isFile && tokenizer.isFile) { "phase0ModelDir must contain context.onnx and tokenizer.json" }
        val opened = OnnxRuntimeSessionFactory.open(model, LibreBoardOnnxContracts.contextEnDe)
        val session = requireNotNull(opened.session) { "context ONNX session unavailable: ${opened.availability}" }
        LiveTypingEngine.contextRescorer.install(
            ContextCandidateRescorer(BpeContextTokenizer.fromJson(tokenizer.readBytes()),
                OnnxContextInferenceSession(session)),
            session,
        )
    }

    private fun setMeasurementConfiguration(
        fixture: Fixture, personalizedDicts: Boolean, neuralStrength: Float,
    ) {
        fixture.context.prefs().edit()
            .putBoolean(Settings.PREF_KEY_USE_PERSONALIZED_DICTS, personalizedDicts)
            .putFloat(Settings.PREF_NEURAL_STRENGTH, neuralStrength)
            .commit()
        Settings.getInstance().loadSettings(fixture.context, Locale.US, fixture.inputAttributes)
    }

    private fun coordinates(raw: String, points: JSONArray, bounds: FloatArray): IntArray {
        val coords = IntArray(raw.codePointCount(0, raw.length) * 2) { Constants.NOT_A_COORDINATE }
        for (index in 0 until points.length()) {
            val point = points.getJSONObject(index)
            if (index * 2 + 1 >= coords.size) break
            coords[index * 2] = (bounds[0] + point.getDouble("x") * bounds[2]).toInt()
            coords[index * 2 + 1] = (bounds[1] + point.getDouble("y") * bounds[3]).toInt()
        }
        return coords
    }

    private fun ngram(precedingContext: String): NgramContext {
        val words = precedingContext.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
            .takeLast(DecoderSpecificConstants.MAX_PREV_WORD_COUNT_FOR_N_GRAM).toTypedArray()
        if (words.isEmpty()) return NgramContext(NgramContext.WordInfo.BEGINNING_OF_SENTENCE_WORD_INFO)
        // WordInfo[0] is the most recent prev word, like NgramContextUtils builds it.
        return NgramContext(*Array(words.size) { i -> NgramContext.WordInfo(words[words.size - 1 - i]) })
    }

    private fun surfaces(suggestedWords: SuggestedWords, raw: String): JSONArray {
        val seen = LinkedHashSet<String>()
        val out = JSONArray()
        val ordered = buildList {
            add(raw)
            for (index in 0 until suggestedWords.size()) add(suggestedWords.getWord(index))
        }
        for (word in ordered) {
            if (word.isNotBlank() && seen.add(normalizeCandidate(word))) out.put(word)
            if (out.length() >= 32) break
        }
        return out
    }

    private fun measure(fixture: Fixture, row: JSONObject): Pair<JSONArray, Double> {
        val raw = row.getString("raw")
        val composer = WordComposer()
        composer.setComposingWord(
            raw.codePoints().toArray(),
            coordinates(raw, row.optJSONArray("touchPoints") ?: JSONArray(), fixture.letterBounds),
        )
        val start = System.nanoTime()
        val suggested = fixture.suggest.getSuggestedWords(
            composer, ngram(row.optString("precedingContext")), fixture.keyboard,
            fixture.settingsForSuggestion, true, SuggestedWords.INPUT_STYLE_TYPING, 1,
        )
        val elapsed = (System.nanoTime() - start) / 1_000_000.0
        return surfaces(suggested, raw) to elapsed
    }

    private fun seedPersonalFixture(fixture: Fixture, row: JSONObject) {
        PersonalizationRuntime.wipe(fixture.context)
        val words = row.optJSONArray("personalWords") ?: return
        for (index in 0 until words.length()) {
            PersonalizationRuntime.observeCommit(
                fixture.context, fixture.editorInfo, false,
                words.getString(index), "en-US", true, null,
            )
        }
    }

    @Test fun replayTapCorpusThroughMeasuredSystems() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardRequirePhase0Measurement") == "true")
        val runId = requireNotNull(args.getString("phase0RunId")) { "phase0RunId is required" }
        require(runId.length in 1..512)
        val environment = requireNotNull(args.getString("phase0Environment")) { "phase0Environment is required" }
        require(environment in setOf("stock_android_hardware", "grapheneos_hardware", "low_ram_emulator"))
        val corpus = File(
            InstrumentationRegistry.getInstrumentation().targetContext.filesDir,
            requireNotNull(args.getString("phase0CorpusFile")),
        )
        val output = File(corpus.parentFile, requireNotNull(args.getString("phase0OutputFile")))
        require(corpus.isFile && corpus.length() in 1..(512L * 1024 * 1024))
        require(!output.isDirectory)
        val limit = args.getString("phase0Limit")?.toIntOrNull() ?: Int.MAX_VALUE
        require(limit > 0)
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        val fixture = buildFixture(context)
        val baselinePssKiB = Debug.getPss().toDouble()
        var peakPssKiB = baselinePssKiB
        args.getString("phase0ModelDir")?.let { installContextModel(context, File(context.filesDir, it)) }
        peakPssKiB = maxOf(peakPssKiB, Debug.getPss().toDouble())

        val writer = output.bufferedWriter()
        var measured = 0
        corpus.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (measured >= limit) break
                if (line.isBlank()) continue
                val row = JSONObject(line)
                if (row.getString("split") != "test" || row.getString("category") == "swipe") continue
                if (row.getString("raw").isEmpty()) continue

                val predictions = JSONObject()
                val latency = JSONObject()

                var start = System.nanoTime()
                val classic = fixture.facilitator.getSuggestionResults(
                    WordComposer().apply {
                        setComposingWord(
                            row.getString("raw").codePoints().toArray(),
                            coordinates(row.getString("raw"),
                                row.optJSONArray("touchPoints") ?: JSONArray(), fixture.letterBounds),
                        )
                    }.composedDataSnapshot,
                    ngram(row.optString("precedingContext")), fixture.keyboard,
                    fixture.settingsForSuggestion, Suggest.SESSION_ID_TYPING,
                    SuggestedWords.INPUT_STYLE_TYPING,
                )
                latency.put("heliboard", (System.nanoTime() - start) / 1_000_000.0)
                val seen = LinkedHashSet<String>()
                val baseline = JSONArray()
                for (word in listOf(row.getString("raw")) + classic.map { it.mWord }) {
                    if (word.isNotBlank() && seen.add(normalizeCandidate(word))) baseline.put(word)
                    if (baseline.length() >= 32) break
                }
                predictions.put("heliboard", baseline)

                setMeasurementConfiguration(fixture, personalizedDicts = false, neuralStrength = 0f)
                var (fusedSurfaces, fusedMs) = measure(fixture, row)
                predictions.put("fused", fusedSurfaces); latency.put("fused", fusedMs)

                seedPersonalFixture(fixture, row)
                setMeasurementConfiguration(fixture, personalizedDicts = true, neuralStrength = 0f)
                val (personalSurfaces, personalMs) = measure(fixture, row)
                predictions.put("fused_personal", personalSurfaces); latency.put("fused_personal", personalMs)

                setMeasurementConfiguration(fixture, personalizedDicts = true, neuralStrength = 50f)
                val (neuralSurfaces, neuralMs) = measure(fixture, row)
                predictions.put("fused_neural", neuralSurfaces); latency.put("fused_neural", neuralMs)
                peakPssKiB = maxOf(peakPssKiB, Debug.getPss().toDouble())

                writer.write(JSONObject(row.toString()).apply {
                    put("schemaVersion", 3)
                    put("environmentKind", environment)
                    put("testRunId", runId)
                    put("predictions", predictions)
                    put("latencyMs", latency)
                }.toString())
                writer.write("\n")
                measured++
            }
        }
        writer.close()
        assertTrue("corpus produced no measurable test rows", measured > 0)
        writeEnvironmentSidecar(context, File(output.parentFile, "environment.json"),
            runId, environment, peakPssKiB - baselinePssKiB)
    }

    private fun writeEnvironmentSidecar(
        context: android.content.Context, file: File, runId: String, environment: String,
        peakAddedNeuralKiB: Double,
    ) {
        val activityManager = context.getSystemService(android.app.ActivityManager::class.java)
        val memoryInfo = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val emulator = Build.FINGERPRINT.contains("generic")
            || Build.HARDWARE in setOf("goldfish", "ranchu")
            || Build.MODEL.contains("Emulator")
            || Build.PRODUCT.contains("sdk")
        val sidecar = JSONObject()
            .put("kind", environment)
            .put("deviceModel", Build.MODEL)
            .put("buildFingerprint", Build.FINGERPRINT)
            .put("testRunId", runId)
            .put("apiLevel", Build.VERSION.SDK_INT)
            .put("physicalDevice", !emulator)
            .put("isLowRamDevice", activityManager.isLowRamDevice)
            .put("memoryMiB", (memoryInfo.totalMem / 1048576).toInt())
            .put("peakAddedNeuralMemoryMiB", peakAddedNeuralKiB / 1024.0)
        if (environment == "grapheneos_hardware") {
            sidecar.put("grapheneOsBuildNumber", Build.DISPLAY)
            sidecar.put("sandboxedGooglePlayInstalled", runCatching {
                context.packageManager.getPackageInfo("app.grapheneos.gmscompat", 0); true
            }.getOrDefault(false))
        }
        file.writeText(sidecar.toString())
    }
}
