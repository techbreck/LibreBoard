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
import helium314.keyboard.keyboard.internal.keyboard_parser.LayoutParser
import helium314.keyboard.latin.DictionaryFacilitatorImpl
import helium314.keyboard.latin.InputAttributes
import helium314.keyboard.latin.NgramContext
import helium314.keyboard.latin.define.DecoderSpecificConstants
import helium314.keyboard.latin.Suggest
import helium314.keyboard.latin.SuggestedWords
import helium314.keyboard.latin.SuggestedWords.SuggestedWordInfo
import helium314.keyboard.latin.WordComposer
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.engine.context.BpeContextTokenizer
import helium314.keyboard.latin.engine.context.ContextCandidateRescorer
import helium314.keyboard.latin.engine.ctc.CtcSwipeDecoder
import helium314.keyboard.latin.engine.geometric.GeometricSwipeDecoder
import helium314.keyboard.latin.engine.integration.HeliBoardGeometricFallback
import helium314.keyboard.latin.engine.integration.HeliBoardSwipeLexicon
import helium314.keyboard.latin.engine.integration.LegacySuggestionFusion
import helium314.keyboard.latin.engine.onnx.LibreBoardOnnxContracts
import helium314.keyboard.latin.engine.onnx.OnnxContextInferenceSession
import helium314.keyboard.latin.engine.onnx.OnnxCtcInferenceSession
import helium314.keyboard.latin.engine.onnx.OnnxRuntimeSessionFactory
import helium314.keyboard.latin.engine.personal.PersonalizationRuntime
import helium314.keyboard.latin.engine.runtime.LiveTypingEngine
import helium314.keyboard.latin.settings.Settings
import helium314.keyboard.latin.utils.DeviceProtectedUtils
import helium314.keyboard.latin.utils.LayoutType
import helium314.keyboard.latin.utils.LayoutUtilsCustom
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
 *   phase0SwipeCorpusFile  prepared swipe JSONL under filesDir (gates the swipe test)
 *   phase0SwipeOutputFile  swipe measurement JSONL filename beside the swipe corpus
 *   phase0SwipeModelFile   optional filesDir CTC swipe .onnx
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

    private fun buildKeyboard(
        context: android.content.Context, editorInfo: EditorInfo, layoutName: String,
    ): Keyboard {
        val layoutParams = KeyboardLayoutSet.Params()
        for ((name, value) in arrayOf<Pair<String, Any>>(
            "editorInfo" to editorInfo,
            "subtype" to helium314.keyboard.latin.RichInputMethodSubtype.get(
                SubtypeUtilsAdditional.createEmojiCapableAdditionalSubtype(Locale.ENGLISH, layoutName, true)),
            "keyboardWidth" to 1000,
            "keyboardHeight" to 400,
        )) {
            KeyboardLayoutSet.Params::class.java.getDeclaredField(name).apply { isAccessible = true }.set(layoutParams, value)
        }
        val builder = KeyboardBuilder<KeyboardParams>(context, KeyboardParams(UniqueKeysCache.NO_CACHE))
        builder.load(KeyboardId(KeyboardElement.ALPHABET, layoutParams))
        return builder.build()
    }

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
        // resetDictionaries bumps swipeLexiconRevision before the index rebuild launches, so
        // revision alone cannot signal readiness. Poll the index itself: the geometric and CTC
        // decoders return empty candidates until the unigram scan publishes.
        val lexiconDeadline = System.nanoTime() + 120_000_000_000L
        while (facilitator.getSwipeLexiconWords(listOf("en-US"), 0, 1, false).isEmpty()) {
            check(System.nanoTime() < lexiconDeadline) { "swipe lexicon did not populate" }
            Thread.sleep(50)
        }

        val keyboard = buildKeyboard(context, editorInfo, "qwerty")
        val letterBounds = letterBounds(keyboard)

        return Fixture(
            context, facilitator, Suggest(facilitator), keyboard, letterBounds,
            SettingsValuesForSuggestion(false, false), editorInfo, inputAttributes,
        )
    }

    private fun letterBounds(keyboard: Keyboard): FloatArray {
        var left = Float.MAX_VALUE; var top = Float.MAX_VALUE
        var right = Float.MIN_VALUE; var bottom = Float.MIN_VALUE
        for (key in keyboard.sortedKeys) {
            if (key.code <= 0 || !Character.isLetter(key.code)) continue
            left = minOf(left, key.x.toFloat()); top = minOf(top, key.y.toFloat())
            right = maxOf(right, (key.x + key.width).toFloat()); bottom = maxOf(bottom, (key.y + key.height).toFloat())
        }
        check(left < right && top < bottom) { "keyboard exposes no letter keys" }
        return floatArrayOf(left, top, right - left, bottom - top)
    }

    /**
     * The swipe corpus normalizes path coordinates over the canonical training layout: three
     * uniform rows of 0.1-width letter keys with the second and third rows indented by the
     * functional keys. Reproduce that geometry as a custom layout so traced paths land on the
     * same letters the model trained on; the production qwerty layout fills each row instead.
     */
    private fun buildCanonicalSwipeKeyboard(
        context: android.content.Context, editorInfo: EditorInfo,
    ): Pair<Keyboard, FloatArray> {
        val rows = listOf("qwertyuiop", "asdfghjkl", "zxcvbnm").joinToString(",") { letters ->
            letters.toCharArray().joinToString(",", "[", "]") { "{\"label\":\"$it\",\"width\":0.1}" }
        }
        val layoutName = LayoutUtilsCustom.getLayoutName(
            "phase0-canonical-qwerty", LayoutType.MAIN, Locale.US)
        val directory = File(DeviceProtectedUtils.getFilesDir(context), "layouts/main")
        check(directory.isDirectory || directory.mkdirs()) { "cannot create custom layout dir" }
        File(directory, layoutName).writeText("[$rows]")
        LayoutUtilsCustom.onLayoutFileChanged()
        LayoutParser.clearCache()
        val keyboard = buildKeyboard(context, editorInfo, layoutName)
        return keyboard to letterBounds(keyboard)
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
        val (shardIndex, shardCount) = shardConfig(args)
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        val fixture = buildFixture(context)
        val baselinePssKiB = Debug.getPss().toDouble()
        var peakPssKiB = baselinePssKiB
        args.getString("phase0ModelDir")?.let { installContextModel(context, File(context.filesDir, it)) }
        peakPssKiB = maxOf(peakPssKiB, Debug.getPss().toDouble())

        val writer = output.bufferedWriter()
        var measured = 0
        var ordinal = 0
        corpus.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (measured >= limit) break
                if (line.isBlank()) continue
                val row = JSONObject(line)
                if (row.getString("split") != "test" || row.getString("category") == "swipe") continue
                if (row.getString("raw").isEmpty()) continue
                if (ordinal++ % shardCount != shardIndex) continue

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

    private fun swipeRequest(row: JSONObject, bounds: FloatArray,
                             geometry: KeyGeometry, sequenceId: Long): TypingRequest {
        val flat = row.getJSONArray("pathCoordinates")
        val path = (0 until flat.length() / 2).map { index ->
            TouchPoint(
                bounds[0] + flat.getDouble(index * 2).toFloat() * bounds[2],
                bounds[1] + flat.getDouble(index * 2 + 1).toFloat() * bounds[3],
                index * 10L,
            )
        }
        return TypingRequest.bounded(
            rawText = "",
            path = path,
            precedingContext = row.optString("precedingContext"),
            geometry = geometry,
            enabledLanguages = listOf("en-US"),
            wordLock = WordLock.Unlocked,
            fieldPolicy = Settings.getValues().mInputAttributes.mFieldPolicy,
            inputStyle = InputStyle.SWIPE,
            sequenceId = sequenceId,
        )
    }

    private fun swipeSurfaces(result: SwipeDecodeResult): JSONArray {
        val seen = LinkedHashSet<String>()
        val out = JSONArray()
        for (candidate in result.candidates) {
            if (candidate.surface.isNotBlank() && seen.add(normalizeCandidate(candidate.surface))) {
                out.put(candidate.surface)
            }
            if (out.length() >= 32) break
        }
        // Schema-3 requires a non-empty prediction list; a sentinel keeps an empty decode an
        // honest miss instead of silently dropping the row.
        if (out.length() == 0) out.put(EMPTY_PREDICTION_SENTINEL)
        return out
    }

    /** Disjoint row partition for multi-environment runs; duplicate ids fail dataset checks. */
    private fun shardConfig(args: android.os.Bundle): Pair<Int, Int> {
        val count = args.getString("phase0ShardCount")?.toIntOrNull() ?: 1
        val index = args.getString("phase0ShardIndex")?.toIntOrNull() ?: 0
        require(count > 0 && index in 0 until count) { "invalid shard config $index/$count" }
        return index to count
    }

    private fun suggestionInfoSurfaces(suggestions: List<SuggestedWordInfo>): JSONArray {
        val seen = LinkedHashSet<String>()
        val out = JSONArray()
        for (info in suggestions) {
            if (info.mWord.isNotBlank() && seen.add(normalizeCandidate(info.mWord))) {
                out.put(info.mWord)
            }
            if (out.length() >= 32) break
        }
        if (out.length() == 0) out.put(EMPTY_PREDICTION_SENTINEL)
        return out
    }

    @Test fun replaySwipeCorpusThroughMeasuredSystems() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardRequirePhase0SwipeMeasurement") == "true")
        val runId = requireNotNull(args.getString("phase0RunId")) { "phase0RunId is required" }
        require(runId.length in 1..512)
        val environment = requireNotNull(args.getString("phase0Environment")) { "phase0Environment is required" }
        require(environment in setOf("stock_android_hardware", "grapheneos_hardware", "low_ram_emulator"))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val corpus = File(context.filesDir, requireNotNull(args.getString("phase0SwipeCorpusFile")))
        val output = File(corpus.parentFile, requireNotNull(args.getString("phase0SwipeOutputFile")))
        require(corpus.isFile && corpus.length() in 1..(512L * 1024 * 1024))
        require(!output.isDirectory)
        val limit = args.getString("phase0Limit")?.toIntOrNull() ?: Int.MAX_VALUE
        require(limit > 0)
        val (shardIndex, shardCount) = shardConfig(args)

        val fixture = buildFixture(context)
        val baselinePssKiB = Debug.getPss().toDouble()
        var peakPssKiB = baselinePssKiB
        val (swipeKeyboard, swipeBounds) = buildCanonicalSwipeKeyboard(context, fixture.editorInfo)
        val geometry = HeliBoardGeometricFallback.toEngineGeometry(swipeKeyboard)
        val lexicon = HeliBoardSwipeLexicon(fixture.facilitator)
        android.util.Log.i("Phase0Swipe", "lexicon revision=${fixture.facilitator.getSwipeLexiconRevision()} " +
            "words(en-US,5)=${lexicon.words(listOf("en-US"), 5).count()} " +
            "words(en-US,0)=${lexicon.words(listOf("en-US"), 0).count()} " +
            "geometryKeys=${geometry.keys.size} geometry=${geometry.width}x${geometry.height}")
        val geometricDecoder = GeometricSwipeDecoder(lexicon)
        val ctcDecoder = args.getString("phase0SwipeModelFile")?.let { name ->
            val opened = OnnxRuntimeSessionFactory.open(
                File(context.filesDir, name), LibreBoardOnnxContracts.swipeCtc)
            val runtime = requireNotNull(opened.session) {
                "swipe ONNX session unavailable: ${opened.availability}" }
            val session = OnnxCtcInferenceSession(runtime)
            CtcSwipeDecoder(session, lexicon).also { decoder ->
                LiveTypingEngine.swipeDecoder.install(decoder, session)
            }
        }
        args.getString("phase0ModelDir")?.let { installContextModel(context, File(context.filesDir, it)) }
        peakPssKiB = maxOf(peakPssKiB, Debug.getPss().toDouble())

        val writer = output.bufferedWriter()
        val swipeExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
        val fusion = LegacySuggestionFusion()
        var measured = 0
        var ordinal = 0
        corpus.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (measured >= limit) break
                if (line.isBlank()) continue
                val row = JSONObject(line)
                if (row.optString("split") != "test") continue
                val path = row.optJSONArray("pathCoordinates") ?: continue
                if (path.length() < 4) continue
                if (ordinal++ % shardCount != shardIndex) continue

                val request = swipeRequest(row, swipeBounds, geometry, measured.toLong() + 1)
                val pointers = InputPointers(request.path.size)
                request.path.forEachIndexed { index, point ->
                    pointers.addPointerAt(index, point.x.toInt(), point.y.toInt(), 0, point.timeMillis.toInt())
                }
                val ngramContext = ngram(row.optString("precedingContext"))
                val predictions = JSONObject()
                val latency = JSONObject()
                val availability = JSONObject()

                // Decode geometric and CTC concurrently under a standalone budget wider than the
                // production 125 ms proposal deadline so each decoder's candidate set is visible.
                val geometricFuture = swipeExecutor.submit<Pair<SwipeDecodeResult, Double>> {
                    val decodeStart = System.nanoTime()
                    geometricDecoder.decode(request, Deadline.afterMillis(SWIPE_STANDALONE_BUDGET_MS)) to
                        (System.nanoTime() - decodeStart) / 1_000_000.0
                }
                var start = System.nanoTime()
                val ctc = ctcDecoder?.decode(request, Deadline.afterMillis(SWIPE_STANDALONE_BUDGET_MS))
                latency.put("ctc", (System.nanoTime() - start) / 1_000_000.0)
                val (geometric, geometricMs) = geometricFuture.get()
                latency.put("geometric", geometricMs)
                predictions.put("geometric", swipeSurfaces(geometric))
                predictions.put("ctc", ctc?.let { swipeSurfaces(it) } ?: JSONArray())
                availability.put("geometric", geometric.availability.name)
                availability.put("ctc", ctc?.availability?.name ?: "UNAVAILABLE")

                // fused_swipe replays through the production batch-input suggestion path under
                // the real proposal budget; on slow runtimes decoders may not land in time.
                val composer = WordComposer()
                composer.setBatchInputPointers(pointers)
                start = System.nanoTime()
                val fused = fixture.suggest.getSuggestedWords(
                    composer, ngramContext,
                    swipeKeyboard, fixture.settingsForSuggestion, true,
                    SuggestedWords.INPUT_STYLE_UPDATE_BATCH, measured + 1,
                )
                latency.put("fused_swipe", (System.nanoTime() - start) / 1_000_000.0)
                predictions.put("fused_swipe", surfaces(fused, "").also {
                    if (it.length() == 0) it.put(EMPTY_PREDICTION_SENTINEL)
                })

                // fused_relaxed composes the same fusion boundary the production batch path uses,
                // but feeds it the standalone decoded candidates and the geometric-trace fallback
                // classics without the proposal deadline. Batch native results are empty under
                // the gesture-policy guard, so the fallback is the entire classic side.
                val fallbackData = HeliBoardGeometricFallback.toTypingComposedData(pointers, swipeKeyboard)
                val fallback = if (fallbackData != null) fixture.facilitator.getSuggestionResults(
                    fallbackData, ngramContext, swipeKeyboard, fixture.settingsForSuggestion,
                    Suggest.SESSION_ID_GESTURE, SuggestedWords.INPUT_STYLE_UPDATE_BATCH,
                ).toList() else emptyList()
                val decodedCandidates = geometric.candidates + (ctc?.candidates ?: emptyList())
                start = System.nanoTime()
                val relaxed = fusion.fuse(
                    rawText = "", classicSuggestions = fallback.toList(),
                    supplementalCandidates = decodedCandidates,
                    enabledLanguageTags = listOf("en-US"), defaultLocale = Locale.US,
                    inputStyle = InputStyle.SWIPE, typingRequest = request,
                    neuralStrength = 0, neuralDeadline = Deadline.afterMillis(2000),
                )
                val auxiliaryLatency = JSONObject()
                val auxiliaryPredictions = JSONObject()
                auxiliaryLatency.put("fused_relaxed", (System.nanoTime() - start) / 1_000_000.0)
                auxiliaryPredictions.put("fused_relaxed", suggestionInfoSurfaces(relaxed.suggestions))

                start = System.nanoTime()
                val relaxedNeural = fusion.fuse(
                    rawText = "", classicSuggestions = fallback.toList(),
                    supplementalCandidates = decodedCandidates,
                    enabledLanguageTags = listOf("en-US"), defaultLocale = Locale.US,
                    inputStyle = InputStyle.SWIPE, typingRequest = request,
                    neuralStrength = 50, neuralDeadline = Deadline.afterMillis(2000),
                )
                auxiliaryLatency.put("fused_relaxed_neural", (System.nanoTime() - start) / 1_000_000.0)
                auxiliaryPredictions.put("fused_relaxed_neural", suggestionInfoSurfaces(relaxedNeural.suggestions))
                peakPssKiB = maxOf(peakPssKiB, Debug.getPss().toDouble())

                writer.write(JSONObject(row.toString()).apply {
                    put("schemaVersion", 3)
                    put("category", "swipe")
                    put("raw", "")
                    put("environmentKind", environment)
                    put("testRunId", runId)
                    put("predictions", predictions)
                    put("latencyMs", latency)
                    put("engineAvailability", availability)
                    put("auxiliaryPredictions", auxiliaryPredictions)
                    put("auxiliaryLatencyMs", auxiliaryLatency)
                }.toString())
                writer.write("\n")
                measured++
            }
        }
        swipeExecutor.shutdown()
        writer.close()
        assertTrue("swipe corpus produced no measurable test rows", measured > 0)
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

    private companion object {
        /** Standalone decoder budget; production gives decoders 125 ms inside fused_swipe. */
        const val SWIPE_STANDALONE_BUDGET_MS = 1500L

        /** Placeholder emitted when a system returns no candidates; always a miss. */
        const val EMPTY_PREDICTION_SENTINEL = "__no_candidates__"
    }
}
