// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.os.Build
import android.os.Debug
import android.text.InputType
import android.util.Log
import android.view.inputmethod.EditorInfo
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.event.Event
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
import helium314.keyboard.latin.CapsMode
import helium314.keyboard.latin.common.Constants
import helium314.keyboard.latin.common.CoordinateUtils
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
 * a bound device/emulator run and emits schema-4 measurement JSONL. The output is evidence only
 * when evaluate_engine.py binds it to a declared environment; it is never a release claim alone.
 *
 * Each tap row carries both the suggestion slate in production's ranking and, per system, the
 * `commits` decision that production would apply when the word is terminated. Scoring the slate's
 * rank one instead is what made the schema-3 tap results unusable: the strip always seats the typed
 * word first, so no correction could ever be observed.
 *
 * Instrumentation arguments:
 *   phase0CorpusFile   prepared corpus JSONL under filesDir, or an absolute path (required)
 *   phase0OutputFile   measurement JSONL written beside the corpus, or an absolute path (required)
 *   phase0RunId        testRunId bound into every test row (required)
 *   phase0Environment  environmentKind: stock_android_hardware | grapheneos_hardware | low_ram_emulator
 *   phase0ModelDir     optional filesDir-relative or absolute directory holding context.onnx + tokenizer.json
 *   phase0Limit        optional row cap for rehearsals
 *   phase0SwipeCorpusFile  prepared swipe JSONL under filesDir, or an absolute path (gates the swipe test)
 *   phase0SwipeOutputFile  swipe measurement JSONL filename beside the swipe corpus, or an absolute path
 *   phase0SwipeModelFile   optional filesDir-relative or absolute CTC swipe .onnx
 *
 * Absolute paths exist so a non-debuggable testOnly APK can be measured without `run-as`. The
 * driver then stages corpora under /data/local/tmp and pulls outputs from the same directory.
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
        /** One-time synchronous cost before any row can be measured; excluded from latencyMs. */
        val dictionaryReadyMs: Long,
        val swipeLexiconReadyMs: Long,
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
        // A composing text field that requests auto-correction, which is what an ordinary message
        // or note field looks like. Bare TYPE_CLASS_TEXT requests neither auto-correction nor
        // multi-line, so SettingsValues sets mAutoCorrectEnabled false and the auto-correction
        // threshold to Float.MAX_VALUE; only whitelist entries such as "cant" -> "can't" then get
        // through, and every ordinary spatial correction is silently blocked.
        val editorInfo = EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or
                InputType.TYPE_TEXT_VARIATION_NORMAL or
                InputType.TYPE_TEXT_FLAG_AUTO_CORRECT
        }
        val inputAttributes = InputAttributes(editorInfo, false, context.packageName)
        Settings.getInstance().loadSettings(context, Locale.US, inputAttributes)
        // Fail loudly rather than reporting a keyboard that structurally cannot correct.
        check(Settings.getValues().mAutoCorrectEnabled) {
            "measurement fixture has auto-correction disabled; every correction would be blocked"
        }

        val facilitator = DictionaryFacilitatorImpl()
        facilitator.resetDictionaries(
            context, Locale.US, false, false, false, true, "", null,
        )
        val dictionaryStart = System.nanoTime()
        val deadline = dictionaryStart + 30_000_000_000L
        while (!facilitator.hasAtLeastOneInitializedMainDictionary()) {
            check(System.nanoTime() < deadline) { "main dictionary did not initialize" }
            Thread.sleep(25)
        }
        val dictionaryReadyMs = (System.nanoTime() - dictionaryStart) / 1_000_000
        // resetDictionaries bumps swipeLexiconRevision before the index rebuild launches, so
        // revision alone cannot signal readiness. Poll the index itself: the geometric and CTC
        // decoders return empty candidates until the unigram scan publishes. The per-word JNI
        // callback scan is one-time cost that can exceed minutes on hardened builds.
        val lexiconStart = System.nanoTime()
        val lexiconDeadline = lexiconStart + 600_000_000_000L
        while (facilitator.getSwipeLexiconWords(listOf("en-US"), 0, 1, false).isEmpty()) {
            check(System.nanoTime() < lexiconDeadline) { "swipe lexicon did not populate" }
            Thread.sleep(50)
        }
        val swipeLexiconReadyMs = (System.nanoTime() - lexiconStart) / 1_000_000
        Log.i(TAG, "swipe lexicon populated in ${swipeLexiconReadyMs}ms")

        val keyboard = buildKeyboard(context, editorInfo, "qwerty")
        val letterBounds = letterBounds(keyboard)

        return Fixture(
            context, facilitator, Suggest(facilitator), keyboard, letterBounds,
            SettingsValuesForSuggestion(false, false), editorInfo, inputAttributes,
            dictionaryReadyMs, swipeLexiconReadyMs,
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
        // LatinIME.loadSettings does this on every settings change; without it Suggest keeps the
        // 0f constructor default instead of the configured auto-correction threshold.
        fixture.suggest.setAutoCorrectionThreshold(Settings.getValues().mAutoCorrectionThreshold)
        check(Settings.getValues().mAutoCorrectEnabled) {
            "auto-correction became disabled mid-run; measured corrections would be blocked"
        }
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

    /**
     * Bounded whole-process PSS sampling around the neural work.
     *
     * The reported figure is an upper bound on added neural memory, not an isolated neural peak:
     * after fixture setup the process still performs classic dictionary suggestion work on every
     * row, and sampling between rows can miss a transient peak entirely. Recording the components
     * separately lets a reader see how much of the headline number the neural path accounts for,
     * instead of reading one opaque total as a memory result.
     */
    private class MemoryProbe(val fixtureKiB: Double) {
        var afterModelOpenKiB = fixtureKiB; private set
        var peakNeuralDisabledKiB = fixtureKiB; private set
        var peakNeuralEnabledKiB = fixtureKiB; private set
        var samples = 0; private set

        private fun sample(): Double {
            samples++
            return Debug.getPss().toDouble()
        }

        fun onModelOpened() {
            afterModelOpenKiB = maxOf(afterModelOpenKiB, sample())
            peakNeuralEnabledKiB = maxOf(peakNeuralEnabledKiB, afterModelOpenKiB)
        }

        /** Sampled after the systems that run with the neural path switched off. */
        fun onNeuralDisabled() {
            peakNeuralDisabledKiB = maxOf(peakNeuralDisabledKiB, sample())
        }

        /** Sampled after the systems that run with the neural path switched on. */
        fun onNeuralEnabled() {
            peakNeuralEnabledKiB = maxOf(peakNeuralEnabledKiB, sample())
        }

        val peakKiB get() = maxOf(peakNeuralDisabledKiB, peakNeuralEnabledKiB)
        val addedKiB get() = peakKiB - fixtureKiB

        fun toJson(): JSONObject = JSONObject()
            .put("fixtureKiB", fixtureKiB)
            .put("afterModelOpenKiB", afterModelOpenKiB)
            .put("peakNeuralDisabledKiB", peakNeuralDisabledKiB)
            .put("peakNeuralEnabledKiB", peakNeuralEnabledKiB)
            .put("samples", samples)
            .put("scope", "Whole-process PSS sampled after fixture setup. " +
                "peakAddedNeuralMemoryMiB is peak minus fixture: an upper bound that also carries " +
                "the classic suggestion work every row performs, and a sparse sample that may miss " +
                "transient peaks. It is not an isolated neural peak and a value under the gate " +
                "does not by itself establish the memory gate.")
    }

    /** One measured system: the strip as production ranks it plus the word it would commit. */
    private class Measured(
        val surfaces: JSONArray,
        val commit: JSONObject,
        val elapsedMs: Double,
        val gateTrace: String,
    )

    /**
     * The suggestion slate in production's own order. Schema 4 deliberately does not hoist the raw
     * surface: production already seats the typed word at [SuggestedWords.INDEX_OF_TYPED_WORD], and
     * the schema-3 harness prepended it a second time, which made rank one the typed text for every
     * row and hid every correction from the evaluator.
     */
    private fun surfaces(suggestedWords: SuggestedWords): JSONArray {
        val seen = LinkedHashSet<String>()
        val out = JSONArray()
        for (index in 0 until suggestedWords.size()) {
            val word = suggestedWords.getWord(index)
            if (word.isNotBlank() && seen.add(normalizeCandidate(word))) out.put(word)
            if (out.length() >= 32) break
        }
        return out
    }

    private fun commitJson(decision: Suggest.CommitDecision): JSONObject = JSONObject()
        .put("committed", decision.committedWord)
        .put("willAutoCorrect", decision.willAutoCorrect)

    /**
     * A freshly typed word, built the way production builds one.
     *
     * [WordComposer.setComposingWord] ends by marking the composition resumed, and production never
     * auto-corrects a resumed word — that is the "user tapped back into an existing word" case. A
     * harness that composed rows that way could not observe a single correction on any system, which
     * is exactly what the first schema-4 hardware run showed: 0% auto-correction even on the classic
     * baseline. Replaying the key events instead leaves the composition fresh, as typing does.
     */
    private fun composerFor(fixture: Fixture, row: JSONObject): WordComposer {
        val raw = row.getString("raw")
        val coords = coordinates(
            raw, row.optJSONArray("touchPoints") ?: JSONArray(), fixture.letterBounds)
        val composer = WordComposer()
        raw.codePoints().toArray().forEachIndexed { index, codePoint ->
            val event = Event.createEventForCodePointFromAlreadyTypedText(
                codePoint,
                CoordinateUtils.xFromArray(coords, index),
                CoordinateUtils.yFromArray(coords, index),
            )
            composer.applyProcessedEvent(composer.processEvent(event))
        }
        composer.adviseCapitalizedModeBeforeFetchingSuggestions(CapsMode.OFF)
        return composer
    }

    private fun measure(fixture: Fixture, row: JSONObject): Measured {
        val raw = row.getString("raw")
        val composer = composerFor(fixture, row)
        val trace = StringBuilder()
        fixture.suggest.autoCorrectionTrace = trace
        val start = System.nanoTime()
        val suggested = fixture.suggest.getSuggestedWords(
            composer, ngram(row.optString("precedingContext")), fixture.keyboard,
            fixture.settingsForSuggestion, true, SuggestedWords.INPUT_STYLE_TYPING, 1,
        )
        val elapsed = (System.nanoTime() - start) / 1_000_000.0
        fixture.suggest.autoCorrectionTrace = null
        val slate = surfaces(suggested)
        val decision = Suggest.commitDecisionOf(suggested, raw)
        // Schema 4 still requires the exact raw surface in every LibreBoard slate so the report can
        // prove the one-tap fallback; production seats it at index 0, so a miss is a real defect.
        check((0 until slate.length()).any { slate.getString(it) == raw }) {
            "measured slate dropped the raw surface for '$raw'"
        }
        // Normalized, exactly as the evaluator checks it. A capitalization correction such as
        // "sox" -> "Sox" commits a surface the slate holds only in its typed-word casing, because
        // the slate must stay normalization-distinct.
        check((0 until slate.length()).any {
            normalizeCandidate(slate.getString(it)) == normalizeCandidate(decision.committedWord)
        }) {
            "committed word '${decision.committedWord}' is absent from its own slate"
        }
        return Measured(slate, commitJson(decision), elapsed, trace.toString())
    }

    /**
     * The next-word suggestion path exactly as production runs it: an empty composer means
     * `!isComposingWord`, so `getSuggestedWords` takes the `resultsArePredictions` branch, calls
     * `getNextWordSuggestions`, and fuses with `InputStyle.PREDICTION`. There is no commit to
     * measure; the strip order is the whole result.
     */
    private fun measurePrediction(fixture: Fixture, precedingContext: String): Measured {
        val composer = WordComposer()
        val trace = StringBuilder()
        fixture.suggest.autoCorrectionTrace = trace
        val start = System.nanoTime()
        val suggested = fixture.suggest.getSuggestedWords(
            composer, ngram(precedingContext), fixture.keyboard,
            fixture.settingsForSuggestion, true, SuggestedWords.INPUT_STYLE_TYPING, 1,
        )
        val elapsed = (System.nanoTime() - start) / 1_000_000.0
        fixture.suggest.autoCorrectionTrace = null
        return Measured(
            surfaces(suggested),
            JSONObject().put("committed", JSONObject.NULL).put("willAutoCorrect", false),
            elapsed,
            trace.toString(),
        )
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

    @Test fun dumpSwipeLexicon() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardDumpSwipeLexicon") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixture = buildFixture(context)
        val words = fixture.facilitator.getSwipeLexiconWords(listOf("en-US"), 0, Int.MAX_VALUE, false)
        val output = File(context.filesDir, "swipe-lexicon-dump.txt")
        output.bufferedWriter().use { writer ->
            for (word in words) writer.append(word.word).append('\n')
        }
        Log.i(TAG, "dumped ${words.size} swipe lexicon words to ${output.absolutePath}")
        assertTrue(words.isNotEmpty())
    }

    @Test fun replayTapCorpusThroughMeasuredSystems() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardRequirePhase0Measurement") == "true")
        val runId = requireNotNull(args.getString("phase0RunId")) { "phase0RunId is required" }
        require(runId.length in 1..512)
        val environment = requireNotNull(args.getString("phase0Environment")) { "phase0Environment is required" }
        require(environment in setOf("stock_android_hardware", "grapheneos_hardware", "low_ram_emulator"))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val corpus = measuredFile(context.filesDir, requireNotNull(args.getString("phase0CorpusFile")))
        val output = measuredFile(
            outputBase(context, args, requireNotNull(corpus.parentFile)),
            requireNotNull(args.getString("phase0OutputFile")))
        require(corpus.isFile && corpus.length() in 1..(512L * 1024 * 1024))
        require(!output.isDirectory)
        output.parentFile?.mkdirs()
        val limit = args.getString("phase0Limit")?.toIntOrNull() ?: Int.MAX_VALUE
        require(limit > 0)
        val (shardIndex, shardCount) = shardConfig(args)

        val fixture = buildFixture(context)
        val memory = MemoryProbe(Debug.getPss().toDouble())
        args.getString("phase0RescorerBudgetMs")?.toLongOrNull()?.let {
            fixture.suggest.contextRescoringBudgetOverrideMs = it
        }
        args.getString("phase0ModelDir")?.let { installContextModel(context, measuredFile(context.filesDir, it)) }
        memory.onModelOpened()

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
                val commits = JSONObject()
                val gateTraces = JSONObject()
                val raw = row.getString("raw")

                // Every system must start from the same state. The previous row left personal
                // entries seeded and neural strength raised, which leaked into this row's baseline.
                PersonalizationRuntime.wipe(fixture.context)
                setMeasurementConfiguration(fixture, personalizedDicts = false, neuralStrength = 0f)

                val baselineComposer = composerFor(fixture, row)
                val baselineNgram = ngram(row.optString("precedingContext"))
                val baselineTrace = StringBuilder()
                fixture.suggest.autoCorrectionTrace = baselineTrace
                var start = System.nanoTime()
                val classic = fixture.facilitator.getSuggestionResults(
                    baselineComposer.composedDataSnapshot,
                    baselineNgram, fixture.keyboard,
                    fixture.settingsForSuggestion, Suggest.SESSION_ID_TYPING,
                    SuggestedWords.INPUT_STYLE_TYPING,
                )
                // The classic baseline has no SuggestedWords to read mWillAutoCorrect from, so the
                // decision comes from production's own shouldBeAutoCorrected via Suggest.
                val baselineDecision = fixture.suggest.classicCommitDecision(
                    baselineComposer, baselineNgram, fixture.keyboard,
                    fixture.settingsForSuggestion, SuggestedWords.INPUT_STYLE_TYPING, classic,
                )
                fixture.suggest.autoCorrectionTrace = null
                gateTraces.put("heliboard", baselineTrace.toString())
                latency.put("heliboard", (System.nanoTime() - start) / 1_000_000.0)
                val seen = LinkedHashSet<String>()
                val baseline = JSONArray()
                // The classic strip seats the typed word first, exactly as production does; scoring
                // now reads the commit decision, so this ordering no longer decides top-1.
                for (word in listOf(raw) + classic.map { it.mWord }) {
                    if (word.isNotBlank() && seen.add(normalizeCandidate(word))) baseline.put(word)
                    if (baseline.length() >= 32) break
                }
                predictions.put("heliboard", baseline)
                commits.put("heliboard", commitJson(baselineDecision))

                val fused = measure(fixture, row)
                predictions.put("fused", fused.surfaces); latency.put("fused", fused.elapsedMs)
                commits.put("fused", fused.commit)
                gateTraces.put("fused", fused.gateTrace)

                seedPersonalFixture(fixture, row)
                setMeasurementConfiguration(fixture, personalizedDicts = true, neuralStrength = 0f)
                val personal = measure(fixture, row)
                predictions.put("fused_personal", personal.surfaces)
                latency.put("fused_personal", personal.elapsedMs)
                commits.put("fused_personal", personal.commit)
                gateTraces.put("fused_personal", personal.gateTrace)
                memory.onNeuralDisabled()

                setMeasurementConfiguration(fixture, personalizedDicts = true, neuralStrength = 50f)
                val neural = measure(fixture, row)
                predictions.put("fused_neural", neural.surfaces)
                latency.put("fused_neural", neural.elapsedMs)
                commits.put("fused_neural", neural.commit)
                gateTraces.put("fused_neural", neural.gateTrace)
                memory.onNeuralEnabled()

                writer.write(JSONObject(row.toString()).apply {
                    put("schemaVersion", SCHEMA_VERSION)
                    put("environmentKind", environment)
                    put("testRunId", runId)
                    put("predictions", predictions)
                    put("latencyMs", latency)
                    put("commits", commits)
                    put("gateTraces", gateTraces)
                }.toString())
                writer.write("\n")
                measured++
            }
        }
        writer.close()
        assertTrue("corpus produced no measurable test rows", measured > 0)
        val sidecar = File(output.parentFile, "environment.json")
        writeEnvironmentSidecar(context, sidecar, runId, environment, memory, fixture)
        publishForHost(output, sidecar)
    }

    /**
     * Next-word prediction replay: for every tap-corpus row carrying a non-empty
     * `precedingContext`, measure the PREDICTION strip that production would show after the user
     * commits the previous word. The target is the word the corpus author actually typed next.
     * `phase0RescorerBudgetMs` optionally lifts the production rescorer budget so diagnostics can
     * observe completed neural scores on slow runtimes; it never changes the default.
     */
    @Test fun replayPredictionCorpusThroughMeasuredSystems() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.getString("libreboardRequirePhase0PredictionMeasurement") == "true")
        val runId = requireNotNull(args.getString("phase0RunId")) { "phase0RunId is required" }
        require(runId.length in 1..512)
        val environment = requireNotNull(args.getString("phase0Environment")) { "phase0Environment is required" }
        require(environment in setOf("stock_android_hardware", "grapheneos_hardware", "low_ram_emulator"))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val corpus = measuredFile(context.filesDir, requireNotNull(args.getString("phase0CorpusFile")))
        val output = measuredFile(
            outputBase(context, args, requireNotNull(corpus.parentFile)),
            requireNotNull(args.getString("phase0PredictionOutputFile")))
        require(corpus.isFile && corpus.length() in 1..(512L * 1024 * 1024))
        require(!output.isDirectory)
        output.parentFile?.mkdirs()
        val limit = args.getString("phase0Limit")?.toIntOrNull() ?: Int.MAX_VALUE
        require(limit > 0)
        val (shardIndex, shardCount) = shardConfig(args)

        val fixture = buildFixture(context)
        val memory = MemoryProbe(Debug.getPss().toDouble())
        args.getString("phase0RescorerBudgetMs")?.toLongOrNull()?.let {
            fixture.suggest.contextRescoringBudgetOverrideMs = it
        }
        args.getString("phase0ModelDir")?.let { installContextModel(context, measuredFile(context.filesDir, it)) }
        memory.onModelOpened()

        val writer = output.bufferedWriter()
        var measured = 0
        var ordinal = 0
        corpus.bufferedReader().useLines { lines ->
            for (line in lines) {
                if (measured >= limit) break
                if (line.isBlank()) continue
                val row = JSONObject(line)
                if (row.getString("split") != "test" || row.getString("category") == "swipe") continue
                val precedingContext = row.optString("precedingContext")
                if (precedingContext.isBlank()) continue
                if (ordinal++ % shardCount != shardIndex) continue

                val predictions = JSONObject()
                val latency = JSONObject()
                val gateTraces = JSONObject()

                PersonalizationRuntime.wipe(fixture.context)
                setMeasurementConfiguration(fixture, personalizedDicts = false, neuralStrength = 0f)
                val fused = measurePrediction(fixture, precedingContext)
                predictions.put("fused", fused.surfaces); latency.put("fused", fused.elapsedMs)
                gateTraces.put("fused", fused.gateTrace)
                memory.onNeuralDisabled()

                setMeasurementConfiguration(fixture, personalizedDicts = false, neuralStrength = 50f)
                val neural = measurePrediction(fixture, precedingContext)
                predictions.put("fused_neural", neural.surfaces)
                latency.put("fused_neural", neural.elapsedMs)
                gateTraces.put("fused_neural", neural.gateTrace)
                memory.onNeuralEnabled()

                writer.write(JSONObject(row.toString()).apply {
                    put("schemaVersion", SCHEMA_VERSION)
                    put("category", "prediction")
                    put("raw", "")
                    put("environmentKind", environment)
                    put("testRunId", runId)
                    put("predictions", predictions)
                    put("latencyMs", latency)
                    put("gateTraces", gateTraces)
                }.toString())
                writer.write("\n")
                measured++
            }
        }
        writer.close()
        assertTrue("prediction corpus produced no measurable test rows", measured > 0)
        val sidecar = File(output.parentFile, "environment.json")
        writeEnvironmentSidecar(context, sidecar, runId, environment, memory, fixture)
        publishForHost(output, sidecar)
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

    /** filesDir-relative extras stay the debug-APK contract; absolute paths skip `run-as`. */
    private fun measuredFile(base: File, path: String): File {
        val file = File(path)
        return if (file.isAbsolute) file else File(base, path)
    }

    /**
     * Non-debuggable testOnly APKs cannot use `run-as`. Outputs go to the app's own
     * getExternalFilesDir so the host can adb-cat them after publishForHost.
     */
    private fun outputBase(context: android.content.Context, args: android.os.Bundle, fallback: File): File {
        if (args.getString("phase0UseExternalFiles") == "true") {
            return requireNotNull(context.getExternalFilesDir("phase0-measurement")) {
                "getExternalFilesDir(phase0-measurement) returned null"
            }
        }
        args.getString("phase0OutputDir")?.let { return measuredFile(context.filesDir, it) }
        return fallback
    }

    /** World-readable so the host can `adb cat` outputs of a non-debuggable testOnly APK. */
    private fun publishForHost(vararg files: File) {
        for (file in files) {
            if (file.isFile) {
                check(file.setReadable(true, false)) {
                    "cannot make ${file.path} world-readable for adb pull"
                }
            }
        }
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
        val corpus = measuredFile(context.filesDir, requireNotNull(args.getString("phase0SwipeCorpusFile")))
        val output = measuredFile(
            outputBase(context, args, requireNotNull(corpus.parentFile)),
            requireNotNull(args.getString("phase0SwipeOutputFile")))
        require(corpus.isFile && corpus.length() in 1..(512L * 1024 * 1024))
        require(!output.isDirectory)
        output.parentFile?.mkdirs()
        val limit = args.getString("phase0Limit")?.toIntOrNull() ?: Int.MAX_VALUE
        require(limit > 0)
        val (shardIndex, shardCount) = shardConfig(args)

        val fixture = buildFixture(context)
        val memory = MemoryProbe(Debug.getPss().toDouble())
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
                measuredFile(context.filesDir, name), LibreBoardOnnxContracts.swipeCtc)
            val runtime = requireNotNull(opened.session) {
                "swipe ONNX session unavailable: ${opened.availability}" }
            val session = OnnxCtcInferenceSession(runtime)
            CtcSwipeDecoder(session, lexicon).also { decoder ->
                LiveTypingEngine.swipeDecoder.install(decoder, session)
            }
        }
        args.getString("phase0ModelDir")?.let { installContextModel(context, measuredFile(context.filesDir, it)) }
        memory.onModelOpened()

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
                val swipeTrace = Suggest.SwipeBatchTrace()
                fixture.suggest.swipeBatchTrace = swipeTrace
                start = System.nanoTime()
                val fused = fixture.suggest.getSuggestedWords(
                    composer, ngramContext,
                    swipeKeyboard, fixture.settingsForSuggestion, true,
                    SuggestedWords.INPUT_STYLE_UPDATE_BATCH, measured + 1,
                )
                latency.put("fused_swipe", (System.nanoTime() - start) / 1_000_000.0)
                fixture.suggest.swipeBatchTrace = null
                predictions.put("fused_swipe", surfaces(fused).also {
                    if (it.length() == 0) it.put(EMPTY_PREDICTION_SENTINEL)
                })
                availability.put("fused", swipeTrace.decodeAvailability)
                val auxiliaryLatency = JSONObject()
                auxiliaryLatency.put("fused_decode", swipeTrace.decodeMs)
                auxiliaryLatency.put("fused_fusion", swipeTrace.fusionMs)
                auxiliaryLatency.put("fused_next_word", swipeTrace.nextWordMs)
                auxiliaryLatency.put("fused_ctc_candidates", swipeTrace.ctcCandidates.toDouble())
                auxiliaryLatency.put("fused_geometric_candidates", swipeTrace.geometricCandidates.toDouble())

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
                // Swipe replays the neural rescorer on every row, so there is no neural-disabled
                // sample to subtract here; the recorded breakdown says so rather than implying one.
                memory.onNeuralEnabled()

                writer.write(JSONObject(row.toString()).apply {
                    put("schemaVersion", SCHEMA_VERSION)
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
        val sidecar = File(output.parentFile, "environment.json")
        writeEnvironmentSidecar(context, sidecar, runId, environment, memory, fixture)
        publishForHost(output, sidecar)
    }

    private fun writeEnvironmentSidecar(
        context: android.content.Context, file: File, runId: String, environment: String,
        memory: MemoryProbe, fixture: Fixture,
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
            .put("peakAddedNeuralMemoryMiB", memory.addedKiB / 1024.0)
            .put("memoryAttribution", memory.toJson())
            .put("fixtureReadinessMs", JSONObject()
                .put("mainDictionary", fixture.dictionaryReadyMs)
                .put("swipeLexicon", fixture.swipeLexiconReadyMs)
                .put("scope", "One-time synchronous setup before the first measured row. It is " +
                    "excluded from every latencyMs value, so a passing suggestion-call latency " +
                    "says nothing about this cost."))
            .put("latencyScope", "latencyMs times the suggestion call only: " +
                "Suggest.getSuggestedWords for the fused systems and the facilitator lookup plus " +
                "its commit decision for the baseline. It excludes editor commit, composing-span " +
                "and UI work, and fixture readiness, so it is not end-to-end IME latency.")
        if (environment == "grapheneos_hardware") {
            sidecar.put("grapheneOsBuildNumber", Build.DISPLAY)
            // gmscompat is a GrapheneOS system app present on every profile; the sandboxed
            // Play check must look for the actual Google packages (as capture_android_device.py does).
            sidecar.put("sandboxedGooglePlayInstalled", SANDBOXED_PLAY_PACKAGES.any { pkg ->
                runCatching { context.packageManager.getPackageInfo(pkg, 0); true }.getOrDefault(false)
            })
        }
        file.writeText(sidecar.toString())
    }

    private companion object {
        /** Standalone decoder budget; production gives decoders 125 ms inside fused_swipe. */
        const val SWIPE_STANDALONE_BUDGET_MS = 1500L

        /** Placeholder emitted when a system returns no candidates; always a miss. */
        const val EMPTY_PREDICTION_SENTINEL = "__no_candidates__"

        const val TAG = "Phase0Measurement"

        /** evaluate_engine.py measurement schema. 4 adds the per-system commit decision. */
        const val SCHEMA_VERSION = 4

        /** User-installed package IDs that indicate sandboxed Google Play is provisioned. */
        val SANDBOXED_PLAY_PACKAGES = arrayOf(
            "com.google.android.gms",
            "com.google.android.gsf",
            "com.android.vending",
        )
    }
}
