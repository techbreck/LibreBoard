// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.ctc

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TouchPoint
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.WordLock
import helium314.keyboard.latin.engine.geometric.LexiconWord
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import helium314.keyboard.latin.engine.geometric.SwipeLengthEstimate
import helium314.keyboard.latin.engine.geometric.SwipeWordGesture
import helium314.keyboard.latin.engine.integration.RevisingSwipeLexicon
import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.Locale
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.math.max

/** Fixed tensor ABI consumed by swipe-latin-v1.onnx. Arrays are row-major. */
data class CtcSwipeFeatures(
    /** Shape `[1, 64, 2]`; x/y are normalized against the live letter-key bounds. */
    val pathCoordinates: FloatArray,
    /** Shape `[1, 64, 2]`; unused slots are zero. */
    val keyCenters: FloatArray,
    /** Shape `[1, 64]`; one for an enabled key slot and zero for padding/disabled slots. */
    val keyMask: FloatArray,
    /** The surface emitted by output classes 1 through 64. Class zero is CTC blank. */
    val keyLabels: List<String?>,
)

data class CtcInferenceResult(
    val availability: EngineAvailability,
    /** Shape `[1, 32, 65]`; values are unnormalized logits. */
    val logits: FloatArray? = null,
    val frameCount: Int = 0,
    val classCount: Int = 0,
)

/** The source-built ONNX bridge implements this interface; decoding policy stays in Kotlin. */
fun interface CtcInferenceSession {
    fun infer(features: CtcSwipeFeatures, deadline: Deadline): CtcInferenceResult
}

/**
 * Layout-conditioned CTC swipe decoder with lexicon-constrained prefix beam search.
 *
 * Runtime absence and malformed output are explicit states. They never throw through the shared
 * candidate pipeline, so the geometric decoder running beside this path can always publish.
 */
class CtcSwipeDecoder(
    private val inferenceSession: CtcInferenceSession,
    private val lexicon: SwipeLexicon,
    private val beamWidth: Int = DEFAULT_BEAM_WIDTH,
    private val maximumLexiconWords: Int = MAX_LEXICON_WORDS,
) : SwipeDecoder {
    init {
        require(beamWidth in 1..MAX_BEAM_WIDTH)
        require(maximumLexiconWords > 0)
    }

    override fun decode(request: TypingRequest, deadline: Deadline): SwipeDecodeResult {
        require(request.inputStyle == InputStyle.SWIPE)
        if (request.path.size < 2) return SwipeDecodeResult(EngineAvailability.AVAILABLE)
        if (deadline.expired) return SwipeDecodeResult(EngineAvailability.TIMEOUT)

        val features = runCatching { featureTensor(request.path, request.geometry) }
            .getOrElse { return SwipeDecodeResult(EngineAvailability.INCOMPATIBLE) }
        val inference = runCatching { inferenceSession.infer(features, deadline) }
            .getOrElse { return SwipeDecodeResult(EngineAvailability.UNAVAILABLE) }
        if (inference.availability != EngineAvailability.AVAILABLE) {
            return SwipeDecodeResult(inference.availability)
        }
        if (!validOutput(inference)) return SwipeDecodeResult(EngineAvailability.INCOMPATIBLE)
        if (deadline.expired) return SwipeDecodeResult(EngineAvailability.TIMEOUT)

        val languageTags = when (val lock = request.wordLock) {
            is WordLock.Manual -> listOf(lock.languageTag).filter(request.enabledLanguages::contains)
            else -> request.enabledLanguages
        }
        if (languageTags.isEmpty()) return SwipeDecodeResult(EngineAvailability.INCOMPATIBLE)
        val greedy = greedyEmission(inference)
        val approximateLength = greedy.size.takeIf { it > 0 }
            ?: SwipeLengthEstimate.fromPath(request.path, request.geometry)
        // The trie only depends on the lexicon contents, language set, length estimate, and key
        // labels. Rebuilding it per decode costs a full lexicon pass inside the shared proposal
        // deadline, so keep a bounded cache keyed on the lexicon's own revision token.
        val revision = (lexicon as? RevisingSwipeLexicon)?.contentRevision
        val cacheKey = revision?.let { TrieCacheKey(it, languageTags, approximateLength, features.keyLabels) }
        val trie = (cacheKey?.let { key -> synchronized(trieCache) { trieCache[key] } }
            ?: buildTrie(
                lexicon.words(languageTags, approximateLength),
                features.keyLabels,
                languageTags.toSet(),
                deadline,
            )?.also { built ->
                if (cacheKey != null) synchronized(trieCache) { trieCache[cacheKey] = built }
            }) ?: return SwipeDecodeResult(EngineAvailability.TIMEOUT)
        if (trie.wordCount == 0) return SwipeDecodeResult(EngineAvailability.AVAILABLE)

        return decodeLogits(inference, trie, deadline, greedy)
    }

    private data class TrieCacheKey(
        val contentRevision: Any,
        val languageTags: List<String>,
        val approximateLength: Int,
        val keyLabels: List<String?>,
    )

    private val trieCache = object : LinkedHashMap<TrieCacheKey, LexiconTrie>(16, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<TrieCacheKey, LexiconTrie>?) =
            size > MAX_CACHED_TRIES
    }

    private fun decodeLogits(
        inference: CtcInferenceResult,
        trie: LexiconTrie,
        deadline: Deadline,
        greedy: IntArray,
    ): SwipeDecodeResult {
        val logits = requireNotNull(inference.logits)
        var beam = mapOf(IntArrayKey.EMPTY to BeamProbability(blank = 0.0))

        for (frame in 0 until inference.frameCount) {
            if (deadline.expired) {
                return SwipeDecodeResult(
                    EngineAvailability.TIMEOUT, candidatesForBeam(beam, trie, inference.frameCount, greedy),
                )
            }
            val frameOffset = frame * inference.classCount
            val logProbabilities = logSoftmax(logits, frameOffset, inference.classCount)
            val next = HashMap<IntArrayKey, BeamProbability>(beamWidth * 4)
            beam.forEach { (prefix, probability) ->
                val node = trie.node(prefix.values) ?: return@forEach
                val total = logAdd(probability.blank, probability.nonBlank)
                next.getOrPut(prefix) { BeamProbability() }.blank = logAdd(
                    next.getValue(prefix).blank,
                    total + logProbabilities[BLANK_CLASS],
                )
                for (outputClass in 1 until inference.classCount) {
                    val repeatedClass = prefix.values.lastOrNull() == outputClass
                    if (repeatedClass) {
                        // A repeated emission without an intervening blank keeps the same prefix.
                        val same = next.getOrPut(prefix) { BeamProbability() }
                        same.nonBlank = logAdd(
                            same.nonBlank,
                            probability.nonBlank + logProbabilities[outputClass],
                        )
                    }
                    val child = node.child(outputClass) ?: continue
                    val extensionProbability = if (repeatedClass) probability.blank else total
                    if (extensionProbability == LOG_ZERO) continue
                    val extended = prefix.append(outputClass)
                    val target = next.getOrPut(extended) { BeamProbability() }
                    target.nonBlank = logAdd(
                        target.nonBlank,
                        extensionProbability + logProbabilities[outputClass],
                    )
                    // Retain the reference so trie reachability is checked while expanding, not
                    // only after the potentially expensive beam has grown.
                    check(child.depth == extended.values.size)
                }
            }
            // These probabilities are now final for this frame. Avoid recalculating the
            // logarithmic sum twice per comparison while sorting the expanded beam.
            next.values.forEach { it.rankingScore = logAdd(it.blank, it.nonBlank) }
            beam = next.entries
                .sortedByDescending { it.value.rankingScore }
                .take(beamWidth)
                .associateTo(LinkedHashMap()) { it.key to it.value }
        }
        // Intermediate slates are never published. Materialize only the final (or timed-out)
        // beam, retaining the same scores and ordering without sorting 32 candidate lists.
        return SwipeDecodeResult(
            EngineAvailability.AVAILABLE, candidatesForBeam(beam, trie, inference.frameCount, greedy),
        )
    }

    private fun candidatesForBeam(
        beam: Map<IntArrayKey, BeamProbability>,
        trie: LexiconTrie,
        frameCount: Int,
        greedy: IntArray,
    ): List<Candidate> = beam.flatMap { (prefix, probability) ->
        val ctcScore = logAdd(probability.blank, probability.nonBlank) / frameCount
        val matchesGreedy = greedy.isNotEmpty() && prefix.values.contentEquals(greedy)
        trie.node(prefix.values)?.words.orEmpty().map { entry ->
            matchesGreedy to Candidate(
                surface = entry.word,
                languageTag = entry.languageTag,
                sources = buildSet {
                    add(CandidateSource.CTC_SWIPE)
                    if (entry.personal) add(CandidateSource.PERSONAL)
                },
                components = ScoreComponents(
                    spatial = ctcScore,
                    staticFrequency = ln1p(entry.frequency.coerceAtLeast(0).toDouble()),
                    personal = if (entry.personal) 1.0 else null,
                ),
                exactPersonalMatch = entry.personal,
                totalScore = ctcScore,
            )
        }
    }.sortedWith(
        compareByDescending<Pair<Boolean, Candidate>> { it.first }
            .thenByDescending { it.second.components.spatial }
            .thenByDescending { it.second.components.staticFrequency }
            .thenBy { it.second.surface },
    ).map { it.second }.distinctBy { normalizeCandidate(it.surface) to it.languageTag }.take(MAX_RESULTS)

    private fun buildTrie(
        words: Sequence<LexiconWord>,
        keyLabels: List<String?>,
        allowedLanguageTags: Set<String>,
        deadline: Deadline,
    ): LexiconTrie? {
        val classByLabel = keyLabels.mapIndexedNotNull { index, label ->
            label?.let { normalizeCandidate(it) to index + 1 }
        }.toMap()
        val asciiClasses = IntArray(128)
        classByLabel.forEach { (label, outputClass) ->
            if (label.length == 1 && label[0].code < asciiClasses.size) asciiClasses[label[0].code] = outputClass
        }
        val trie = LexiconTrie()
        val iterator = words.iterator()
        var inspected = 0
        while (iterator.hasNext() && inspected < maximumLexiconWords) {
            if (deadline.expired) return null
            inspected++
            val word = iterator.next()
            if (word.languageTag !in allowedLanguageTags) continue
            if (word.word.length in 1..MAX_EMISSION_LENGTH && word.word.all { it in 'a'..'z' }) {
                // This is the single unchanged variant from SwipeWordGesture. Validate the
                // entire spelling before adding nodes: unreachable partial words would alter
                // the beam. Avoid allocating variant lists and emission arrays per word.
                if (word.word.all { asciiClasses[it.code] != 0 }) trie.addAscii(word, asciiClasses)
                continue
            }
            val emissions = emissionClasses(word, classByLabel, asciiClasses)
            emissions.forEach { trie.add(it, word) }
        }
        return trie
    }

    private fun emissionClasses(word: LexiconWord, classByLabel: Map<String, Int>, asciiClasses: IntArray): List<IntArray> {
        val emissions = SwipeWordGesture.variants(word.word, word.languageTag).mapNotNull { gesture ->
            val classes = IntArray(gesture.length)
            var count = 0
            var index = 0
            while (index < gesture.length) {
                val codePoint = gesture.codePointAt(index)
                val outputClass = if (codePoint < asciiClasses.size) asciiClasses[codePoint] else
                    classByLabel[String(Character.toChars(codePoint))] ?: 0
                if (outputClass == 0) return@mapNotNull null
                classes[count++] = outputClass
                index += Character.charCount(codePoint)
            }
            if (count == 0 || count > MAX_EMISSION_LENGTH) null else
                if (count == classes.size) classes else classes.copyOf(count)
        }
        return if (emissions.size <= 1) emissions else emissions.distinctBy { it.toList() }
    }

    private fun validOutput(result: CtcInferenceResult): Boolean {
        val logits = result.logits ?: return false
        return result.frameCount == OUTPUT_FRAMES &&
            result.classCount == OUTPUT_CLASSES &&
            logits.size == OUTPUT_FRAMES * OUTPUT_CLASSES &&
            logits.all(Float::isFinite)
    }

    internal companion object {
        const val PATH_POINTS = 64
        const val KEY_SLOTS = 64
        const val OUTPUT_FRAMES = 32
        const val OUTPUT_CLASSES = KEY_SLOTS + 1
        const val BLANK_CLASS = 0
        const val MAX_RESULTS = 32
        const val DEFAULT_BEAM_WIDTH = 64
        const val MAX_BEAM_WIDTH = 256
        const val MAX_LEXICON_WORDS = 100_000
        const val MAX_CACHED_TRIES = 32
        const val MAX_EMISSION_LENGTH = 64
        const val LOG_ZERO = Double.NEGATIVE_INFINITY

        internal fun featureTensor(path: List<TouchPoint>, geometry: KeyGeometry): CtcSwipeFeatures {
            require(path.size >= 2)
            require(path.all { it.x.isFinite() && it.y.isFinite() }) { "swipe path is not finite" }
            val keys = geometry.keys.take(KEY_SLOTS)
            require(keys.all {
                it.centerX.isFinite() && it.centerY.isFinite() &&
                    it.width.isFinite() && it.height.isFinite() && it.width > 0f && it.height > 0f
            }) { "live keyboard geometry is invalid" }
            require(keys.any { it.enabled }) { "live keyboard has no enabled key slots" }
            val enabledKeys = keys.filter { it.enabled }
            val letterKeys = enabledKeys.filter { key ->
                val codePoint = key.label.codePointAtOrNull(0)
                codePoint != null && key.label.codePointCount(0, key.label.length) == 1 &&
                    Character.isLetter(codePoint)
            }.ifEmpty { enabledKeys }
            val left = letterKeys.minOf { it.centerX - it.width / 2f }
            val right = letterKeys.maxOf { it.centerX + it.width / 2f }
            val top = letterKeys.minOf { it.centerY - it.height / 2f }
            val bottom = letterKeys.maxOf { it.centerY + it.height / 2f }
            val width = (right - left).coerceAtLeast(1f)
            val height = (bottom - top).coerceAtLeast(1f)
            fun normalizedX(value: Float) = ((value - left) / width).coerceIn(-0.5f, 1.5f)
            fun normalizedY(value: Float) = ((value - top) / height).coerceIn(-0.5f, 1.5f)

            val sampled = resample(path, PATH_POINTS)
            val pathCoordinates = FloatArray(PATH_POINTS * 2)
            sampled.forEachIndexed { index, point ->
                pathCoordinates[index * 2] = normalizedX(point.x)
                pathCoordinates[index * 2 + 1] = normalizedY(point.y)
            }
            val keyCenters = FloatArray(KEY_SLOTS * 2)
            val keyMask = FloatArray(KEY_SLOTS)
            val keyLabels = MutableList<String?>(KEY_SLOTS) { null }
            keys.forEachIndexed { index, key ->
                keyCenters[index * 2] = normalizedX(key.centerX)
                keyCenters[index * 2 + 1] = normalizedY(key.centerY)
                val validLabel = key.label.lowercase(Locale.ROOT).takeIf {
                    it.codePointCount(0, it.length) == 1 && it.isNotBlank()
                }
                if (key.enabled && validLabel != null) {
                    keyMask[index] = 1f
                    keyLabels[index] = validLabel
                }
            }
            require(keyLabels.filterNotNull().map(::normalizeCandidate).distinct().size == keyLabels.count { it != null }) {
                "CTC output key labels must be unique"
            }
            return CtcSwipeFeatures(pathCoordinates, keyCenters, keyMask, keyLabels)
        }

        private fun resample(points: List<TouchPoint>, count: Int): List<TouchPoint> {
            val cumulative = FloatArray(points.size)
            for (index in 1 until points.size) {
                cumulative[index] = cumulative[index - 1] + hypot(
                    (points[index].x - points[index - 1].x).toDouble(),
                    (points[index].y - points[index - 1].y).toDouble(),
                ).toFloat()
            }
            val total = cumulative.last()
            if (total <= 0f) return List(count) { points.first() }
            return List(count) { sampleIndex ->
                val target = total * sampleIndex / (count - 1)
                var upper = cumulative.binarySearch(target)
                if (upper < 0) upper = -upper - 1
                upper = upper.coerceIn(1, points.lastIndex)
                val lower = upper - 1
                val span = cumulative[upper] - cumulative[lower]
                val fraction = if (span == 0f) 0f else (target - cumulative[lower]) / span
                TouchPoint(
                    x = points[lower].x + (points[upper].x - points[lower].x) * fraction,
                    y = points[lower].y + (points[upper].y - points[lower].y) * fraction,
                    timeMillis = points[lower].timeMillis +
                        ((points[upper].timeMillis - points[lower].timeMillis) * fraction).toLong(),
                )
            }
        }

        private fun logSoftmax(values: FloatArray, offset: Int, count: Int): DoubleArray {
            val maximum = (offset until offset + count).maxOf { values[it].toDouble() }
            var sum = 0.0
            for (index in offset until offset + count) sum += exp(values[index] - maximum)
            val denominator = maximum + ln(sum)
            return DoubleArray(count) { values[offset + it] - denominator }
        }

        internal fun greedyEmission(result: CtcInferenceResult): IntArray {
            val logits = result.logits ?: return IntArray(0)
            if (result.frameCount <= 0 || result.classCount <= 1 ||
                logits.size != result.frameCount * result.classCount
            ) return IntArray(0)
            val emissions = IntArray(result.frameCount)
            var previous = -1
            var count = 0
            for (frame in 0 until result.frameCount) {
                val offset = frame * result.classCount
                var bestClass = 0
                var bestLogit = logits[offset]
                for (outputClass in 1 until result.classCount) {
                    val value = logits[offset + outputClass]
                    if (value > bestLogit) {
                        bestLogit = value
                        bestClass = outputClass
                    }
                }
                if (bestClass != BLANK_CLASS && bestClass != previous) {
                    emissions[count++] = bestClass
                }
                previous = bestClass
            }
            return if (count == emissions.size) emissions else emissions.copyOf(count)
        }

        internal fun greedyEmissionLength(result: CtcInferenceResult): Int = greedyEmission(result).size

        private fun logAdd(first: Double, second: Double): Double {
            if (first == LOG_ZERO) return second
            if (second == LOG_ZERO) return first
            val maximum = max(first, second)
            return maximum + ln(exp(first - maximum) + exp(second - maximum))
        }

        private fun String.codePointAtOrNull(index: Int): Int? =
            takeIf { index in indices }?.codePointAt(index)
    }

    private class LexiconTrie {
        val root = TrieNode(0)
        var wordCount = 0
            private set

        fun add(classes: IntArray, word: LexiconWord) {
            var node = root
            classes.forEach { outputClass ->
                node = node.getOrCreateChild(outputClass)
            }
            addTerminal(node, word)
        }

        fun addAscii(word: LexiconWord, asciiClasses: IntArray) {
            var node = root
            word.word.forEach { node = node.getOrCreateChild(asciiClasses[it.code]) }
            addTerminal(node, word)
        }

        private fun addTerminal(node: TrieNode, word: LexiconWord) {
            if (node.words.none { it.word == word.word && it.languageTag == word.languageTag }) {
                node.words = if (node.words.isEmpty()) listOf(word) else node.words + word
                wordCount++
            }
        }

        fun node(classes: IntArray): TrieNode? {
            var node = root
            classes.forEach { outputClass -> node = node.child(outputClass) ?: return null }
            return node
        }
    }

    private class TrieNode(val depth: Int) {
        // Most dictionary nodes have zero or one child and no terminal word. Avoid allocating
        // two collections (plus their backing tables) for every character of every gesture.
        private var firstClass = 0
        private var firstChild: TrieNode? = null
        private var otherChildren: HashMap<Int, TrieNode>? = null
        var words: List<LexiconWord> = emptyList()

        fun child(outputClass: Int): TrieNode? =
            if (outputClass == firstClass) firstChild else otherChildren?.get(outputClass)

        fun getOrCreateChild(outputClass: Int): TrieNode {
            child(outputClass)?.let { return it }
            val created = TrieNode(depth + 1)
            if (firstChild == null) {
                firstClass = outputClass
                firstChild = created
            } else {
                val remaining = otherChildren ?: HashMap<Int, TrieNode>(2).also { otherChildren = it }
                remaining[outputClass] = created
            }
            return created
        }
    }

    private class BeamProbability(
        var blank: Double = LOG_ZERO,
        var nonBlank: Double = LOG_ZERO,
    ) {
        var rankingScore: Double = LOG_ZERO
    }

    private class IntArrayKey private constructor(val values: IntArray) {
        private val hash = values.contentHashCode()

        fun append(value: Int) = IntArrayKey(values + value)
        override fun equals(other: Any?) = other is IntArrayKey && values.contentEquals(other.values)
        override fun hashCode() = hash

        companion object {
            val EMPTY = IntArrayKey(IntArray(0))
        }
    }
}
