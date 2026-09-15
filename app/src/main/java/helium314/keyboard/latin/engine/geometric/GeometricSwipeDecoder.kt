// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.CandidateSource
import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.InputStyle
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.KeySlot
import helium314.keyboard.latin.engine.ScoreComponents
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TouchPoint
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.normalizeCandidate
import java.util.Locale
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.ln1p
import kotlin.math.roundToInt

data class LexiconWord(
    val word: String,
    val languageTag: String,
    val frequency: Int,
    val personal: Boolean = false,
    val possiblyOffensive: Boolean = false,
)

fun interface SwipeLexicon {
    fun words(languageTags: List<String>, approximateLength: Int): Sequence<LexiconWord>
}

/**
 * Pure Kotlin decoder used whenever the CTC model is missing, disabled, late, or incompatible.
 * It deliberately consumes live key geometry rather than assuming QWERTY coordinates.
 */
class GeometricSwipeDecoder(private val lexicon: SwipeLexicon) : SwipeDecoder {
    override fun decode(request: TypingRequest, deadline: Deadline): SwipeDecodeResult {
        require(request.inputStyle == InputStyle.SWIPE)
        if (request.path.size < 2) return SwipeDecodeResult(EngineAvailability.AVAILABLE)
        val normalizedPath = resample(request.path, SAMPLE_COUNT).map {
            Point(it.x / request.geometry.width, it.y / request.geometry.height)
        }
        val traced = TraceKeySequence.decode(request.path, request.geometry)
        val pathTurns = turnCount(normalizedPath)
        val approximateLength = SwipeLengthEstimate.fromPath(request.path, request.geometry)
        // KeyGeometry.keyFor is a linear scan; hoisting it to a map keeps the per-word endpoint
        // check cheap enough to reject most of the lexicon before any template resample.
        val keyByChar = HashMap<Char, KeySlot>(request.geometry.keys.size * 2)
        request.geometry.keys.forEach { slot ->
            val lower = slot.label.lowercase(Locale.ROOT)
            if (slot.enabled && lower.length == 1 &&
                slot.label.codePointCount(0, slot.label.length) == 1) {
                keyByChar.putIfAbsent(lower[0], slot)
            }
        }
        // Normalized key points for the cheap endpoint bound, built once per decode so the
        // per-word path allocates nothing before a variant earns the template resample.
        val keyPoint = HashMap<Char, Point>(keyByChar.size * 2)
        keyByChar.forEach { (char, slot) ->
            keyPoint[char] = Point(slot.centerX / request.geometry.width, slot.centerY / request.geometry.height)
        }
        val pathStart = normalizedPath.first()
        val pathEnd = normalizedPath.last()
        val editBufferA = IntArray(MAX_EDIT_LENGTH + 1)
        val editBufferB = IntArray(MAX_EDIT_LENGTH + 1)
        // Only the best RESULT_LIMIT candidates are ever published, so retain a bounded worst-first
        // heap. Its head is the running cutoff: every cost term is non-negative, which makes the
        // cheap endpoint/edit terms a safe lower bound for skipping the full template resample.
        // The lexicon arrives frequency-ordered; on equal costs the earlier word keeps the seat,
        // matching the previous stable sort of the unbounded list.
        val heap = java.util.PriorityQueue<ScoredWord>(
            RESULT_LIMIT, compareByDescending<ScoredWord> { it.cost }.thenByDescending { it.seq },
        )
        var seq = 0

        for (entry in lexicon.words(request.enabledLanguages, approximateLength)) {
            if (deadline.expired) {
                return SwipeDecodeResult(EngineAvailability.TIMEOUT, publish(heap))
            }
            if (entry.languageTag !in request.enabledLanguages) continue
            val worst = heap.peek()?.cost ?: Double.POSITIVE_INFINITY
            val cutoff = if (heap.size < RESULT_LIMIT) Double.POSITIVE_INFINITY else worst
            val cost = cheapestVariantCost(
                entry, normalizedPath, traced, pathTurns, pathStart, pathEnd,
                request.geometry, keyPoint, editBufferA, editBufferB, cutoff,
            ) ?: continue
            if (heap.size < RESULT_LIMIT || cost < worst) {
                if (heap.size == RESULT_LIMIT) heap.poll()
                heap.offer(ScoredWord(cost, entry, seq++))
            }
        }
        return SwipeDecodeResult(EngineAvailability.AVAILABLE, publish(heap))
    }

    private class ScoredWord(val cost: Double, val word: LexiconWord, val seq: Int)

    private fun publish(heap: java.util.PriorityQueue<ScoredWord>): List<Candidate> =
        heap.sortedWith(compareBy<ScoredWord> { it.cost }.thenBy { it.seq }).map { scored ->
            val entry = scored.word
            Candidate(
                surface = entry.word,
                languageTag = entry.languageTag,
                sources = buildSet {
                    add(CandidateSource.GEOMETRIC_SWIPE)
                    if (entry.personal) add(CandidateSource.PERSONAL)
                },
                components = ScoreComponents(
                    spatial = -scored.cost,
                    staticFrequency = ln1p(entry.frequency.coerceAtLeast(0).toDouble()),
                    personal = if (entry.personal) 1.0 else null,
                ),
                exactPersonalMatch = entry.personal,
                totalScore = -scored.cost,
            )
        }

    /**
     * The minimum gesture-variant cost for [entry], or null when no variant maps onto the live
     * geometry. Variants whose endpoint-plus-trace lower bound already exceeds [cutoff] never
     * reach the template resample; the bound is exact because the dropped terms are non-negative.
     */
    private fun cheapestVariantCost(
        entry: LexiconWord,
        normalizedPath: List<Point>,
        traced: String,
        pathTurns: Int,
        pathStart: Point,
        pathEnd: Point,
        geometry: KeyGeometry,
        keyPoint: Map<Char, Point>,
        editBufferA: IntArray,
        editBufferB: IntArray,
        cutoff: Double,
    ): Double? {
        var best = Double.POSITIVE_INFINITY
        for (gesture in SwipeWordGesture.variants(entry.word, entry.languageTag)) {
            val firstPoint = keyPoint[gesture.first().lowercaseChar()] ?: continue
            val lastPoint = keyPoint[gesture.last().lowercaseChar()] ?: continue
            val startEndCost = distance(pathStart, firstPoint) + distance(pathEnd, lastPoint)
            // The normalized edit distance is at least the length difference over the longer
            // side; check that free bound before paying for the quadratic distance itself.
            val lengthBound = abs(traced.length - gesture.length).toDouble() /
                maxOf(traced.length, gesture.length, 1)
            var lowerBound = startEndCost * 1.4 + lengthBound * 0.8
            if (lowerBound >= best || lowerBound >= cutoff) continue
            val traceEditCost = normalizedEditDistance(traced, gesture, editBufferA, editBufferB)
            lowerBound = startEndCost * 1.4 + traceEditCost * 0.8
            if (lowerBound >= best || lowerBound >= cutoff) continue
            val template = template(gesture, geometry, keyPoint) ?: continue
            val shapeCost = averageDistance(normalizedPath, template)
            val turnCost = abs(pathTurns - turnCount(template)).toDouble() /
                gesture.length.coerceAtLeast(1)
            val cost = shapeCost * 2.2 + startEndCost * 1.4 + traceEditCost * 0.8 + turnCost * 0.25
            if (cost < best) best = cost
        }
        return if (best.isFinite()) best else null
    }

    private fun template(word: String, geometry: KeyGeometry, keyPoint: Map<Char, Point>): List<Point>? {
        val keyPoints = word.lowercase().map { character ->
            keyPoint[character] ?: return null
        }
        if (keyPoints.isEmpty()) return null
        return resamplePoints(keyPoints, SAMPLE_COUNT)
    }

    private fun averageDistance(a: List<Point>, b: List<Point>): Double =
        a.indices.sumOf { distance(a[it], b[it]) } / a.size

    private fun normalizedEditDistance(
        first: String, second: String, bufferA: IntArray, bufferB: IntArray,
    ): Double {
        if (first.isEmpty() || second.isEmpty()) return 1.0
        var previous = if (second.length < bufferA.size) bufferA else IntArray(second.length + 1)
        var current = if (second.length < bufferB.size) bufferB else IntArray(second.length + 1)
        for (index in 0..second.length) previous[index] = index
        first.forEachIndexed { firstIndex, firstCharacter ->
            current[0] = firstIndex + 1
            second.forEachIndexed { secondIndex, secondCharacter ->
                current[secondIndex + 1] = minOf(
                    current[secondIndex] + 1,
                    previous[secondIndex + 1] + 1,
                    previous[secondIndex] + if (firstCharacter == secondCharacter) 0 else 1,
                )
            }
            val swap = previous; previous = current; current = swap
        }
        return previous[second.length].toDouble() / maxOf(first.length, second.length)
    }

    private fun turnCount(points: List<Point>): Int {
        var turns = 0
        var previousDirection = 0
        for (index in 2 until points.size) {
            val a = points[index - 2]
            val b = points[index - 1]
            val c = points[index]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val direction = when {
                cross > 0.002f -> 1
                cross < -0.002f -> -1
                else -> 0
            }
            if (direction != 0 && previousDirection != 0 && direction != previousDirection) turns++
            if (direction != 0) previousDirection = direction
        }
        return turns
    }

    private data class Point(val x: Float, val y: Float)

    private fun distance(a: Point, b: Point): Double = hypot((a.x - b.x).toDouble(), (a.y - b.y).toDouble())

    private fun resample(points: List<TouchPoint>, count: Int): List<TouchPoint> =
        resampleGeneric(points, count, { it.x }, { it.y }) { x, y -> TouchPoint(x, y) }

    private fun resamplePoints(points: List<Point>, count: Int): List<Point> =
        resampleGeneric(points, count, { it.x }, { it.y }) { x, y -> Point(x, y) }

    private fun <T> resampleGeneric(
        points: List<T>,
        count: Int,
        x: (T) -> Float,
        y: (T) -> Float,
        make: (Float, Float) -> T,
    ): List<T> {
        if (points.size == 1) return List(count) { points.first() }
        val cumulative = FloatArray(points.size)
        for (index in 1 until points.size) {
            cumulative[index] = cumulative[index - 1] + hypot(
                (x(points[index]) - x(points[index - 1])).toDouble(),
                (y(points[index]) - y(points[index - 1])).toDouble(),
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
            make(
                x(points[lower]) + (x(points[upper]) - x(points[lower])) * fraction,
                y(points[lower]) + (y(points[upper]) - y(points[lower])) * fraction,
            )
        }
    }

    companion object {
        private const val SAMPLE_COUNT = 64
        private const val RESULT_LIMIT = 32
        private const val MAX_EDIT_LENGTH = 64
    }
}

object TraceKeySequence {
    /** Returns the nearest-key trace with consecutive duplicates collapsed. */
    fun decode(path: List<TouchPoint>, geometry: KeyGeometry): String {
        if (path.isEmpty()) return ""
        val enabled = geometry.keys.filter { it.enabled && it.label.codePointCount(0, it.label.length) == 1 }
        if (enabled.isEmpty()) return ""
        val out = StringBuilder()
        var previousId: Int? = null
        path.forEach { point ->
            val nearest = enabled.minByOrNull { key ->
                val normalizedX = (point.x - key.centerX) / key.width.coerceAtLeast(1f)
                val normalizedY = (point.y - key.centerY) / key.height.coerceAtLeast(1f)
                normalizedX * normalizedX + normalizedY * normalizedY
            } ?: return@forEach
            if (nearest.id != previousId) {
                out.append(nearest.label.lowercase())
                previousId = nearest.id
            }
        }
        return out.toString()
    }
}

/** Language-scoped surface-to-gesture aliases; candidates retain their original tagged surface. */
internal object SwipeWordGesture {
    fun variants(word: String, languageTag: String): List<String> {
        // Lowercase ASCII letters are already NFKC-normalized in every supported language.
        if (word.isNotEmpty() && word.all { it in 'a'..'z' }) return listOf(word)
        val normalized = normalizeCandidate(word)
        // Almost every dictionary word has one spelling path. Avoid constructing a sequence,
        // set and list for each character when no German alternatives can be produced.
        if (normalized.none { it == 'ä' || it == 'ö' || it == 'ü' || it == 'ß' } ||
            Locale.forLanguageTag(languageTag).language != "de") {
            val gesture = if (normalized.any { it == '\'' || it == '\u2019' || it == '-' }) {
                normalized.filterNot { it == '\'' || it == '\u2019' || it == '-' }
            } else normalized
            return if (gesture.isEmpty()) emptyList() else listOf(gesture)
        }
        var variants = listOf("")
        var index = 0
        while (index < normalized.length) {
            val codePoint = normalized.codePointAt(index)
            val character = String(Character.toChars(codePoint))
            val alternatives = when {
                codePoint == '\''.code || codePoint == 0x2019 || codePoint == '-'.code -> listOf("")
                character == "ä" -> listOf("ä", "a")
                character == "ö" -> listOf("ö", "o")
                character == "ü" -> listOf("ü", "u")
                character == "ß" -> listOf("ß", "s", "ss")
                else -> listOf(character)
            }
            variants = variants.asSequence()
                .flatMap { prefix -> alternatives.asSequence().map { prefix + it } }
                .distinct()
                .take(MAX_VARIANTS)
                .toList()
            index += Character.charCount(codePoint)
        }
        return variants.filter(String::isNotEmpty)
    }

    private const val MAX_VARIANTS = 8
}

/**
 * Estimates the intended word length without counting every key crossed by a continuous gesture.
 *
 * A nearest-key trace is useful as a shape feature, but it is not a word-length estimate: a swipe
 * from one distant key to another naturally crosses several unrelated keys. The scale below is
 * calibrated on the session-separated validation partition of the pinned open swipe corpus. It is
 * expressed in live-key units, so resizing, one-handed mode, and split layouts do not change it.
 */
internal object SwipeLengthEstimate {
    fun fromPath(path: List<TouchPoint>, geometry: KeyGeometry): Int {
        if (path.size < 2) return 0
        val letterKeys = geometry.keys.filter {
            it.enabled && it.width.isFinite() && it.height.isFinite() && it.width > 0f && it.height > 0f &&
                it.label.codePointCount(0, it.label.length) == 1
        }
        if (letterKeys.isEmpty()) return 0
        val keyWidth = median(letterKeys.map { it.width })
        val keyHeight = median(letterKeys.map { it.height })
        if (!keyWidth.isFinite() || !keyHeight.isFinite() || keyWidth <= 0f || keyHeight <= 0f) return 0

        var pathLengthInKeys = 0.0
        for (index in 1 until path.size) {
            val previous = path[index - 1]
            val current = path[index]
            if (!previous.x.isFinite() || !previous.y.isFinite() || !current.x.isFinite() || !current.y.isFinite()) {
                return 0
            }
            pathLengthInKeys += hypot(
                ((current.x - previous.x) / keyWidth).toDouble(),
                ((current.y - previous.y) / keyHeight).toDouble(),
            )
        }
        return (pathLengthInKeys / KEY_UNITS_PER_EMISSION).roundToInt()
            .plus(1)
            .coerceIn(1, MAX_EMISSION_LENGTH)
    }

    private fun median(values: List<Float>): Float {
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2f
    }

    private const val KEY_UNITS_PER_EMISSION = 3.9
    private const val MAX_EMISSION_LENGTH = 64
}
