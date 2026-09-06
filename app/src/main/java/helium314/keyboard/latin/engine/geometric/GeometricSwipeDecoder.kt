// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.geometric

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
        val approximateLength = SwipeLengthEstimate.fromPath(request.path, request.geometry)
        val scored = mutableListOf<Candidate>()

        for (entry in lexicon.words(request.enabledLanguages, approximateLength)) {
            if (deadline.expired) {
                return SwipeDecodeResult(EngineAvailability.TIMEOUT, scored.sortedByDescending {
                    it.components.spatial
                }.take(32))
            }
            if (entry.languageTag !in request.enabledLanguages) continue
            val gestureVariants = SwipeWordGesture.variants(entry.word, entry.languageTag)
            val cost = gestureVariants.mapNotNull { gesture ->
                val template = template(gesture, request.geometry) ?: return@mapNotNull null
                val shapeCost = averageDistance(normalizedPath, template)
                val startEndCost = distance(normalizedPath.first(), template.first()) +
                    distance(normalizedPath.last(), template.last())
                val traceEditCost = normalizedEditDistance(traced, gesture)
                val turnCost = abs(turnCount(normalizedPath) - turnCount(template)).toDouble() /
                    gesture.length.coerceAtLeast(1)
                shapeCost * 2.2 + startEndCost * 1.4 + traceEditCost * 0.8 + turnCost * 0.25
            }.minOrNull() ?: continue
            scored += Candidate(
                surface = entry.word,
                languageTag = entry.languageTag,
                sources = buildSet {
                    add(CandidateSource.GEOMETRIC_SWIPE)
                    if (entry.personal) add(CandidateSource.PERSONAL)
                },
                components = ScoreComponents(
                    spatial = -cost,
                    staticFrequency = ln1p(entry.frequency.coerceAtLeast(0).toDouble()),
                    personal = if (entry.personal) 1.0 else null,
                ),
                exactPersonalMatch = entry.personal,
                totalScore = -cost,
            )
        }
        return SwipeDecodeResult(
            EngineAvailability.AVAILABLE,
            scored.sortedByDescending { it.components.spatial }.take(32),
        )
    }

    private fun template(word: String, geometry: KeyGeometry): List<Point>? {
        val keyPoints = word.lowercase().map { character ->
            val key = geometry.keyFor(character) ?: return null
            Point(key.centerX / geometry.width, key.centerY / geometry.height)
        }
        if (keyPoints.isEmpty()) return null
        return resamplePoints(keyPoints, SAMPLE_COUNT)
    }

    private fun averageDistance(a: List<Point>, b: List<Point>): Double =
        a.indices.sumOf { distance(a[it], b[it]) } / a.size

    private fun normalizedEditDistance(first: String, second: String): Double {
        if (first.isEmpty() || second.isEmpty()) return 1.0
        val previous = IntArray(second.length + 1) { it }
        val current = IntArray(second.length + 1)
        first.forEachIndexed { firstIndex, firstCharacter ->
            current[0] = firstIndex + 1
            second.forEachIndexed { secondIndex, secondCharacter ->
                current[secondIndex + 1] = minOf(
                    current[secondIndex] + 1,
                    previous[secondIndex + 1] + 1,
                    previous[secondIndex] + if (firstCharacter == secondCharacter) 0 else 1,
                )
            }
            current.copyInto(previous)
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
        val german = Locale.forLanguageTag(languageTag).language == "de"
        var variants = listOf("")
        val normalized = normalizeCandidate(word)
        var index = 0
        while (index < normalized.length) {
            val codePoint = normalized.codePointAt(index)
            val character = String(Character.toChars(codePoint))
            val alternatives = when {
                codePoint == '\''.code || codePoint == 0x2019 || codePoint == '-'.code -> listOf("")
                !german -> listOf(character)
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
