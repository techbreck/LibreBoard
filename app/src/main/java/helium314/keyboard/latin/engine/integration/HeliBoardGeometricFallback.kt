// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.integration

import helium314.keyboard.keyboard.Keyboard
import helium314.keyboard.latin.common.ComposedData
import helium314.keyboard.latin.common.InputPointers
import helium314.keyboard.latin.engine.KeyGeometry
import helium314.keyboard.latin.engine.KeySlot
import helium314.keyboard.latin.engine.TouchPoint
import helium314.keyboard.latin.engine.geometric.TraceKeySequence

/** Bridges retained HeliBoard pointer collection to the LibreBoard data-only fallback decoder. */
object HeliBoardGeometricFallback {
    fun toTypingComposedData(inputPointers: InputPointers, keyboard: Keyboard): ComposedData? {
        val size = inputPointers.pointerSize
        if (size < 2) return null
        val geometry = keyboard.toEngineGeometry()
        val xs = inputPointers.xCoordinates
        val ys = inputPointers.yCoordinates
        val times = inputPointers.times
        val path = List(size) { index ->
            TouchPoint(xs[index].toFloat(), ys[index].toFloat(), times[index].toLong())
        }
        val trace = TraceKeySequence.decode(path, geometry)
        if (trace.isBlank()) return null

        val syntheticPointers = InputPointers(trace.length)
        trace.forEachIndexed { index, character ->
            val key = geometry.keyFor(character) ?: return@forEachIndexed
            syntheticPointers.addPointer(key.centerX.toInt(), key.centerY.toInt(), 0, index * 10)
        }
        return ComposedData(syntheticPointers, false, trace)
    }

    private fun Keyboard.toEngineGeometry(): KeyGeometry {
        val slots = sortedKeys.mapNotNull { key ->
            val code = key.code
            if (code <= 0 || !Character.isLetter(code)) return@mapNotNull null
            KeySlot(
                id = code,
                label = String(Character.toChars(code)),
                centerX = key.x + key.width / 2f,
                centerY = key.y + key.height / 2f,
                width = key.width.toFloat(),
                height = key.height.toFloat(),
            )
        }.distinctBy { it.id }
        return KeyGeometry(
            width = mOccupiedWidth.toFloat().coerceAtLeast(1f),
            height = mOccupiedHeight.toFloat().coerceAtLeast(1f),
            keys = slots.take(64),
        )
    }
}
