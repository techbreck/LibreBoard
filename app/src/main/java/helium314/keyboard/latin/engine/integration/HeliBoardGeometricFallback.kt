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
        val path = toEnginePath(inputPointers)
        if (path.size < 2) return null
        val geometry = toEngineGeometry(keyboard)
        val trace = TraceKeySequence.decode(path, geometry)
        if (trace.isBlank()) return null

        val syntheticPointers = InputPointers(trace.length)
        trace.forEachIndexed { index, character ->
            val key = geometry.keyFor(character) ?: return@forEachIndexed
            syntheticPointers.addPointer(key.centerX.toInt(), key.centerY.toInt(), 0, index * 10)
        }
        return ComposedData(syntheticPointers, false, trace)
    }

    fun toEnginePath(inputPointers: InputPointers): List<TouchPoint> {
        val size = inputPointers.pointerSize.coerceAtMost(1024)
        val xs = inputPointers.xCoordinates
        val ys = inputPointers.yCoordinates
        val times = inputPointers.times
        return List(size) { index ->
            TouchPoint(xs[index].toFloat(), ys[index].toFloat(), times[index].toLong())
        }
    }

    fun toEngineGeometry(keyboard: Keyboard): KeyGeometry {
        val slots = keyboard.sortedKeys.mapNotNull { key ->
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
            width = keyboard.mOccupiedWidth.toFloat().coerceAtLeast(1f),
            height = keyboard.mOccupiedHeight.toFloat().coerceAtLeast(1f),
            keys = slots.take(64),
        )
    }
}
