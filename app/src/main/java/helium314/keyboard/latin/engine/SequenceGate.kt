// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import java.util.concurrent.atomic.AtomicLong

/** Rejects late asynchronous results after composition, editor, or privacy-state changes. */
class SequenceGate(initialSequence: Long = 0) {
    private val current = AtomicLong(initialSequence)

    fun next(): Long = current.incrementAndGet()
    fun invalidate(): Long = next()
    fun current(): Long = current.get()
    fun accepts(sequenceId: Long): Boolean = sequenceId == current.get()
}
