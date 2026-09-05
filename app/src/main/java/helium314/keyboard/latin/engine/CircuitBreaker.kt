// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

class DeadlineCircuitBreaker(
    private val maximumOverruns: Int = 3,
    private val windowMillis: Long = 60_000,
    private val clockMillis: () -> Long = System::currentTimeMillis,
) {
    private val overruns = ArrayDeque<Long>()

    fun isOpen(): Boolean {
        discardExpired()
        return overruns.size >= maximumOverruns
    }

    fun recordOverrun() {
        discardExpired()
        overruns.addLast(clockMillis())
    }

    fun reset() = overruns.clear()

    private fun discardExpired() {
        val cutoff = clockMillis() - windowMillis
        while (overruns.firstOrNull()?.let { it < cutoff } == true) overruns.removeFirst()
    }
}
