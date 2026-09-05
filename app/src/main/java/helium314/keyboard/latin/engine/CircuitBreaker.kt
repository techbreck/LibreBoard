// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

class DeadlineCircuitBreaker(
    private val maximumOverruns: Int = 3,
    private val windowMillis: Long = 60_000,
    private val clockMillis: () -> Long = { System.nanoTime() / 1_000_000L },
) {
    private val overruns = ArrayDeque<Long>()

    init {
        require(maximumOverruns > 0)
        require(windowMillis > 0)
    }

    @Synchronized
    fun isOpen(): Boolean {
        discardExpired()
        return overruns.size >= maximumOverruns
    }

    @Synchronized
    fun recordOverrun() {
        discardExpired()
        overruns.addLast(clockMillis())
    }

    @Synchronized
    fun reset() = overruns.clear()

    private fun discardExpired() {
        val cutoff = clockMillis() - windowMillis
        while (overruns.firstOrNull()?.let { it < cutoff } == true) overruns.removeFirst()
    }
}
