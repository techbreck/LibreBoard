// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.DeadlineCircuitBreaker
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.SwipeDecodeResult
import helium314.keyboard.latin.engine.SwipeDecoder
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.ctc.CtcSwipeDecoder
import helium314.keyboard.latin.engine.geometric.SwipeLexicon
import helium314.keyboard.latin.engine.onnx.LoadedSwipeModel
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/** Hard-deadline process owner for the optional CTC model session. */
class LiveSwipeModelSlot(
    private val circuitBreaker: DeadlineCircuitBreaker = DeadlineCircuitBreaker(),
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "LibreBoardLiveSwipe").apply { isDaemon = true }
    },
) : SwipeDecoder, AutoCloseable {
    private val monitor = Any()
    private var installed: Entry? = null
    private var closed = false

    fun install(decoder: SwipeDecoder, owner: AutoCloseable) {
        var closeNow: AutoCloseable? = null
        synchronized(monitor) {
            check(!closed) { "Swipe model slot is closed" }
            installed?.takeIf { it.owner === owner }?.let {
                it.decoder = decoder
                circuitBreaker.reset()
                return
            }
            installed?.let { previous ->
                previous.retired = true
                if (previous.activeCalls == 0) closeNow = previous.owner
            }
            installed = Entry(decoder, owner)
            circuitBreaker.reset()
        }
        closeNow?.closeQuietly()
    }

    fun clear() {
        var closeNow: AutoCloseable? = null
        synchronized(monitor) {
            installed?.let { previous ->
                previous.retired = true
                if (previous.activeCalls == 0) closeNow = previous.owner
            }
            installed = null
            circuitBreaker.reset()
        }
        closeNow?.closeQuietly()
    }

    override fun decode(request: TypingRequest, deadline: Deadline): SwipeDecodeResult {
        if (!request.fieldPolicy.allowsSuggestions) return SwipeDecodeResult(EngineAvailability.DISABLED)
        if (circuitBreaker.isOpen()) return SwipeDecodeResult(EngineAvailability.CIRCUIT_OPEN)
        synchronized(monitor) {
            if (closed || installed == null) return SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        }
        val remaining = deadline.remainingMillis
        if (remaining <= 0) return SwipeDecodeResult(EngineAvailability.TIMEOUT)
        val future = executor.submit<SwipeDecodeResult> {
            val entry = acquire() ?: return@submit SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
            try {
                entry.decoder.decode(request, deadline)
            } finally {
                release(entry)
            }
        }
        return try {
            future.get(remaining, TimeUnit.MILLISECONDS).also { result ->
                if (result.availability == EngineAvailability.TIMEOUT) circuitBreaker.recordOverrun()
            }
        } catch (_: TimeoutException) {
            future.cancel(true)
            circuitBreaker.recordOverrun()
            SwipeDecodeResult(EngineAvailability.TIMEOUT)
        } catch (_: ExecutionException) {
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        } catch (_: InterruptedException) {
            future.cancel(true)
            Thread.currentThread().interrupt()
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        }
    }

    override fun close() {
        var closeNow: AutoCloseable? = null
        synchronized(monitor) {
            if (closed) return
            closed = true
            installed?.let { previous ->
                previous.retired = true
                if (previous.activeCalls == 0) closeNow = previous.owner
            }
            installed = null
        }
        executor.shutdownNow()
        closeNow?.closeQuietly()
    }

    private fun acquire(): Entry? = synchronized(monitor) {
        if (closed) return@synchronized null
        installed?.also { it.activeCalls++ }
    }

    private fun release(entry: Entry) {
        var closeNow: AutoCloseable? = null
        synchronized(monitor) {
            entry.activeCalls--
            check(entry.activeCalls >= 0)
            if (entry.retired && entry.activeCalls == 0) closeNow = entry.owner
        }
        closeNow?.closeQuietly()
    }

    private fun AutoCloseable.closeQuietly() = runCatching { close() }.getOrDefault(Unit)

    private class Entry(
        var decoder: SwipeDecoder,
        val owner: AutoCloseable,
        var activeCalls: Int = 0,
        var retired: Boolean = false,
    )
}

fun LiveSwipeModelSlot.install(model: LoadedSwipeModel, lexicon: SwipeLexicon) {
    install(CtcSwipeDecoder(model.inferenceSession, lexicon), model)
}
