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
import java.util.concurrent.CancellationException
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicReference

/**
 * Measurement hook: where one [LiveSwipeModelSlot.decode] spent its wall time. Production never
 * sets this; the Phase 0 harness uses it to separate slot overhead from decoder work. Times are
 * milliseconds. [runMs] is written on the decoder thread and may land after a timed-out decode
 * has already returned.
 */
class SwipeSlotTrace {
    /** Waiting for leftover inference from the previous swipe before submitting. */
    @Volatile var awaitIdleMs = 0.0
    /** From submit until the decoder thread started the task. */
    @Volatile var queueMs = 0.0
    /** Decoder run time on the decoder thread. */
    @Volatile var runMs = 0.0
    /** Whether a [SwipeDecodeBoost] was active for this decode. */
    @Volatile var boosted = false
}

/**
 * Optional platform performance hint for the decoder thread. It is created, used and closed only
 * on that thread, so an implementation may bind the thread's id; its failures never reach
 * decoding.
 */
interface SwipeDecodeBoost : AutoCloseable {
    /** Reports one decode's wall time on the decoder thread. */
    fun afterDecode(elapsedNanos: Long)
}

/** Hard-deadline process owner for the optional CTC model session. */
class LiveSwipeModelSlot(
    private val circuitBreaker: DeadlineCircuitBreaker = DeadlineCircuitBreaker(),
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "LibreBoardLiveSwipe").apply { isDaemon = true }
    },
    private val partialGraceMillis: Long = PARTIAL_GRACE_MILLIS,
) : SwipeDecoder, AutoCloseable {
    private val monitor = Any()
    private var installed: Entry? = null
    private var closed = false
    private val inFlight = AtomicReference<Future<SwipeDecodeResult>?>()

    /** Measurement hook; production never sets this. Read once at the start of each decode. */
    @Volatile var decodeTrace: SwipeSlotTrace? = null

    /**
     * Creates the [SwipeDecodeBoost] on the decoder thread at its next decode; null disables it.
     * Assigning a different factory closes the previous boost and creates a new one.
     */
    @Volatile var boostFactory: (() -> SwipeDecodeBoost?)? = null

    // Decoder thread only.
    private var boost: SwipeDecodeBoost? = null
    private var boostSource: (() -> SwipeDecodeBoost?)? = null

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

    fun hasDecoder(): Boolean = synchronized(monitor) { !closed && installed != null }

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
        val trace = decodeTrace
        val idleStart = System.nanoTime()
        if (!awaitIdle(deadline)) {
            circuitBreaker.recordOverrun()
            return SwipeDecodeResult(EngineAvailability.TIMEOUT)
        }
        val remaining = deadline.remainingMillis
        if (remaining <= 0) return SwipeDecodeResult(EngineAvailability.TIMEOUT)
        val submitted = System.nanoTime()
        trace?.awaitIdleMs = (submitted - idleStart) / 1_000_000.0
        val future = executor.submit<SwipeDecodeResult> {
            val started = System.nanoTime()
            trace?.queueMs = (started - submitted) / 1_000_000.0
            val activeBoost = currentBoost()
            trace?.boosted = activeBoost != null
            val entry = acquire() ?: return@submit SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
            try {
                entry.decoder.decode(request, deadline)
            } finally {
                release(entry)
                val elapsed = System.nanoTime() - started
                trace?.runMs = elapsed / 1_000_000.0
                activeBoost?.let { runCatching { it.afterDecode(elapsed) } }
            }
        }
        inFlight.set(future)
        return try {
            publish(future.get(remaining, TimeUnit.MILLISECONDS), future)
        } catch (_: TimeoutException) {
            // Self-bounded CTC may still return TIMEOUT partials a few ms late. Interrupting
            // now would turn that slate into UNAVAILABLE; wait the grace window first.
            awaitTimedOutResult(future)
        } catch (_: ExecutionException) {
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        } catch (_: CancellationException) {
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.TIMEOUT)
        } catch (_: InterruptedException) {
            future.cancel(true)
            clearInFlight(future)
            Thread.currentThread().interrupt()
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        }
    }

    /** Decoder thread only: follows [boostFactory], closing a replaced boost. */
    private fun currentBoost(): SwipeDecodeBoost? {
        val factory = boostFactory
        if (factory !== boostSource) {
            boost?.let { runCatching { it.close() } }
            boost = factory?.let { runCatching { it() }.getOrNull() }
            boostSource = factory
        }
        return boost
    }

    /**
     * The decoder thread is single-slot and ONNX ignores interrupt. Queueing another decode
     * behind leftover inference spends the next swipe's budget waiting and publishes nothing.
     */
    private fun awaitIdle(deadline: Deadline): Boolean {
        val previous = inFlight.get() ?: return true
        if (previous.isDone) {
            clearInFlight(previous)
            return true
        }
        val wait = deadline.remainingMillis
        if (wait <= 0) return false
        return try {
            previous.get(wait, TimeUnit.MILLISECONDS)
            clearInFlight(previous)
            true
        } catch (_: TimeoutException) {
            false
        } catch (_: ExecutionException) {
            clearInFlight(previous)
            true
        } catch (_: CancellationException) {
            clearInFlight(previous)
            true
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
            false
        }
    }

    private fun awaitTimedOutResult(future: Future<SwipeDecodeResult>): SwipeDecodeResult {
        completedResult(future)?.let { return it }
        val grace = partialGraceMillis.coerceAtLeast(0L)
        return try {
            publish(future.get(grace, TimeUnit.MILLISECONDS), future)
        } catch (_: TimeoutException) {
            completedResult(future)?.let { return it }
            future.cancel(true)
            circuitBreaker.recordOverrun()
            SwipeDecodeResult(EngineAvailability.TIMEOUT)
        } catch (_: ExecutionException) {
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        } catch (_: CancellationException) {
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.TIMEOUT)
        } catch (_: InterruptedException) {
            future.cancel(true)
            clearInFlight(future)
            Thread.currentThread().interrupt()
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        }
    }

    private fun completedResult(future: Future<SwipeDecodeResult>): SwipeDecodeResult? {
        if (!future.isDone) return null
        return try {
            publish(future.get(), future)
        } catch (_: ExecutionException) {
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        } catch (_: CancellationException) {
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.TIMEOUT)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
            clearInFlight(future)
            SwipeDecodeResult(EngineAvailability.UNAVAILABLE)
        }
    }

    private fun publish(result: SwipeDecodeResult, future: Future<SwipeDecodeResult>): SwipeDecodeResult {
        clearInFlight(future)
        if (result.availability == EngineAvailability.TIMEOUT && result.candidates.isEmpty()) {
            circuitBreaker.recordOverrun()
        }
        return result
    }

    private fun clearInFlight(future: Future<SwipeDecodeResult>) {
        inFlight.compareAndSet(future, null)
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

    private companion object {
        const val PARTIAL_GRACE_MILLIS = 15L
    }
}

fun LiveSwipeModelSlot.install(model: LoadedSwipeModel, lexicon: SwipeLexicon) {
    install(CtcSwipeDecoder(model.inferenceSession, lexicon), model)
}
