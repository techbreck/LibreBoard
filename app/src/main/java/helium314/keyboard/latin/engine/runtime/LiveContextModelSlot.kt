// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.DeadlineCircuitBreaker
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.NeuralRescorer
import helium314.keyboard.latin.engine.NeuralScoreResult
import helium314.keyboard.latin.engine.TypingRequest
import helium314.keyboard.latin.engine.Candidate
import helium314.keyboard.latin.engine.onnx.LoadedContextModel
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

/**
 * Process-scoped owner for the one context-model session. Every call is hard-deadline bounded even
 * if native inference ignores interruption. Retired sessions close only after their active call
 * returns, so replacing a model can never close native state from underneath inference.
 */
class LiveContextModelSlot(
    private val circuitBreaker: DeadlineCircuitBreaker = DeadlineCircuitBreaker(),
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "LibreBoardLiveContext").apply { isDaemon = true }
    },
) : NeuralRescorer, AutoCloseable {
    private val monitor = Any()
    private var installed: Entry? = null
    private var closed = false

    fun install(rescorer: NeuralRescorer, owner: AutoCloseable) {
        var closeNow: AutoCloseable? = null
        synchronized(monitor) {
            check(!closed) { "Context model slot is closed" }
            if (installed?.owner === owner) return
            installed?.let { previous ->
                previous.retired = true
                if (previous.activeCalls == 0) closeNow = previous.owner
            }
            installed = Entry(rescorer, owner)
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

    override fun score(
        request: TypingRequest,
        candidates: List<Candidate>,
        deadline: Deadline,
    ): NeuralScoreResult {
        if (!request.fieldPolicy.allowsContextRead || !request.fieldPolicy.allowsSuggestions) {
            return NeuralScoreResult(EngineAvailability.DISABLED)
        }
        if (circuitBreaker.isOpen()) return NeuralScoreResult(EngineAvailability.CIRCUIT_OPEN)
        synchronized(monitor) {
            if (closed || installed == null) return NeuralScoreResult(EngineAvailability.UNAVAILABLE)
        }
        val remaining = deadline.remainingMillis
        if (remaining <= 0) return NeuralScoreResult(EngineAvailability.TIMEOUT)
        val future = executor.submit<NeuralScoreResult> {
            val entry = acquire() ?: return@submit NeuralScoreResult(EngineAvailability.UNAVAILABLE)
            try {
                entry.rescorer.score(request, candidates, deadline)
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
            NeuralScoreResult(EngineAvailability.TIMEOUT)
        } catch (_: ExecutionException) {
            NeuralScoreResult(EngineAvailability.UNAVAILABLE)
        } catch (_: InterruptedException) {
            future.cancel(true)
            Thread.currentThread().interrupt()
            NeuralScoreResult(EngineAvailability.UNAVAILABLE)
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
        val rescorer: NeuralRescorer,
        val owner: AutoCloseable,
        var activeCalls: Int = 0,
        var retired: Boolean = false,
    )
}

object LiveTypingEngine {
    val contextRescorer = LiveContextModelSlot()

    fun installContext(model: LoadedContextModel) {
        contextRescorer.install(model.rescorer, model)
    }

    fun clearContext() = contextRescorer.clear()
}
