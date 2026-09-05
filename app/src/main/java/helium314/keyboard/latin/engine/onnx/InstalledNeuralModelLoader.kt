// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.onnx

import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.ModelKind
import helium314.keyboard.latin.engine.ModelRegistry
import helium314.keyboard.latin.engine.NeuralRescorer
import helium314.keyboard.latin.engine.context.BpeContextTokenizer
import helium314.keyboard.latin.engine.context.ContextCandidateRescorer
import helium314.keyboard.latin.engine.ctc.CtcInferenceSession
import java.io.File

data class NeuralComponentOpenResult<T : AutoCloseable>(
    val availability: EngineAvailability,
    val component: T? = null,
)

class LoadedContextModel internal constructor(
    val rescorer: NeuralRescorer,
    private val session: OnnxContextInferenceSession,
) : AutoCloseable {
    override fun close() = session.close()
}

class LoadedSwipeModel internal constructor(
    val inferenceSession: CtcInferenceSession,
    private val session: OnnxCtcInferenceSession,
) : AutoCloseable {
    override fun close() = session.close()
}

/** Joins independently verified CE model slots to their fixed ORT tensor adapters. */
object InstalledNeuralModelLoader {
    fun openContext(registry: ModelRegistry): NeuralComponentOpenResult<LoadedContextModel> =
        openContext(registry.activeModel(), OnnxRuntimeSessionFactory::open)

    fun openSwipe(registry: ModelRegistry): NeuralComponentOpenResult<LoadedSwipeModel> =
        openSwipe(registry.activeModel(), OnnxRuntimeSessionFactory::open)

    internal fun openContext(
        active: ModelRegistry.ActiveModel?,
        openRuntime: (File, OnnxModelContract) -> OnnxSessionOpenResult,
    ): NeuralComponentOpenResult<LoadedContextModel> {
        if (active == null) return NeuralComponentOpenResult(EngineAvailability.UNAVAILABLE)
        if (active.manifest.modelKind != ModelKind.CONTEXT_RESCORER || active.tokenizer == null) {
            return NeuralComponentOpenResult(EngineAvailability.INCOMPATIBLE)
        }
        val tokenizer = runCatching { BpeContextTokenizer.fromJson(active.tokenizer.readBytes()) }
            .getOrElse { return NeuralComponentOpenResult(EngineAvailability.INCOMPATIBLE) }
        val opened = openRuntime(active.model, LibreBoardOnnxContracts.contextEnDe)
        val runtime = opened.session ?: return NeuralComponentOpenResult(opened.availability)
        val session = OnnxContextInferenceSession(runtime)
        return NeuralComponentOpenResult(
            EngineAvailability.AVAILABLE,
            LoadedContextModel(ContextCandidateRescorer(tokenizer, session), session),
        )
    }

    internal fun openSwipe(
        active: ModelRegistry.ActiveModel?,
        openRuntime: (File, OnnxModelContract) -> OnnxSessionOpenResult,
    ): NeuralComponentOpenResult<LoadedSwipeModel> {
        if (active == null) return NeuralComponentOpenResult(EngineAvailability.UNAVAILABLE)
        if (active.manifest.modelKind != ModelKind.SWIPE_CTC || active.tokenizer != null) {
            return NeuralComponentOpenResult(EngineAvailability.INCOMPATIBLE)
        }
        val opened = openRuntime(active.model, LibreBoardOnnxContracts.swipeCtc)
        val runtime = opened.session ?: return NeuralComponentOpenResult(opened.availability)
        val session = OnnxCtcInferenceSession(runtime)
        return NeuralComponentOpenResult(
            EngineAvailability.AVAILABLE,
            LoadedSwipeModel(session, session),
        )
    }
}
