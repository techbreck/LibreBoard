// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.onnx

import helium314.keyboard.latin.engine.Deadline
import helium314.keyboard.latin.engine.EngineAvailability
import helium314.keyboard.latin.engine.context.ContextCandidateRescorer
import helium314.keyboard.latin.engine.context.ContextInferenceResult
import helium314.keyboard.latin.engine.context.ContextInferenceSession
import helium314.keyboard.latin.engine.context.ContextModelBatch
import helium314.keyboard.latin.engine.ctc.CtcInferenceResult
import helium314.keyboard.latin.engine.ctc.CtcInferenceSession
import helium314.keyboard.latin.engine.ctc.CtcSwipeDecoder
import helium314.keyboard.latin.engine.ctc.CtcSwipeFeatures

object LibreBoardOnnxContracts {
    val swipeCtc = OnnxModelContract(
        inputs = listOf(
            OnnxTensorSpec("path_coordinates", OnnxElementType.FLOAT32, listOf(1, 64, 2)),
            OnnxTensorSpec("key_centers", OnnxElementType.FLOAT32, listOf(1, 64, 2)),
            OnnxTensorSpec("key_mask", OnnxElementType.FLOAT32, listOf(1, 64)),
        ),
        outputs = listOf(
            OnnxTensorSpec("logits", OnnxElementType.FLOAT32, listOf(1, 32, 65)),
        ),
    )

    val contextEnDe = OnnxModelContract(
        inputs = listOf(
            OnnxTensorSpec("input_ids", OnnxElementType.INT64, listOf(-1, 32)),
            OnnxTensorSpec("attention_mask", OnnxElementType.INT64, listOf(-1, 32)),
            OnnxTensorSpec("candidate_mask", OnnxElementType.FLOAT32, listOf(-1, 32)),
            OnnxTensorSpec("field_class", OnnxElementType.INT64, listOf(-1)),
        ),
        outputs = listOf(
            OnnxTensorSpec("candidate_log_likelihood", OnnxElementType.FLOAT32, listOf(-1)),
        ),
    )
}

class OnnxCtcInferenceSession(
    private val runtime: OnnxRuntimeSession,
) : CtcInferenceSession, AutoCloseable {
    override fun infer(features: CtcSwipeFeatures, deadline: Deadline): CtcInferenceResult {
        val result = runtime.run(
            mapOf(
                "path_coordinates" to OnnxInputTensor.Float32(features.pathCoordinates, longArrayOf(1, 64, 2)),
                "key_centers" to OnnxInputTensor.Float32(features.keyCenters, longArrayOf(1, 64, 2)),
                "key_mask" to OnnxInputTensor.Float32(features.keyMask, longArrayOf(1, 64)),
            ),
            deadline,
        )
        if (result.availability != EngineAvailability.AVAILABLE) {
            return CtcInferenceResult(result.availability)
        }
        val logits = result.floatOutputs["logits"]
            ?: return CtcInferenceResult(EngineAvailability.INCOMPATIBLE)
        if (!logits.shape.contentEquals(longArrayOf(1, 32, 65)) ||
            logits.values.size != CtcSwipeDecoder.OUTPUT_FRAMES * CtcSwipeDecoder.OUTPUT_CLASSES
        ) return CtcInferenceResult(EngineAvailability.INCOMPATIBLE)
        return CtcInferenceResult(
            availability = EngineAvailability.AVAILABLE,
            logits = logits.values,
            frameCount = CtcSwipeDecoder.OUTPUT_FRAMES,
            classCount = CtcSwipeDecoder.OUTPUT_CLASSES,
        )
    }

    override fun close() = runtime.close()
}

class OnnxContextInferenceSession(
    private val runtime: OnnxRuntimeSession,
) : ContextInferenceSession, AutoCloseable {
    override fun infer(batch: ContextModelBatch, deadline: Deadline): ContextInferenceResult {
        if (batch.sequenceLength != ContextCandidateRescorer.SEQUENCE_LENGTH || batch.batchSize <= 0) {
            return ContextInferenceResult(EngineAvailability.INCOMPATIBLE)
        }
        val rows = batch.batchSize.toLong()
        val result = runtime.run(
            mapOf(
                "input_ids" to OnnxInputTensor.Int64(batch.inputIds, longArrayOf(rows, 32)),
                "attention_mask" to OnnxInputTensor.Int64(batch.attentionMask, longArrayOf(rows, 32)),
                "candidate_mask" to OnnxInputTensor.Float32(batch.candidateMask, longArrayOf(rows, 32)),
                "field_class" to OnnxInputTensor.Int64(batch.fieldClasses, longArrayOf(rows)),
            ),
            deadline,
        )
        if (result.availability != EngineAvailability.AVAILABLE) {
            return ContextInferenceResult(result.availability)
        }
        val scores = result.floatOutputs["candidate_log_likelihood"]
            ?: return ContextInferenceResult(EngineAvailability.INCOMPATIBLE)
        if (!scores.shape.contentEquals(longArrayOf(rows)) || scores.values.size != batch.batchSize) {
            return ContextInferenceResult(EngineAvailability.INCOMPATIBLE)
        }
        return ContextInferenceResult(EngineAvailability.AVAILABLE, scores.values)
    }

    override fun close() = runtime.close()
}
