// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import helium314.keyboard.latin.engine.ModelKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.math.BigInteger
import java.security.KeyPairGenerator
import java.security.spec.RSAKeyGenParameterSpec
import kotlin.test.assertFailsWith

class OfficialModelPolicyTest {
    @Test
    fun projectKeyRequiresStrongRsaWithTheFixedExponent() {
        val strong = KeyPairGenerator.getInstance("RSA").apply { initialize(3072) }.generateKeyPair()
        assertEquals(strong.public, OfficialModelPolicy.parsePublicKey(strong.public.encoded))

        val weak = KeyPairGenerator.getInstance("RSA").apply { initialize(2048) }.generateKeyPair()
        assertFailsWith<IllegalArgumentException> {
            OfficialModelPolicy.parsePublicKey(weak.public.encoded)
        }
        val wrongAlgorithm = KeyPairGenerator.getInstance("EC").apply { initialize(256) }.generateKeyPair()
        assertFailsWith<Exception> {
            OfficialModelPolicy.parsePublicKey(wrongAlgorithm.public.encoded)
        }
        val wrongExponent = KeyPairGenerator.getInstance("RSA").apply {
            initialize(RSAKeyGenParameterSpec(3072, BigInteger.valueOf(3)))
        }.generateKeyPair()
        assertFailsWith<IllegalArgumentException> {
            OfficialModelPolicy.parsePublicKey(wrongExponent.public.encoded)
        }
        assertEquals(
            strong.public,
            OfficialModelPolicy.readPublicKey(ByteArrayInputStream(strong.public.encoded)),
        )
        assertFailsWith<IllegalArgumentException> {
            OfficialModelPolicy.readPublicKey(ByteArrayInputStream(ByteArray(8 * 1024 + 1)))
        }
    }

    @Test
    fun modelKindsHaveIndependentExactReleaseCeilings() {
        val swipe = OfficialModelPolicy.limits(ModelKind.SWIPE_CTC, 7)
        val context = OfficialModelPolicy.limits(ModelKind.CONTEXT_RESCORER, 7)

        assertEquals(setOf(ModelKind.SWIPE_CTC), swipe.acceptedModelKinds)
        assertEquals(2_621_440L, swipe.maximumBytes)
        assertEquals(1_000_000L, swipe.maximumParameterCount)
        assertEquals(setOf(ModelKind.CONTEXT_RESCORER), context.acceptedModelKinds)
        assertEquals(25_165_824L, context.maximumBytes)
        assertEquals(40_000_000L, context.maximumParameterCount)
        assertEquals(7, context.appVersionCode)
        assertEquals(EXPECTED_OPERATORS, context.allowedOperators)
        assertEquals(EXPECTED_OPERATORS, swipe.allowedOperators)
    }

    @Test
    fun bundledSwipeActivatesOnlyWhenMissingOrTheAppVersionChanges() {
        assertTrue(OfficialModelPolicy.shouldActivateBundledSwipe(1, 1, hasActiveModel = false))
        assertTrue(OfficialModelPolicy.shouldActivateBundledSwipe(1, 2, hasActiveModel = true))
        assertFalse(OfficialModelPolicy.shouldActivateBundledSwipe(2, 2, hasActiveModel = true))
    }

    private companion object {
        val EXPECTED_OPERATORS = setOf(
            "Add", "And", "Cast", "Clip", "Concat", "Constant", "ConstantOfShape", "Div", "Equal",
            "Erf", "Expand", "Gather", "Gemm", "LayerNormalization", "LessOrEqual", "MatMul", "Mod",
            "Mul", "Not", "Pow", "ReduceL2", "ReduceMean", "ReduceSum", "Reshape", "Shape", "Sigmoid",
            "Slice", "Softmax", "Sqrt", "Squeeze", "Sub", "Transpose", "Unsqueeze", "Where",
            "com.microsoft::GatherBlockQuantized", "com.microsoft::MatMulNBits",
        )
    }
}
