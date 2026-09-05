// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.text.InputType
import android.view.inputmethod.EditorInfo
import helium314.keyboard.latin.InputAttributes
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class FieldPolicyTest {
    @Test
    fun passwordDisablesEveryDataBearingCapability() {
        val editor = EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        val policy = FieldPolicyResolver.resolve(editor)
        assertEquals(FieldPolicy.SENSITIVE, policy)
        assertFalse(policy.allowsContextRead)
        assertFalse(policy.allowsSuggestions)
        assertFalse(policy.allowsComposing)
        assertFalse(policy.allowsClipboardCapture)
        assertFalse(policy.allowsPersistence)
    }

    @Test
    fun noPersonalizedLearningAlsoSuppressesContextAndSuggestions() {
        val editor = EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT
            imeOptions = EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING
        }
        assertEquals(FieldPolicy.NO_LEARNING, FieldPolicyResolver.resolve(editor))
    }

    @Test
    fun noSuggestionsOverrideOnlyAffectsThatFlag() {
        val noSuggestions = EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        assertEquals(FieldPolicy.NO_SUGGESTIONS, FieldPolicyResolver.resolve(noSuggestions))
        assertEquals(FieldPolicy.NORMAL, FieldPolicyResolver.resolve(noSuggestions, allowNoSuggestionsOverride = true))

        val password = EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD or
                InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        assertEquals(FieldPolicy.SENSITIVE, FieldPolicyResolver.resolve(password, allowNoSuggestionsOverride = true))
    }

    @Test
    fun terminalUsesDirectCommitButStillAllowsKeyboardLocalPreviews() {
        val policy = FieldPolicyResolver.resolve(EditorInfo().apply { inputType = InputType.TYPE_NULL })
        assertEquals(FieldPolicy.TERMINAL, policy)
        assertTrue(policy.allowsSuggestions)
        assertFalse(policy.allowsComposing)
        assertFalse(policy.allowsAutoCorrection)
        assertFalse(policy.allowsPersistence)
    }

    @Test
    fun inputAttributesCannotOverrideRestrictedPolicies() {
        val noSuggestions = InputAttributes(EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }, false, "")
        assertEquals(FieldPolicy.NO_SUGGESTIONS, noSuggestions.mFieldPolicy)
        assertTrue(noSuggestions.mMayOverrideShowingSuggestions)

        val email = InputAttributes(EditorInfo().apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS
        }, false, "")
        assertEquals(FieldPolicy.EMAIL_URI, email.mFieldPolicy)
        assertFalse(email.mMayOverrideShowingSuggestions)
        assertFalse(email.mShouldShowSuggestions)

        val terminal = InputAttributes(EditorInfo().apply {
            inputType = InputType.TYPE_NULL
        }, false, "")
        assertEquals(FieldPolicy.TERMINAL, terminal.mFieldPolicy)
        assertTrue(terminal.mShouldShowSuggestions)
        assertFalse(terminal.mInputTypeShouldAutoCorrect)
    }
}
