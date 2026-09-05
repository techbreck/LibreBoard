// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.text.InputType
import android.view.inputmethod.EditorInfo

/**
 * A single, auditable decision about which IME behaviours are safe for the current editor.
 * Callers must use these capabilities instead of repeating InputType checks.
 */
enum class FieldPolicy(
    val allowsContextRead: Boolean,
    val allowsSuggestions: Boolean,
    val allowsComposing: Boolean,
    val allowsAutoCorrection: Boolean,
    val allowsClipboardCapture: Boolean,
    val allowsPersistence: Boolean,
) {
    NORMAL(true, true, true, true, true, true),
    NO_LEARNING(false, false, true, false, false, false),
    NO_SUGGESTIONS(false, false, true, false, false, false),
    SENSITIVE(false, false, false, false, false, false),
    EMAIL_URI(false, false, true, false, false, false),
    TERMINAL(false, true, false, false, false, false),
}

object FieldPolicyResolver {
    fun resolve(editorInfo: EditorInfo?, allowNoSuggestionsOverride: Boolean = false): FieldPolicy {
        if (editorInfo == null) return FieldPolicy.SENSITIVE
        val inputType = editorInfo.inputType
        val inputClass = inputType and InputType.TYPE_MASK_CLASS
        val variation = inputType and InputType.TYPE_MASK_VARIATION

        if (isPassword(inputClass, variation)) return FieldPolicy.SENSITIVE
        if (isEmailOrUri(inputClass, variation)) return FieldPolicy.EMAIL_URI
        if (editorInfo.imeOptions and EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING != 0) {
            return FieldPolicy.NO_LEARNING
        }
        if (!allowNoSuggestionsOverride && inputType and InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS != 0) {
            return FieldPolicy.NO_SUGGESTIONS
        }
        if (inputType == InputType.TYPE_NULL || isTerminal(editorInfo)) return FieldPolicy.TERMINAL
        return FieldPolicy.NORMAL
    }

    private fun isPassword(inputClass: Int, variation: Int): Boolean =
        (inputClass == InputType.TYPE_CLASS_TEXT && variation in setOf(
            InputType.TYPE_TEXT_VARIATION_PASSWORD,
            InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
            InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD,
        )) || (inputClass == InputType.TYPE_CLASS_NUMBER && variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD)

    private fun isEmailOrUri(inputClass: Int, variation: Int): Boolean =
        inputClass == InputType.TYPE_CLASS_TEXT && variation in setOf(
            InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS,
            InputType.TYPE_TEXT_VARIATION_WEB_EMAIL_ADDRESS,
            InputType.TYPE_TEXT_VARIATION_URI,
        )

    private fun isTerminal(editorInfo: EditorInfo): Boolean {
        val privateOptions = editorInfo.privateImeOptions.orEmpty().lowercase()
        return "terminal" in privateOptions || "termux" in privateOptions
    }
}
