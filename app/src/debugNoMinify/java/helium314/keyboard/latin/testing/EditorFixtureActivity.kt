// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.testing

import android.app.Activity
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.inputmethod.InputMethodManager
import android.widget.EditText

/** Visible editor fixture exists only in the unminified debug variant, never in a release APK. */
class EditorFixtureActivity : Activity() {
    lateinit var editor: EditText
        private set

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        editor = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            gravity = Gravity.TOP or Gravity.START
            setTextIsSelectable(false)
        }
        setContentView(editor)
        editor.requestFocus()
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) editor.post { showKeyboard() }
    }

    fun showKeyboard() {
        getSystemService(InputMethodManager::class.java).showSoftInput(editor, InputMethodManager.SHOW_IMPLICIT)
    }
}
