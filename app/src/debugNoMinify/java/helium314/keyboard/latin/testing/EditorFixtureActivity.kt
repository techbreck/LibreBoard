// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.testing

import android.app.Activity
import android.content.Context
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.inputmethod.InputMethodManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputConnectionWrapper
import android.widget.EditText

/** Visible editor fixture exists only in the unminified debug variant, never in a release APK. */
class EditorFixtureActivity : Activity() {
    lateinit var editor: TrackingEditText
        private set

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        editor = TrackingEditText(this).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            if (intent.getBooleanExtra("terminal", false)) privateImeOptions = "org.libreboard.terminal"
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
        val manager = getSystemService(InputMethodManager::class.java)
        // Focus can arrive before the restarted IME has a served input connection. An early
        // API 36 show request can latch requested-visible state without displaying a window.
        if (editor.hasWindowFocus() && manager.isActive(editor)) {
            manager.showSoftInput(editor, InputMethodManager.SHOW_IMPLICIT)
        }
    }

    class TrackingEditText(context: Context) : EditText(context) {
        var composingUpdates = 0
            private set

        override fun onCreateInputConnection(outAttrs: EditorInfo): InputConnection? {
            val connection = super.onCreateInputConnection(outAttrs) ?: return null
            return object : InputConnectionWrapper(connection, false) {
                override fun setComposingText(text: CharSequence?, newCursorPosition: Int): Boolean {
                    composingUpdates++
                    return super.setComposingText(text, newCursorPosition)
                }

                override fun setComposingRegion(start: Int, end: Int): Boolean {
                    composingUpdates++
                    return super.setComposingRegion(start, end)
                }
            }
        }
    }
}
