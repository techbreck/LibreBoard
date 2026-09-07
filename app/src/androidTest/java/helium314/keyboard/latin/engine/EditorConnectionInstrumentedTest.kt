// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.inputmethodservice.InputMethodService
import android.text.InputType
import android.view.inputmethod.BaseInputConnection
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.ExtractedTextRequest
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputConnectionWrapper
import android.widget.EditText
import androidx.test.core.app.ApplicationProvider
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.latin.RichInputConnection
import helium314.keyboard.latin.InputAttributes
import helium314.keyboard.latin.settings.Settings
import java.util.Locale
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Real Android Editable/InputConnection behavior; not a substitute for external-app IME tests. */
@RunWith(AndroidJUnit4::class)
class EditorConnectionInstrumentedTest {
    private class EditorService : InputMethodService() {
        lateinit var connection: InputConnection
        lateinit var info: EditorInfo
        override fun getCurrentInputConnection() = connection
        override fun getCurrentInputEditorInfo() = info
    }

    private class ObservedConnection(target: InputConnection) : InputConnectionWrapper(target, false) {
        var reads = 0
        override fun getTextBeforeCursor(length: Int, flags: Int): CharSequence? {
            reads++
            return super.getTextBeforeCursor(length, flags)
        }
        override fun getTextAfterCursor(length: Int, flags: Int): CharSequence? {
            reads++
            return super.getTextAfterCursor(length, flags)
        }
        override fun getSelectedText(flags: Int): CharSequence? {
            reads++
            return super.getSelectedText(flags)
        }
        override fun getExtractedText(request: ExtractedTextRequest, flags: Int) =
            super.getExtractedText(request, flags).also { reads++ }
    }

    private fun withEditor(text: String, action: (EditText, RichInputConnection, ObservedConnection) -> Unit) {
        InstrumentationRegistry.getInstrumentation().runOnMainSync {
            val editor = EditText(ApplicationProvider.getApplicationContext()).apply {
                inputType = InputType.TYPE_CLASS_TEXT
                setText(text)
                setSelection(text.length)
            }
            val info = EditorInfo()
            val observed = ObservedConnection(requireNotNull(editor.onCreateInputConnection(info)))
            Settings.init(editor.context)
            Settings.getInstance().loadSettings(
                editor.context, Locale.US, InputAttributes(info, false, editor.context.packageName),
            )
            val service = EditorService().apply {
                this.info = info
                connection = observed
            }
            val rich = RichInputConnection(service)
            rich.onStartInput(true)
            assertTrue(rich.resetCachesUponCursorMoveAndReturnSuccess(text.length, text.length, false))
            try {
                action(editor, rich, observed)
            } finally {
                observed.finishComposingText()
                observed.closeConnection()
            }
        }
    }

    @Test fun composingUpdatesReplaceTheSameRegionAndCommitOnce() = withEditor("hello ") { editor, rich, _ ->
        assertTrue(rich.setComposingText("w", 1))
        assertTrue(rich.setComposingText("wor", 1))
        assertTrue(rich.setComposingText("world", 1))
        assertEquals("hello world", editor.text.toString())
        assertEquals(6, BaseInputConnection.getComposingSpanStart(editor.text))
        assertEquals(11, BaseInputConnection.getComposingSpanEnd(editor.text))
        rich.commitText("world", 1)
        assertEquals("hello world", editor.text.toString())
        assertEquals(-1, BaseInputConnection.getComposingSpanStart(editor.text))
        assertEquals(11, editor.selectionStart)
        assertEquals(editor.selectionStart, rich.expectedSelectionStart)
    }

    @Test fun selectionReplacementAndCursorRefreshPreserveSurroundingText() = withEditor("hello world!") { editor, rich, _ ->
        assertTrue(rich.setSelection(6, 11))
        rich.commitText("friend", 1)
        assertEquals("hello friend!", editor.text.toString())
        assertEquals(12, editor.selectionStart)
        assertTrue(rich.resetCachesUponCursorMoveAndReturnSuccess(12, 12, true))
        assertEquals("hello friend", rich.getTextBeforeCursor(100, 0).toString())
        assertEquals("!", rich.getTextAfterCursor(100, 0).toString())
        assertEquals(12, rich.expectedSelectionStart)
    }

    @Test fun restrictedTransitionBlocksReadsThroughCursorChangesAndStillAllowsTyping() =
        withEditor("private prefix") { editor, rich, observed ->
            assertTrue(observed.reads > 0)
            rich.setContextReadsAllowed(false)
            observed.reads = 0
            assertTrue(rich.setSelection(7, 14))
            rich.commitText("value", 1)
            assertTrue(rich.resetCachesUponCursorMoveAndReturnSuccess(12, 12, true))
            assertEquals("", rich.getTextBeforeCursor(100, 0).toString())
            assertEquals("", rich.getTextAfterCursor(100, 0).toString())
            assertNull(rich.getSelectedText(0))
            assertEquals(0, observed.reads)
            assertEquals("privatevalue", editor.text.toString())
            assertEquals(-1, BaseInputConnection.getComposingSpanStart(editor.text))
            rich.setContextReadsAllowed(true)
            assertTrue(rich.resetCachesUponCursorMoveAndReturnSuccess(12, 12, false))
            assertTrue(observed.reads > 0)
            assertEquals("privatevalue", rich.getTextBeforeCursor(100, 0).toString())
        }
}
