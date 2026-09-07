// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine

import android.content.Intent
import android.os.SystemClock
import android.provider.Settings
import android.view.InputDevice
import android.view.MotionEvent
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.runner.AndroidJUnit4
import helium314.keyboard.keyboard.KeyboardSwitcher
import helium314.keyboard.latin.testing.EditorFixtureActivity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Opt-in test drives the actual IME window through platform-injected touch events. */
@RunWith(AndroidJUnit4::class)
class LiveImeInstrumentedTest {
    @Test fun onscreenTypingCommitsOnceAndReplacesTheSelectedWord() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        assumeTrue(InstrumentationRegistry.getArguments().getString("libreboardRequireLiveIme") == "true")
        val context = instrumentation.targetContext
        val selectedIme = Settings.Secure.getString(context.contentResolver, Settings.Secure.DEFAULT_INPUT_METHOD)
        assertTrue("select this debug package as the emulator IME before running", selectedIme?.startsWith(context.packageName + "/") == true)
        val intent = Intent(context, EditorFixtureActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        val activity = instrumentation.startActivitySync(intent) as EditorFixtureActivity
        fun onMain(block: () -> Boolean): Boolean {
            var value = false
            instrumentation.runOnMainSync { value = block() }
            return value
        }
        fun await(message: String, condition: () -> Boolean) {
            val end = SystemClock.uptimeMillis() + 30_000
            while (SystemClock.uptimeMillis() < end) {
                if (onMain(condition)) return
                SystemClock.sleep(25)
            }
            assertTrue(message, onMain(condition))
        }
        fun tap(code: Int) {
            var x = 0f
            var y = 0f
            instrumentation.runOnMainSync {
                val view = requireNotNull(KeyboardSwitcher.getInstance().mainKeyboardView)
                val key = requireNotNull(requireNotNull(view.keyboard).getKey(code)) { "live key is missing: $code" }
                val location = IntArray(2)
                view.getLocationOnScreen(location)
                x = location[0] + view.paddingLeft + key.hitBox.exactCenterX()
                y = location[1] + view.paddingTop + key.hitBox.exactCenterY()
            }
            val down = SystemClock.uptimeMillis()
            for (action in listOf(MotionEvent.ACTION_DOWN, MotionEvent.ACTION_UP)) {
                val event = MotionEvent.obtain(down, SystemClock.uptimeMillis(), action, x, y, 0).apply {
                    source = InputDevice.SOURCE_TOUCHSCREEN
                }
                try {
                    assertTrue("platform must inject the live key touch", instrumentation.uiAutomation.injectInputEvent(event, true))
                } finally {
                    event.recycle()
                }
                SystemClock.sleep(40)
            }
        }
        try {
            var nextShowRequest = 0L
            await("the selected IME must show its real keyboard") {
                // Instrumentation restarts its target process, including the selected IME. Retry
                // the platform show request after window focus while that service reconnects.
                if (activity.editor.hasWindowFocus() && SystemClock.uptimeMillis() >= nextShowRequest) {
                    activity.showKeyboard()
                    nextShowRequest = SystemClock.uptimeMillis() + 1_000
                }
                KeyboardSwitcher.getInstance().mainKeyboardView?.let {
                    it.isShown && it.isAttachedToWindow && it.width > 0 && it.height > 0 && it.keyboard != null
                } == true
            }
            // Input injection synchronizes surface animations after coordinates are sampled.
            // Wait for stable geometry first so an opening IME cannot move below the first touch.
            var previousBounds = ""
            var stableSince = SystemClock.uptimeMillis()
            await("keyboard geometry must settle before sampling touch coordinates") {
                val view = requireNotNull(KeyboardSwitcher.getInstance().mainKeyboardView)
                val location = IntArray(2)
                view.getLocationOnScreen(location)
                val bounds = "${location[0]},${location[1]},${view.width},${view.height}"
                if (bounds != previousBounds) {
                    previousBounds = bounds
                    stableSince = SystemClock.uptimeMillis()
                }
                SystemClock.uptimeMillis() - stableSince >= 400
            }
            "cat ".forEachIndexed { index, character ->
                tap(character.code)
                await("each live key must reach the real editor") {
                    activity.editor.text.toString() == "cat ".take(index + 1)
                }
            }
            await("one word and one space must reach the real editor") { activity.editor.text.toString() == "cat " }
            instrumentation.runOnMainSync { activity.editor.setSelection(0, 3) }
            instrumentation.waitForIdleSync()
            "dog".forEach { tap(it.code) }
            await("typing must replace the selected word and preserve the trailing space") { activity.editor.text.toString() == "dog " }
            instrumentation.runOnMainSync { assertEquals("dog ", activity.editor.text.toString()) }
        } finally {
            instrumentation.runOnMainSync { activity.finish() }
        }
    }
}
