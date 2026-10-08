// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import android.content.Context
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process
import androidx.annotation.RequiresApi

/**
 * Android Dynamic Performance Framework hint for the swipe decoder thread. That thread idles
 * between swipes, so the scheduler wakes it on a little core and migrates it only partway through
 * the decode. Reporting each decode's wall time against a target lets the platform raise the
 * thread's minimum CPU capacity. Unsupported platforms get no boost.
 */
internal object AdpfSwipeDecodeBoost {
    /**
     * Target for one decode. It sits below the ~124 ms a decode needs on a mid core at the clock
     * the scheduler otherwise picks, so reports run over target and ask for more capacity. On a
     * GrapheneOS Pixel 9 Pro XL this cut fused_swipe p50 from 162 to 121 ms.
     */
    const val TARGET_NANOS = 100_000_000L

    /** A [LiveSwipeModelSlot.boostFactory]; it runs on the decoder thread and binds that tid. */
    fun factory(context: Context, targetNanos: Long): () -> SwipeDecodeBoost? {
        val applicationContext = context.applicationContext
        return {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) create(applicationContext, targetNanos) else null
        }
    }

    /** Platform hint update rate, or -1 when hint sessions are unsupported or unavailable. */
    fun preferredUpdateRateNanos(context: Context): Long =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            context.getSystemService(PerformanceHintManager::class.java)?.preferredUpdateRateNanos ?: -1L
        } else -1L

    @RequiresApi(Build.VERSION_CODES.S)
    private fun create(context: Context, targetNanos: Long): SwipeDecodeBoost? {
        val manager = context.getSystemService(PerformanceHintManager::class.java) ?: return null
        if (manager.preferredUpdateRateNanos < 0) return null
        val session = manager.createHintSession(intArrayOf(Process.myTid()), targetNanos) ?: return null
        return HintSessionBoost(session)
    }

    @RequiresApi(Build.VERSION_CODES.S)
    private class HintSessionBoost(private val session: PerformanceHintManager.Session) : SwipeDecodeBoost {
        override fun afterDecode(elapsedNanos: Long) = session.reportActualWorkDuration(elapsedNanos)

        override fun close() = session.close()
    }
}
