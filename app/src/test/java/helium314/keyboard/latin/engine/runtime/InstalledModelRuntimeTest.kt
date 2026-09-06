// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.engine.runtime

import android.content.Context
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import helium314.keyboard.latin.engine.ModelKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class InstalledModelRuntimeTest {
    @Test
    fun lowRamDevicesRejectContextModels() {
        assertFalse(NeuralDevicePolicy.allowsModel(ModelKind.CONTEXT_RESCORER, isLowRamDevice = true))
        assertTrue(NeuralDevicePolicy.allowsModel(ModelKind.CONTEXT_RESCORER, isLowRamDevice = false))
        assertTrue(NeuralDevicePolicy.allowsModel(ModelKind.SWIPE_CTC, isLowRamDevice = true))
    }

    @Test
    fun coreOnlyBuildRejectsManualModelImportWithoutOpeningTheUri() {
        val completed = CountDownLatch(1)
        var result: SignedModelImportResult? = null

        InstalledModelRuntime.installFromUri(
            ApplicationProvider.getApplicationContext<Context>(),
            Uri.parse("content://unavailable.invalid/model.lbmodel"),
        ) {
            result = it
            completed.countDown()
        }

        assertTrue(completed.await(2, TimeUnit.SECONDS))
        assertEquals(SignedModelImportResult.Unavailable, result)
    }
}
