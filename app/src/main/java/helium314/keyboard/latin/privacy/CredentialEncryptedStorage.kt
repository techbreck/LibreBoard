// SPDX-License-Identifier: GPL-3.0-only
package helium314.keyboard.latin.privacy

import android.content.Context
import android.os.UserManager

object CredentialEncryptedStorage {
    /** Returns null before first unlock; callers must degrade without opening private state. */
    fun contextOrNull(context: Context): Context? {
        val userManager = context.getSystemService(UserManager::class.java) ?: return null
        if (!userManager.isUserUnlocked) return null
        // The application default is credential-protected. Static Direct-Boot state explicitly
        // opts into createDeviceProtectedStorageContext() through DeviceProtectedUtils.
        return context.applicationContext
    }
}
