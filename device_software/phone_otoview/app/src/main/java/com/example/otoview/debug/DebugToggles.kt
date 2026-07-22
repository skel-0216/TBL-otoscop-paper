package com.example.otoview.debug

/**
 * Centralized runtime debug toggles for verbose logs.
 * - Call setDefaults(...) once from your app code (e.g., MainActivity).
 * - You can flip these at runtime (e.g., hidden dev menu).
 */
object DebugToggles {
    @Volatile var codecVerbose: Boolean = false
    @Volatile var viewVerbose: Boolean = false

    /** Initialize defaults at runtime. */
    fun setDefaults(isDebug: Boolean) {
        codecVerbose = isDebug
        viewVerbose  = isDebug
    }
}
