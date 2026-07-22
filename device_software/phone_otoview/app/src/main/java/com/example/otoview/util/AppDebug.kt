package com.example.otoview.util

import android.content.Context
import android.content.pm.ApplicationInfo

object AppDebug {
    /** Returns true if the app is debuggable (android:debuggable or debug build). */
    fun isDebuggable(context: Context): Boolean {
        return (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }
}
