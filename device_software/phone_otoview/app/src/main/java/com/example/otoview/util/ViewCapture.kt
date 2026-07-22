package com.example.otoview.util

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.view.PixelCopy
import android.view.TextureView
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import androidx.core.graphics.createBitmap

object ViewCapture {

    /**
     * Captures the TextureView "exactly as shown on screen" (including aspect correction/ROI/letterbox).
     * Uses PixelCopy(window, rect, …) and falls back to getBitmap() on failure.
     *
     * @return Bitmap on success, null on failure
     */
    suspend fun captureTextureView(activity: Activity, tv: TextureView): Bitmap? =
        suspendCancellableCoroutine { cont ->
            if (tv.width == 0 || tv.height == 0) {
                cont.resume(null); return@suspendCancellableCoroutine
            }

            val outBmp = createBitmap(tv.width, tv.height)

            // compute the TextureView coordinates within the window
            val loc = IntArray(2)
            tv.getLocationInWindow(loc)
            val srcRect = Rect(loc[0], loc[1], loc[0] + tv.width, loc[1] + tv.height)

            PixelCopy.request(
                /* source = */ activity.window,
                /* srcRect  = */ srcRect,
                /* dest     = */ outBmp,
                /* listener = */ { result ->
                    if (result == PixelCopy.SUCCESS) {
                        cont.resume(outBmp)
                    } else {
                        // fallback: the transform may not be fully reflected, but kept for device compatibility
                        outBmp.recycle()
                        runCatching {
                            createBitmap(tv.width, tv.height).also {
                                tv.getBitmap(it)
                            }
                        }.onSuccess { cont.resume(it) }
                            .onFailure { cont.resume(null) }
                    }
                },
                /* listenerThread = */ Handler(Looper.getMainLooper())
            )
        }
}
