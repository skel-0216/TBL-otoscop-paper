package com.example.otoview.bench

import android.os.SystemClock
import org.json.JSONArray
import org.json.JSONObject

/**
 * Counts frames actually released to the display surface and records inter-frame intervals.
 *
 * This is deliberately separate from the event log: it runs once per decoded frame on the
 * decoder thread, so it allocates nothing and takes no locks.
 */
class FrameMeter {

    /** ~11 min of 30 fps recording; further frames are counted but their intervals dropped. */
    private val intervalsUs = LongArray(20_000)

    @Volatile private var written: Int = 0
    @Volatile private var total: Long = 0
    @Volatile private var lastNs: Long = 0
    @Volatile private var firstNs: Long = 0

    // sliding 1 s window, for the HUD only
    @Volatile private var windowStartNs: Long = 0
    @Volatile private var windowCount: Int = 0
    @Volatile private var fpsRecent: Double = 0.0

    fun reset() {
        written = 0
        total = 0
        lastNs = 0
        firstNs = 0
        windowStartNs = 0
        windowCount = 0
        fpsRecent = 0.0
    }

    /** Call immediately after releaseOutputBuffer(index, render = true). */
    fun onFrameRendered() {
        if (!BenchLog.enabled) return
        val now = SystemClock.elapsedRealtimeNanos()
        total++
        if (lastNs != 0L) {
            val i = written
            if (i < intervalsUs.size) {
                intervalsUs[i] = (now - lastNs) / 1000
                written = i + 1
            }
        } else {
            firstNs = now
        }
        lastNs = now

        if (windowStartNs == 0L) windowStartNs = now
        windowCount++
        val elapsed = now - windowStartNs
        if (elapsed >= 1_000_000_000L) {
            fpsRecent = windowCount * 1e9 / elapsed
            windowStartNs = now
            windowCount = 0
        }
    }

    fun fpsRecent(): Double = fpsRecent

    fun totalFrames(): Long = total

    /** Mean fps over the whole session, from first to last rendered frame. */
    fun fpsOverall(): Double {
        val span = lastNs - firstNs
        return if (span <= 0L || total < 2) 0.0 else (total - 1) * 1e9 / span
    }

    fun toJson(): JSONObject {
        val arr = JSONArray()
        val n = written
        for (i in 0 until n) arr.put(intervalsUs[i])
        return JSONObject().apply {
            put("total_frames", total)
            put("fps_overall", fpsOverall())
            put("interval_us", arr)
            put("intervals_truncated", total - 1 > n)
        }
    }
}
