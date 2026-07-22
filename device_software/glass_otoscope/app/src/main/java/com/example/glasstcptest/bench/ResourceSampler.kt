package com.example.glasstcptest.bench

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.util.Log
import java.io.File

/**
 * Samples process memory, CPU time, battery and thermal state once per second while a
 * benchmark session is running. Each sample is appended to [BenchLog] as a "resource" event.
 */
class ResourceSampler(context: Context) {

    private val appContext = context.applicationContext
    private val batteryManager =
        appContext.getSystemService(Context.BATTERY_SERVICE) as? BatteryManager
    private val powerManager =
        appContext.getSystemService(Context.POWER_SERVICE) as? PowerManager
    private val cores = Runtime.getRuntime().availableProcessors()
    private val clockTicksPerSec = 100.0 // Android's USER_HZ; getconf is unavailable to apps

    private var thread: Thread? = null
    @Volatile private var running = false

    private var lastCpuTicks: Long = -1
    private var lastUptimeMs: Long = -1

    fun start() {
        if (!BenchLog.enabled || running) return
        running = true
        thread = Thread({
            while (running) {
                try {
                    sampleOnce()
                    Thread.sleep(1000)
                } catch (_: InterruptedException) {
                    break
                } catch (t: Throwable) {
                    Log.w("ResourceSampler", "sample failed: ${t.message}")
                }
            }
        }, "bench-resource").also {
            it.isDaemon = true
            it.start()
        }
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
        lastCpuTicks = -1
        lastUptimeMs = -1
    }

    private fun sampleOnce() {
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        val rt = Runtime.getRuntime()

        BenchLog.event(
            "resource", null, mapOf(
                "pss_kb" to mi.totalPss,
                "private_dirty_kb" to mi.totalPrivateDirty,
                "java_heap_kb" to ((rt.totalMemory() - rt.freeMemory()) / 1024).toInt(),
                "cpu_percent" to cpuPercent(),
                "battery_percent" to batteryPercent(),
                "battery_current_ua" to batteryCurrentUa(),
                "thermal_status" to thermalStatus()
            )
        )
    }

    /**
     * Process CPU utilisation since the previous sample, as a percentage of one core.
     * Divide by core count for a whole-device figure; the raw value is kept so that the
     * offline report can present both.
     */
    private fun cpuPercent(): Double? {
        val ticks = readSelfCpuTicks() ?: return null
        val nowMs = android.os.SystemClock.elapsedRealtime()
        val prevTicks = lastCpuTicks
        val prevMs = lastUptimeMs
        lastCpuTicks = ticks
        lastUptimeMs = nowMs
        if (prevTicks < 0 || nowMs <= prevMs) return null
        val cpuMs = (ticks - prevTicks) / clockTicksPerSec * 1000.0
        return cpuMs / (nowMs - prevMs) * 100.0
    }

    /** utime + stime from /proc/self/stat (fields 14 and 15, 1-based). */
    private fun readSelfCpuTicks(): Long? = try {
        val stat = File("/proc/self/stat").readText()
        // The second field is the executable name in parentheses and may contain spaces.
        val after = stat.substring(stat.lastIndexOf(')') + 2)
        val f = after.split(" ")
        f[11].toLong() + f[12].toLong()
    } catch (t: Throwable) {
        null
    }

    private fun batteryPercent(): Int? =
        batteryManager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
            ?.takeIf { it in 0..100 }

    /** Instantaneous current in microamperes. Not implemented on every device. */
    private fun batteryCurrentUa(): Int? =
        batteryManager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            ?.takeIf { it != Int.MIN_VALUE && it != 0 }

    /** PowerManager thermal status, API 29+. Null on older devices such as Glass EE2. */
    private fun thermalStatus(): Int? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) powerManager?.currentThermalStatus
        else null
}
