package com.example.otoview.bench

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Deployment benchmark recorder.
 *
 * Records timing events on the real user path (streaming, capture, inference) so that the
 * numbers reported in the paper come from normal use of the app, not from a synthetic driver.
 *
 * Disabled unless the running APK is debuggable, so a release build carries no measurement
 * overhead and needs no code removal before publishing. Note this project configures AGP so
 * that BuildConfig is not generated; FLAG_DEBUGGABLE is the available signal.
 */
object BenchLog {

    private const val TAG = "BenchLog"
    private const val MAX_EVENTS = 20000

    @Volatile var enabled: Boolean = false
        private set

    private val events = ArrayList<Event>(4096)
    private var sessionId: String = "-"
    private var sessionLabel: String = "-"
    private var sessionStartWallMs: Long = 0L
    private var sessionStartNs: Long = 0L
    private var deviceInfo: JSONObject = JSONObject()

    /**
     * Frames released by the decoder. MediaCodec drains in bursts, so these timings show
     * decode throughput, not what the clinician sees.
     */
    val frames = FrameMeter()

    /**
     * Frames actually made available to the view, from onSurfaceTextureUpdated. This is the
     * display-side series and the one to report as frame rate and jitter.
     */
    val display = FrameMeter()

    private class Event(
        val tNs: Long,
        val type: String,
        val durMs: Double?,
        val extra: Map<String, Any?>?
    )

    // ---- lifecycle ---------------------------------------------------------

    fun init(context: Context) {
        val debuggable =
            (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        enabled = debuggable
        deviceInfo = JSONObject().apply {
            put("manufacturer", Build.MANUFACTURER)
            put("model", Build.MODEL)
            put("device", Build.DEVICE)
            put("android_release", Build.VERSION.RELEASE)
            put("sdk_int", Build.VERSION.SDK_INT)
            put("hardware", Build.HARDWARE)
            put("abi", Build.SUPPORTED_ABIS.joinToString(","))
            put("cores", Runtime.getRuntime().availableProcessors())
            put("app_package", context.packageName)
        }
        Log.i(TAG, "BenchLog enabled=$enabled on ${Build.MANUFACTURER} ${Build.MODEL}")
    }

    /** Start a new measurement session. Any previously buffered events are dropped. */
    @Synchronized
    fun startSession(label: String) {
        if (!enabled) return
        synchronized(events) { events.clear() }
        frames.reset()
        display.reset()
        sessionStartWallMs = System.currentTimeMillis()
        sessionStartNs = SystemClock.elapsedRealtimeNanos()
        sessionLabel = label
        sessionId = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date(sessionStartWallMs))
        event("session_start", null, mapOf("label" to label))
    }

    fun hasSession(): Boolean = sessionStartNs != 0L

    // ---- recording ---------------------------------------------------------

    fun nowNs(): Long = SystemClock.elapsedRealtimeNanos()

    fun event(type: String, durMs: Double? = null, extra: Map<String, Any?>? = null) {
        if (!enabled) return
        synchronized(events) {
            if (events.size >= MAX_EVENTS) return
            events.add(Event(nowNs(), type, durMs, extra))
        }
    }

    /** Record the wall time of [block] under [type]. */
    inline fun <T> time(type: String, extra: Map<String, Any?>? = null, block: () -> T): T {
        if (!enabled) return block()
        val t0 = nowNs()
        try {
            return block()
        } finally {
            event(type, (nowNs() - t0) / 1e6, extra)
        }
    }

    /** Record a duration measured elsewhere, e.g. across a callback boundary. */
    fun since(type: String, startNs: Long, extra: Map<String, Any?>? = null) {
        if (!enabled) return
        event(type, (nowNs() - startNs) / 1e6, extra)
    }

    // ---- reporting ---------------------------------------------------------

    /** Durations in ms recorded under [type], in order. */
    fun durations(type: String): DoubleArray {
        synchronized(events) {
            return events.asSequence()
                .filter { it.type == type && it.durMs != null }
                .map { it.durMs!! }
                .toList()
                .toDoubleArray()
        }
    }

    fun count(type: String): Int {
        synchronized(events) { return events.count { it.type == type } }
    }

    /** One-line summary for the on-screen HUD. */
    fun hudLine(): String {
        if (!enabled || !hasSession()) return "bench off"
        val inf = Stats.of(durations("infer_forward"))
        val e2e = Stats.of(durations("capture_e2e"))
        return "bench[$sessionLabel] fps=${"%.1f".format(display.fpsRecent())} " +
                "n=${inf.n} fwd=${"%.0f".format(inf.median)}ms " +
                "e2e=${if (e2e.n == 0) "-" else "%.0f".format(e2e.median) + "ms"}"
    }

    /** Write the current session to external app storage. Returns the file, or null. */
    @Synchronized
    fun export(context: Context): File? {
        if (!enabled || !hasSession()) return null
        val root = File(context.getExternalFilesDir(null), "bench")
        if (!root.exists() && !root.mkdirs()) {
            Log.w(TAG, "cannot create $root")
            return null
        }
        val out = File(root, "bench_${sessionId}_$sessionLabel.json")
        val json = JSONObject().apply {
            put("schema", 1)
            put("session_id", sessionId)
            put("session_label", sessionLabel)
            put("session_start_wall_ms", sessionStartWallMs)
            put("device", deviceInfo)
            put("frames", frames.toJson())
            put("display", display.toJson())
            put("events", eventsToJson())
        }
        out.writeText(json.toString(2))
        Log.i(TAG, "exported ${out.absolutePath} (${out.length()} B)")
        return out
    }

    private fun eventsToJson(): JSONArray {
        val arr = JSONArray()
        synchronized(events) {
            for (e in events) {
                val o = JSONObject()
                o.put("t_ms", (e.tNs - sessionStartNs) / 1e6)
                o.put("type", e.type)
                if (e.durMs != null) o.put("dur_ms", e.durMs)
                e.extra?.forEach { (k, v) -> o.put(k, v ?: JSONObject.NULL) }
                arr.put(o)
            }
        }
        return arr
    }
}
