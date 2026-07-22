package com.example.otoview.stream

import android.media.MediaCodec
import android.media.MediaCodec.BufferInfo
import android.media.MediaFormat
import android.util.Log
import android.view.Surface
import com.example.otoview.bench.BenchLog
import com.example.otoview.debug.DebugToggles
import java.nio.ByteBuffer
import kotlin.math.max

/**
 * Very small synchronous H.264 (Annex-B) decoder wrapper for TCP streaming.
 * - Lazily configures MediaCodec when SPS/PPS are observed.
 * - Drains output on each feed to avoid backpressure.
 * - Logs output format changes (display/crop/SAR) for aspect-ratio debugging.
 */
class H264TcpDecoder(
    private val surface: Surface,
    private val onSizeChanged: ((Int, Int) -> Unit)? = null,
    private val widthHint: Int = 1280,
    private val heightHint: Int = 720
) {
    private var codec: MediaCodec? = null
    private var configured = false

    // Parameter sets
    private var sps: ByteArray? = null
    private var pps: ByteArray? = null

    // PTS generator (monotonic-ish)
    private var ptsUs: Long = 0

    /**
     * Feed one NAL unit (Annex-B start code removed).
     */
    fun feedNal(nal: ByteArray) {
        if (nal.isEmpty()) return

        val nalType = nal[0].toInt() and 0x1F
        when (nalType) {
            7 -> { // SPS
                sps = nal
                if (DebugToggles.codecVerbose) Log.d(TAG, "SPS len=${nal.size}")
            }
            8 -> { // PPS
                pps = nal
                if (DebugToggles.codecVerbose) Log.d(TAG, "PPS len=${nal.size}")
            }
        }

        if (!configured && sps != null && pps != null) {
            configureAndStart()
        }
        if (!configured) return

        // queue normal access unit
        queueAccessUnit(withStartCode(nal), isConfig = false)
        drainOutput(maxFrames = 8)
    }

    /** Release all codec resources. */
    fun release() {
        try { codec?.stop() } catch (_: Throwable) {}
        try { codec?.release() } catch (_: Throwable) {}
        codec = null
        configured = false
        sps = null
        pps = null
        ptsUs = 0
    }

    // ---- internal ----------------------------------------------------------

    private fun configureAndStart() {
        val spsBytes = sps ?: return
        val ppsBytes = pps ?: return

        val format = MediaFormat.createVideoFormat(MIME, widthHint, heightHint)
        // Provide CSD with start codes
        format.setByteBuffer("csd-0", ByteBuffer.wrap(withStartCode(spsBytes)))
        format.setByteBuffer("csd-1", ByteBuffer.wrap(withStartCode(ppsBytes)))

        val c = MediaCodec.createDecoderByType(MIME)
        c.configure(format, surface, null, 0)
        c.start()
        codec = c
        configured = true
        ptsUs = 0

        // Some devices prefer receiving CODEC_CONFIG buffers too.
        queueAccessUnit(withStartCode(spsBytes), isConfig = true)
        queueAccessUnit(withStartCode(ppsBytes), isConfig = true)

        // Immediately drain to trigger INFO_OUTPUT_FORMAT_CHANGED
        drainOutput(maxFrames = 4)
        // Notify initial hint (actual size will come on format change)
        onSizeChanged?.invoke(widthHint, heightHint)
        if (DebugToggles.codecVerbose) Log.d(TAG, "Decoder configured. hint=${widthHint}x${heightHint}")
    }

    private fun queueAccessUnit(data: ByteArray, isConfig: Boolean) {
        val c = codec ?: return
        val inIndex = c.dequeueInputBuffer(0)
        if (inIndex < 0) return
        val ibuf: ByteBuffer = c.getInputBuffer(inIndex) ?: return
        ibuf.clear()
        ibuf.put(data)
        val flags = if (isConfig) MediaCodec.BUFFER_FLAG_CODEC_CONFIG else 0
        c.queueInputBuffer(
            inIndex,
            0,
            data.size,
            (ptsUs++ * 33_366L),  // ~29.97fps spacing
            flags
        )
    }

    private fun drainOutput(maxFrames: Int) {
        val c = codec ?: return
        val info = BufferInfo()
        var frames = 0
        while (frames < maxFrames) {
            val outIndex = c.dequeueOutputBuffer(info, 0)
            when {
                outIndex >= 0 -> {
                    // Render to surface
                    c.releaseOutputBuffer(outIndex, true)
                    // Counted here rather than per NAL: this is the frame the clinician sees.
                    BenchLog.frames.onFrameRendered()
                    frames++
                }
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val fmt = c.outputFormat
                    if (DebugToggles.codecVerbose) logOutputFormat(fmt)
                    // Size callback (use crop/display if present)
                    val (w, h) = outputSize(fmt)
                    onSizeChanged?.invoke(w, h)
                }
                outIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    break
                }
                outIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> {
                    // ignored for modern API
                }
            }
        }
    }

    private fun outputSize(fmt: MediaFormat): Pair<Int, Int> {
        // Prefer display or crop size if available
        val w = fmt.getInteger(MediaFormat.KEY_WIDTH)
        val h = fmt.getInteger(MediaFormat.KEY_HEIGHT)

        val dw = if (fmt.containsKey("display-width")) fmt.getInteger("display-width") else w
        val dh = if (fmt.containsKey("display-height")) fmt.getInteger("display-height") else h

        val cw = if (fmt.containsKey("crop-right") && fmt.containsKey("crop-left"))
            fmt.getInteger("crop-right") - fmt.getInteger("crop-left") + 1 else w
        val ch = if (fmt.containsKey("crop-bottom") && fmt.containsKey("crop-top"))
            fmt.getInteger("crop-bottom") - fmt.getInteger("crop-top") + 1 else h

        // choose the most "specific" info available
        val useW = max(1, if (fmt.containsKey("display-width")) dw else cw)
        val useH = max(1, if (fmt.containsKey("display-height")) dh else ch)
        return Pair(useW, useH)
    }

    private fun logOutputFormat(fmt: MediaFormat) {
        val keys = arrayOf(
            MediaFormat.KEY_WIDTH, MediaFormat.KEY_HEIGHT,
            "display-width", "display-height",
            "crop-left", "crop-top", "crop-right", "crop-bottom",
            "sar-width", "sar-height",
            "stride", "slice-height",
            "color-format", "rotation-degrees"
        )
        val sb = StringBuilder("OutputFormat: ")
        for (k in keys) {
            if (fmt.containsKey(k)) {
                val v: Any? = try { fmt.getInteger(k) }
                catch (_: Throwable) {
                    try { fmt.getFloat(k) }
                    catch (_: Throwable) {
                        try { fmt.getString(k) } catch (_: Throwable) { null }
                    }
                }
                sb.append("$k=$v ")
            }
        }
        Log.d(TAG, sb.toString())

        // Derived PAR (pixel aspect) hint
        val w = fmt.getInteger(MediaFormat.KEY_WIDTH)
        val h = fmt.getInteger(MediaFormat.KEY_HEIGHT)
        val dw = if (fmt.containsKey("display-width")) fmt.getInteger("display-width") else w
        val dh = if (fmt.containsKey("display-height")) fmt.getInteger("display-height") else h
        val cw = if (fmt.containsKey("crop-right") && fmt.containsKey("crop-left"))
            fmt.getInteger("crop-right") - fmt.getInteger("crop-left") + 1 else w
        val ch = if (fmt.containsKey("crop-bottom") && fmt.containsKey("crop-top"))
            fmt.getInteger("crop-bottom") - fmt.getInteger("crop-top") + 1 else h
        val sarW = if (fmt.containsKey("sar-width")) fmt.getInteger("sar-width") else 1
        val sarH = if (fmt.containsKey("sar-height")) fmt.getInteger("sar-height") else 1

        Log.d(TAG, "derived: width=${w} height=${h} crop=${cw}x${ch} display=${dw}x${dh} sar=${sarW}:${sarH}")
    }

    private fun withStartCode(payload: ByteArray): ByteArray {
        val out = ByteArray(4 + payload.size)
        START_CODE.copyInto(out, 0)
        payload.copyInto(out, 4)
        return out
    }

    companion object {
        private const val TAG = "H264TcpDecoder"
        private const val MIME = "video/avc"
        private val START_CODE = byteArrayOf(0x00, 0x00, 0x00, 0x01)
    }
}
