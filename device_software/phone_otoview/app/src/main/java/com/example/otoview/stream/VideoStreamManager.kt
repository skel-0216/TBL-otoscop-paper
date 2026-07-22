package com.example.otoview.stream

import android.content.Context
import android.util.Log
import android.view.Surface
import com.example.otoview.bench.BenchLog
import kotlinx.coroutines.*
import java.io.BufferedInputStream
import java.net.InetSocketAddress
import java.net.Socket

class VideoStreamManager(private val context: Context) {

    sealed class State {
        data object IDLE : State()
        data object CONNECTING : State()
        data class PLAYING(val fps: Int) : State()
        data class DISCONNECTED(val reason: String) : State()
        data class ERROR(val message: String) : State()
    }

    private var job: Job? = null

    fun start(
        host: String,
        port: Int,
        surface: Surface,
        autoRetry: Boolean,
        onVideoSize: ((Int, Int) -> Unit)? = null,
        onState: (State, String?) -> Unit
    ) {
        stop()
        job = CoroutineScope(Dispatchers.IO + SupervisorJob()).launch {
            // Guarantee all state callbacks on Main
            suspend fun postState(s: State, detail: String? = null) =
                withContext(Dispatchers.Main) { onState(s, detail) }

            // Guarantee onVideoSize on Main too (safe even if called from the decoder thread)
            val scope = this
            val safeOnVideoSize: ((Int, Int) -> Unit)? = onVideoSize?.let { cb ->
                { w: Int, h: Int ->
                    scope.launch(Dispatchers.Main) {
                        try { cb(w, h) } catch (t: Throwable) {
                            Log.w("VideoStreamManager", "onVideoSize callback failed: ${t.message}")
                        }
                    }
                }
            }

            val retryDelayMs = 2000L
            postState(State.CONNECTING, null)

            while (isActive) {
                var socket: Socket? = null
                // Stream startup latency: from connection attempt start to first rendered frame
                val attemptStartNs = BenchLog.nowNs()
                var firstFrameLogged = false
                try {
                    socket = BenchLog.time("stream_connect") {
                        Socket().apply {
                            tcpNoDelay = true
                            soTimeout = 0
                            try { receiveBufferSize = 1 shl 20 } catch (_: Throwable) {}
                            connect(InetSocketAddress(host, port), 4000)
                        }
                    }

                    val input = BufferedInputStream(socket.getInputStream(), 1 shl 20)
                    val annex = AnnexBReader(input)
                    val decoder = H264TcpDecoder(surface, safeOnVideoSize)
                    val framesAtStart = BenchLog.frames.totalFrames()

                    var fpsCounter = 0
                    var lastTs = System.currentTimeMillis()
                    postState(State.PLAYING(0), null)

                    while (isActive && !socket.isClosed) {
                        val nal = annex.nextNal() ?: break
                        decoder.feedNal(nal)
                        fpsCounter++

                        if (!firstFrameLogged && BenchLog.frames.totalFrames() > framesAtStart) {
                            firstFrameLogged = true
                            BenchLog.since("stream_first_frame", attemptStartNs)
                        }

                        val now = System.currentTimeMillis()
                        if (now - lastTs >= 1000) {
                            postState(State.PLAYING(fpsCounter), null)
                            fpsCounter = 0
                            lastTs = now
                        }
                    }

                    decoder.release()
                    BenchLog.event("stream_disconnected", null, mapOf("reason" to "stream ended"))
                    postState(State.DISCONNECTED("Stream ended"), null)
                } catch (t: Throwable) {
                    BenchLog.event("stream_error", null, mapOf("reason" to (t.message ?: "unknown")))
                    Log.e("VideoStreamManager", "stream error", t)
                    postState(State.ERROR(t.message ?: "stream error"), t.stackTraceToString())
                } finally {
                    try { socket?.close() } catch (_: Throwable) {}
                }

                if (!autoRetry) break
                delay(retryDelayMs)
                postState(State.CONNECTING, null)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
    }
}
