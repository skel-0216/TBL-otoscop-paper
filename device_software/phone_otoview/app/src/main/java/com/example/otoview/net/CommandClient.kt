package com.example.otoview.net

import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.Socket
import javax.net.SocketFactory

class CommandClient {
    private var socket: Socket? = null
    private var out: OutputStream? = null

    // Save last connection info (for reconnect on onDestroy)
    private var savedCm: ConnectivityManager? = null
    private var savedHost: String? = null
    private var savedPort: Int? = null

    // Internal IO scope for one-line call from onDestroy
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    val isConnected: Boolean
        get() = socket?.isConnected == true && socket?.isClosed == false

    /**
     * Create a socket with the given SocketFactory and connect
     */
    private fun createAndConnect(factory: SocketFactory, host: String, port: Int, timeoutMs: Int): Socket {
        val s = factory.createSocket() as Socket
        try { s.receiveBufferSize = 1 shl 20 } catch (_: Throwable) {}
        try { s.sendBufferSize = 1 shl 16 } catch (_: Throwable) {}
        s.tcpNoDelay = true
        s.keepAlive = true
        s.connect(InetSocketAddress(host, port), timeoutMs)
        return s
    }

    /**
     * Try to connect via one given network
     */
    private fun connectViaNetwork(network: Network, host: String, port: Int, timeoutMs: Int): Socket {
        val factory = network.socketFactory
        return createAndConnect(factory, host, port, timeoutMs)
    }

    /**
     * Iterate networks by priority and try to connect
     * - WIFI / ETHERNET / WIFI_AWARE / (fallback) activeNetwork
     * - On success, save last connection info (savedCm/host/port)
     */
    suspend fun connectSmart(
        cm: ConnectivityManager,
        host: String,
        port: Int,
        timeoutMs: Int = 8000
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            close()

            val candidates = orderedNetworks(cm)
            var lastError: Throwable? = null
            for (net in candidates) {
                try {
                    val s = connectViaNetwork(net, host, port, timeoutMs)
                    socket = s
                    out = s.getOutputStream()

                    // Save last connection info (for reconnect on onDestroy)
                    savedCm = cm
                    savedHost = host
                    savedPort = port
                    return@runCatching
                } catch (t: Throwable) {
                    lastError = t
                }
            }

            error(lastError?.message ?: "No network could connect")
        }
    }

    /**
     * Force connect on a single network (when a network is already selected)
     * - Connecting via this path may leave savedCm empty (if reconnect is needed, running the connectSmart path at least once is recommended).
     */
    suspend fun connectOnNetwork(
        network: Network,
        host: String,
        port: Int,
        timeoutMs: Int = 8000
    ): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            close()
            val s = connectViaNetwork(network, host, port, timeoutMs)
            socket = s
            out = s.getOutputStream()

            // record host/port, cm unknown
            savedHost = host
            savedPort = port
        }
    }

    private fun orderedNetworks(cm: ConnectivityManager): List<Network> {
        val nets = cm.allNetworks.toList()
        if (nets.isEmpty()) return emptyList()

        fun score(n: Network): Int {
            val nc = cm.getNetworkCapabilities(n) ?: return 0
            var s = 0
            if (nc.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) s += 10
            if (nc.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) s += 100
            if (nc.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)) s += 90
            if (nc.hasTransport(NetworkCapabilities.TRANSPORT_WIFI_AWARE)) s += 80
            if (nc.hasTransport(NetworkCapabilities.TRANSPORT_VPN)) s -= 20
            if (nc.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR)) s -= 50
            return s
        }

        val sorted = nets.sortedByDescending { score(it) }.toMutableList()
        cm.activeNetwork?.let { an ->
            if (sorted.remove(an)) sorted.add(0, an)
        }
        return sorted
    }

    private suspend fun sendRaw(cmd: String): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            val os = out ?: error("Not connected")
            os.write(cmd.toByteArray(Charsets.UTF_8))
            os.flush()
        }
    }

    suspend fun sendLight(on: Boolean): Result<Unit> =
        sendRaw(if (on) "command_light_on" else "command_light_off")

    /**
     * [IMPORTANT] Overridden as a dedicated method to always turn the LED off on app exit.
     * - Instead of the old "command_quit" send, always sends only LED OFF.
     * - If not connected, reconnect via last connection info (savedCm/host/port) then send.
     * - Closes the socket regardless of success/failure.
     * - Exceptions are wrapped in Result and do not disturb the caller.
     */
    suspend fun sendQuit(): Result<Unit> = withContext(Dispatchers.IO) {
        runCatching {
            if (!isConnected) {
                val cm = savedCm
                val host = savedHost
                val port = savedPort
                if (cm != null && host != null && port != null) {
                    connectSmart(cm, host, port).getOrThrow()
                } else {
                    // No reconnect info, so just attempt exit (close() below always runs)
                }
            }

            // Send LED OFF (attempt send if connected)
            if (isConnected) {
                sendLight(false).getOrThrow()
            }
        }.also {
            // Clean up connection regardless of success/failure
            close()
        }
    }

    /**
     * Async wrapper for one-line call from onDestroy.
     * Does not block the UI thread; internally calls sendQuit().
     */
    fun sendQuitOnDestroy() {
        ioScope.launch {
            // Ignore result so a failure does not disturb app exit
            sendQuit()
        }
    }

    fun close() {
        try { out?.flush() } catch (_: Throwable) {}
        try { out?.close() } catch (_: Throwable) {}
        try { socket?.close() } catch (_: Throwable) {}
        out = null
        socket = null
    }
}
