package com.example.glasstcptest

import java.io.IOException
import java.io.InputStream
import java.net.InetSocketAddress
import java.net.Socket

class TcpIpReader(cameraData: CameraData) {
    companion object {
//        const val IO_TIMEOUT = 5000
//        private const val CONNECT_TIMEOUT = 5000
        const val IO_TIMEOUT = 1000
        private const val CONNECT_TIMEOUT = 5000
    }

    private var socket: Socket? = null
    private var inputStream: InputStream? = null

    init {
        try {
            socket = getConnection(cameraData.address, cameraData.port, CONNECT_TIMEOUT)
            socket?.soTimeout = IO_TIMEOUT
            inputStream = socket?.getInputStream()
        } catch (ex: Exception) {
            ex.printStackTrace()
        }
    }

    fun read(buffer: ByteArray): Int {
        return try {
            inputStream?.read(buffer) ?: 0
        } catch (ex: IOException) {
            ex.printStackTrace()
            0
        }
    }

    fun isConnected(): Boolean {
        return socket != null && socket!!.isConnected
    }

    fun close() {
        inputStream?.close()
        socket?.close()
        inputStream = null
        socket = null
    }

    private fun getConnection(baseAddress: String, port: Int, timeout: Int): Socket? {
        var socket: Socket?
        try {
            socket = Socket()
            val socketAddress = InetSocketAddress(baseAddress, port)
            socket.connect(socketAddress, timeout)
        } catch (ex: Exception) {
            ex.printStackTrace()
            socket = null
        }
        return socket
    }
}
