package com.example.otoview.stream

import java.io.InputStream

/**
 * Returns the next NAL from an Annex-B H.264 bytestream (start code excluded)
 * - Internal buffer expanded to 2MB to safely hold large NALs.
 */
class AnnexBReader(private val input: InputStream) {
    // 2 MB buffer (was 32KB → greatly increased)
    private val buffer = ByteArray(1 shl 21)
    private var filled = 0
    private var pos = 0

    fun nextNal(): ByteArray? {
        var start = findStartCode() ?: return null
        var end: Int?
        do {
            end = findStartCode(from = start + 3)
            if (end == null) {
                if (!fillMore()) break
            }
        } while (end == null)

        val nalStart = skipStartCode(start)
        val nalEnd = (end ?: filled)
        val size = nalEnd - nalStart
        if (size <= 0) return null
        val out = ByteArray(size)
        System.arraycopy(buffer, nalStart, out, 0, size)
        pos = nalEnd
        compactIfNeeded()
        return out
    }

    private fun findStartCode(from: Int = pos): Int? {
        var i = from
        while (true) {
            if (i + 4 >= filled) {
                if (!fillMore()) return null
                if (i + 4 >= filled) return null
            }
            // 0x000001
            if (buffer[i].toInt() == 0 && buffer[i+1].toInt() == 0 && buffer[i+2].toInt() == 1) return i
            // 0x00000001
            if (buffer[i].toInt() == 0 && buffer[i+1].toInt() == 0 &&
                buffer[i+2].toInt() == 0 && buffer[i+3].toInt() == 1) return i
            i++
        }
    }

    private fun skipStartCode(idx: Int): Int =
        if (buffer[idx+2].toInt() == 1) idx + 3 else idx + 4

    private fun fillMore(): Boolean {
        if (filled == buffer.size) return false
        val read = input.read(buffer, filled, buffer.size - filled)
        if (read <= 0) return false
        filled += read
        return true
    }

    private fun compactIfNeeded() {
        if (pos > (buffer.size / 2)) {
            val remaining = filled - pos
            System.arraycopy(buffer, pos, buffer, 0, remaining)
            filled = remaining
            pos = 0
        }
    }
}
