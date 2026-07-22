package com.example.glasstcptest

class SpsReader(private val nal: ByteArray, private val length: Int) {

    companion object {
        private const val TAG = "SpsReader"
    }

    private var currentBit = 0

    fun skipBits(n: Int) {
        currentBit += n
    }

    fun readBit(): Int {
        val index = currentBit / 8
        val offset = currentBit % 8 + 1

        currentBit++
        return if (index < length) ((nal[index].toInt() ushr (8 - offset)) and 0x01) else 0
    }

    fun readBits(n: Int): Int {
        var bits = 0
        repeat(n) {
            val bit = readBit()
            bits = (bits shl 1) + bit
        }
        return bits
    }

    private fun readCode(signed: Boolean): Int {
        var zeros = 0
        while (readBit() == 0) {
            zeros++
        }

        var code = (1 shl zeros) - 1 + readBits(zeros)
        if (signed) {
            code = (code + 1) / 2 * if (code % 2 == 0) -1 else 1
        }

        return code
    }

    fun readExpGolombCode(): Int {
        return readCode(false)
    }

    fun readSignedExpGolombCode(): Int {
        return readCode(true)
    }

    fun isEnd(): Boolean {
        return (currentBit / 8) >= length
    }
}