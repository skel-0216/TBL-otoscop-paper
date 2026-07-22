package com.example.glasstcptest

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import androidx.annotation.WorkerThread
import com.example.glasstcptest.bench.BenchLog
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import kotlin.math.exp

data class ClassifyResult(
    val topIndex: Int,
    val topLabel: String,
    val topProb: Float,
    val probs: List<Float>,
    val labels: List<String>
)

class ModelProcessor(context: Context) {

    // Load the model once per process. Even if the activity is recreated, the 45MB module is not read again.
    private val module: Module = sharedModule(context)

    private val labels: List<String> = sharedLabels(context)

    fun labelsList(): List<String> = labels

    @WorkerThread
    fun classify(bitmap: Bitmap): FloatArray = analyze(bitmap).probs.toFloatArray()

    @WorkerThread
    fun analyze(bitmap: Bitmap): ClassifyResult {
        val index = inferCount++   // per process. Only index 0 is a cold inference that includes the model load.
        val meta = mapOf("index" to index, "cold" to (index == 0))
        val t0 = BenchLog.nowNs()

        val input = BenchLog.time("infer_preprocess", meta) {
            bitmapToCHWFloatBuffer(bitmap, 256, 256, mean = 0.5f, std = 0.5f)
        }
        val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, 256, 256))
        // The first call includes the model load. Separated by the cold flag in offline analysis.
        val out = BenchLog.time("infer_forward", meta) {
            module.forward(IValue.from(tensor)).toTensor()
        }
        val logits = out.dataAsFloatArray
        require(logits.size == labels.size) {
            "Model/labels size mismatch: ${logits.size} vs ${labels.size}. " +
                "Check assets/labels.txt (should have ${logits.size} lines)."
        }
        val probs = softmax(logits)
        val (idx, p) = probs.withIndex().maxByOrNull { it.value }!!.let { it.index to it.value }
        BenchLog.since("infer_total", t0, meta + mapOf("label" to labels.getOrElse(idx) { "$idx" }))
        return ClassifyResult(idx, labels.getOrElse(idx) { "$idx" }, p, probs.toList(), labels)
    }

    private fun softmax(x: FloatArray): FloatArray {
        val max = x.maxOrNull() ?: 0f
        val exps = FloatArray(x.size)
        var sum = 0.0
        for (i in x.indices) {
            val e = exp((x[i] - max).toDouble()).toFloat()
            exps[i] = e
            sum += e
        }
        val denom = sum.toFloat().coerceAtLeast(1e-12f)
        return FloatArray(x.size) { exps[it] / denom }
    }

    private fun bitmapToCHWFloatBuffer(
        src: Bitmap,
        dstW: Int,
        dstH: Int,
        mean: Float,
        std: Float
    ): FloatArray {
        val scaled = if (src.width != dstW || src.height != dstH) {
            Bitmap.createScaledBitmap(src, dstW, dstH, true)
        } else src

        val pixels = IntArray(dstW * dstH)
        scaled.getPixels(pixels, 0, dstW, 0, 0, dstW, dstH)

        val out = FloatArray(3 * dstW * dstH)
        val plane = dstW * dstH
        var p = 0
        for (y in 0 until dstH) {
            for (x in 0 until dstW) {
                val c = pixels[p++]
                val r = ((c shr 16) and 0xFF) / 255f
                val g = ((c shr 8) and 0xFF) / 255f
                val b = (c and 0xFF) / 255f
                out[y * dstW + x] = (r - mean) / std
                out[plane + y * dstW + x] = (g - mean) / std
                out[2 * plane + y * dstW + x] = (b - mean) / std
            }
        }
        if (scaled !== src) scaled.recycle()
        return out
    }

    companion object {

        @Volatile private var cachedModule: Module? = null
        @Volatile private var cachedLabels: List<String>? = null

        /** Per-process inference count. Only index 0 is cold. */
        @Volatile private var inferCount: Int = 0

        private fun sharedModule(context: Context): Module =
            cachedModule ?: synchronized(this) {
                cachedModule ?: loadModule(context.applicationContext).also { cachedModule = it }
            }

        private fun sharedLabels(context: Context): List<String> =
            cachedLabels ?: synchronized(this) {
                cachedLabels ?: (
                    loadLabels(context.applicationContext.assets, "labels.txt")
                        ?: listOf("NOR", "PER", "RET", "TYM")
                    ).also { cachedLabels = it }
            }

        private fun loadModule(context: Context): Module {
            // If already copied, assetFilePath returns immediately. Recorded separately from the first-run cost.
            val cached = File(context.filesDir, "model.ptl").let { it.exists() && it.length() > 0L }
            val path = BenchLog.time("model_asset_copy", mapOf("already_present" to cached)) {
                assetFilePath(context, "model.ptl")
            }
            BenchLog.event("model_file", null, mapOf("bytes" to File(path).length()))
            return BenchLog.time("model_load") { LiteModuleLoader.load(path) }
        }

        private fun loadLabels(am: AssetManager, name: String): List<String>? =
            try {
                am.open(name).use { ins ->
                    BufferedReader(InputStreamReader(ins)).readLines()
                        .map { it.trim() }.filter { it.isNotEmpty() }
                }
            } catch (_: Throwable) { null }

        private fun assetFilePath(context: Context, assetName: String): String {
            val outFile = File(context.filesDir, assetName)
            if (outFile.exists() && outFile.length() > 0L) return outFile.absolutePath
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(4 * 1024)
                    while (true) {
                        val r = input.read(buffer)
                        if (r == -1) break
                        output.write(buffer, 0, r)
                    }
                    output.flush()
                }
            }
            return outFile.absolutePath
        }
    }
}
