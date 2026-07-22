package com.example.glasstcptest

import android.content.Context
import android.graphics.Matrix
import android.util.AttributeSet
import android.view.TextureView
import kotlin.math.max
import kotlin.math.min

class StreamingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : TextureView(context, attrs, defStyle) {

    enum class ScaleMode { COVER, FIT, STRETCH }

    private var srcW = 1280
    private var srcH = 720
    private var scaleMode = ScaleMode.FIT
    private var parX = 1f
    private var parY = 1f

    fun setSourceSize(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        if (srcW == width && srcH == height) return
        srcW = width; srcH = height
        applyTransform()
    }

    fun setScaleMode(mode: ScaleMode) {
        if (scaleMode == mode) return
        scaleMode = mode
        applyTransform()
    }

    fun setPixelAspectFix(x: Float, y: Float = 1f) {
        val nx = x.coerceIn(0.25f, 4f)
        val ny = y.coerceIn(0.25f, 4f)
        if (parX == nx && parY == ny) return
        parX = nx; parY = ny
        applyTransform()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        applyTransform()
    }

    fun applyTransform() {
        if (width == 0 || height == 0 || srcW <= 0 || srcH <= 0) {
            setTransform(null); return
        }

        val rectW = srcW.toFloat()
        val rectH = srcH.toFloat()
        val dstW = width.toFloat()
        val dstH = height.toFloat()
        val m = Matrix()

        if (scaleMode == ScaleMode.STRETCH) {
            m.setScale(dstW / rectW, dstH / rectH)
        } else {
            val effW = rectW * parX
            val effH = rectH * parY
            val sx = dstW / effW
            val sy = dstH / effH
            val uniform = if (scaleMode == ScaleMode.COVER) max(sx, sy) else min(sx, sy)
            m.setScale(parX * uniform, parY * uniform)
            val tx = (dstW - effW * uniform) / 2f
            val ty = (dstH - effH * uniform) / 2f
            m.postTranslate(tx, ty)
        }

        m.preScale(srcW.toFloat() / dstW, srcH.toFloat() / dstH)
        setTransform(m)
    }
}
