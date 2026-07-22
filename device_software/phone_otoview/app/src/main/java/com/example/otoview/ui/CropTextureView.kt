package com.example.otoview.ui

import android.content.Context
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.util.AttributeSet
import android.util.Log
import android.view.TextureView
import com.example.otoview.R
import com.example.otoview.debug.DebugToggles
import kotlin.math.max
import kotlin.math.min

class CropTextureView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : TextureView(context, attrs, defStyle) {

    // COVER = keep aspect ratio and fill (overflow cropped), FIT = keep aspect ratio and show all (letterbox),
    // STRETCH = stretch the crop rect independently in X/Y to fill the view exactly (aspect distortion allowed)
    enum class ScaleMode { COVER, FIT, STRETCH }

    private var srcW: Int = 1280
    private var srcH: Int = 720

    // Crop settings (source coordinate system)
    private var useFullFrame: Boolean = false
    private var cropCenterX: Float = 520f
    private var cropCenterY: Float = 360f
    private var cropSize: Int = 512           // square crop
    private var cropW: Int = 0                // rect crop
    private var cropH: Int = 0

    // Allow the crop center to go outside the source (area with no source = black padding)
    private var allowOutOfBounds: Boolean = false

    private var scaleMode: ScaleMode = ScaleMode.COVER

    // Pixel Aspect Ratio fix (PAR). Default 1:1
    private var parX: Float = 1f
    private var parY: Float = 1f

    // last applied transform (for gesture calc)
    private var lastUniformScale: Float = 1f  // uniform scale from applyTransform()
    private var lastEffRectW: Float = 1f      // rectW * parX
    private var lastEffRectH: Float = 1f      // rectH * parY
    private var lastScaleX: Float = 1f        // final X scale in view coords (mode-independent)
    private var lastScaleY: Float = 1f        // final Y scale in view coords (mode-independent)

    init {
        if (attrs != null) {
            val a = context.obtainStyledAttributes(attrs, R.styleable.CropTextureView)
            try {
                srcW = a.getInt(R.styleable.CropTextureView_srcWidth, srcW)
                srcH = a.getInt(R.styleable.CropTextureView_srcHeight, srcH)
                cropCenterX = a.getFloat(R.styleable.CropTextureView_cropCenterX, cropCenterX)
                cropCenterY = a.getFloat(R.styleable.CropTextureView_cropCenterY, cropCenterY)
                cropSize = a.getInt(R.styleable.CropTextureView_cropSize, cropSize)
                cropW = a.getInt(R.styleable.CropTextureView_cropWidth, cropW)
                cropH = a.getInt(R.styleable.CropTextureView_cropHeight, cropH)
                useFullFrame = a.getBoolean(R.styleable.CropTextureView_useFullFrame, useFullFrame)
                scaleMode = when (a.getInt(R.styleable.CropTextureView_cropScaleMode, 0)) {
                    1 -> ScaleMode.FIT
                    2 -> ScaleMode.STRETCH
                    else -> ScaleMode.COVER
                }
            } finally { a.recycle() }
        }

        setSurfaceTextureListener(object : SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) {
                st.setDefaultBufferSize(srcW, srcH)
                applyTransform()
            }
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {
                applyTransform()
            }
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean = true
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        })
    }

    // -------- public API --------

    fun setSourceSize(width: Int, height: Int) {
        if (width > 0 && height > 0) {
            if (srcW == width && srcH == height) return
            srcW = width; srcH = height
            surfaceTexture?.setDefaultBufferSize(srcW, srcH)
            applyTransform()
        }
    }

    /** full frame (no crop) */
    fun setFullFrame(enabled: Boolean) {
        if (useFullFrame == enabled) return
        useFullFrame = enabled
        applyTransform()
    }

    fun setScaleMode(mode: ScaleMode) {
        if (scaleMode == mode) return
        scaleMode = mode
        applyTransform()
    }

    /** Allow the crop center to go outside the source (area with no source shown as black padding) */
    fun setAllowOutOfBounds(enabled: Boolean) {
        if (allowOutOfBounds == enabled) return
        allowOutOfBounds = enabled
        applyTransform()
    }

    /** set square crop */
    fun setCrop(centerX: Float, centerY: Float, size: Int) {
        useFullFrame = false
        cropW = 0; cropH = 0
        if (size > 0) {
            cropCenterX = centerX; cropCenterY = centerY; cropSize = size
            clampCenterToBounds()
            applyTransform()
        }
    }

    /** set rect crop */
    fun setCropRect(centerX: Float, centerY: Float, width: Int, height: Int) {
        useFullFrame = false
        cropW = max(1, width); cropH = max(1, height)
        cropCenterX = centerX; cropCenterY = centerY
        clampCenterToBounds()
        applyTransform()
    }

    /** change size only in rect mode */
    fun setCropRectSize(width: Int, height: Int) {
        if (useFullFrame) return
        cropW = max(1, width); cropH = max(1, height)
        clampCenterToBounds()
        applyTransform()
    }

    /** Pixel Aspect Ratio fix: scale X/Y before uniform scaling */
    fun setPixelAspectFix(x: Float, y: Float = 1f) {
        // generous range (for debug): 0.25x ~ 4.00x
        val nx = x.coerceIn(0.25f, 4.00f)
        val ny = y.coerceIn(0.25f, 4.00f)
        if (parX == nx && parY == ny) return
        parX = nx; parY = ny
        applyTransform()
    }

    /** gesture: apply view-space delta as source-space delta (only when not FULL) */
    fun moveCropByViewDelta(dxView: Float, dyView: Float) {
        if (useFullFrame) return
        val dxSrc = dxView / lastScaleX
        val dySrc = dyView / lastScaleY

        cropCenterX += dxSrc
        cropCenterY += dySrc
        clampCenterToBounds()
        applyTransform()
    }

    /** gesture: change square/rect crop size scale factor */
    fun scaleCropBy(factor: Float) {
        if (useFullFrame) return
        val f = factor.coerceIn(0.5f, 2.0f) // avoid overshoot in a single pinch
        if (isRectCrop()) {
            val newW = (cropW * f).toInt().coerceIn(16, srcW)
            val newH = (cropH * f).toInt().coerceIn(16, srcH)
            cropW = newW; cropH = newH
        } else {
            val maxSide = min(srcW, srcH)
            cropSize = (cropSize * f).toInt().coerceIn(16, maxSide)
        }
        clampCenterToBounds()
        applyTransform()
    }

    /** query current FULL flag, center/size (for debug) */
    fun isFull(): Boolean = useFullFrame
    fun isRectCrop(): Boolean = (cropW > 0 && cropH > 0)
    fun getCropCenter(): Pair<Int, Int> = Pair(cropCenterX.toInt(), cropCenterY.toInt())
    fun getSquareSize(): Int = cropSize
    fun getRectSize(): Pair<Int, Int> = Pair(cropW, cropH)

    // -------- internal implementation --------

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        applyTransform()
    }

    fun applyTransform() {
        if (width == 0 || height == 0 || srcW <= 0 || srcH <= 0) {
            setTransform(null); return
        }

        // 1) crop rect (source coords)
        val rectW: Float
        val rectH: Float
        val left: Float
        val top: Float

        if (useFullFrame) {
            rectW = srcW.toFloat()
            rectH = srcH.toFloat()
            left = 0f
            top = 0f
        } else if (isRectCrop()) {
            val halfW = cropW / 2f
            val halfH = cropH / 2f
            // with allowOutOfBounds, let it extend past the source (area not covered by video = black padding)
            left = if (allowOutOfBounds) cropCenterX - halfW
                   else clamp(cropCenterX - halfW, 0f, max(0f, (srcW - cropW).toFloat()))
            top  = if (allowOutOfBounds) cropCenterY - halfH
                   else clamp(cropCenterY - halfH, 0f, max(0f, (srcH - cropH).toFloat()))
            rectW = cropW.toFloat()
            rectH = cropH.toFloat()
        } else {
            // for square crop too, with allowOutOfBounds do not clamp to source size
            val size = if (allowOutOfBounds) cropSize else min(cropSize, min(srcW, srcH))
            val half = size / 2f
            left = if (allowOutOfBounds) cropCenterX - half
                   else clamp(cropCenterX - half, 0f, max(0f, (srcW - size).toFloat()))
            top  = if (allowOutOfBounds) cropCenterY - half
                   else clamp(cropCenterY - half, 0f, max(0f, (srcH - size).toFloat()))
            rectW = size.toFloat()
            rectH = size.toFloat()
        }

        // 2) View mapping
        val dstW = width.toFloat()
        val dstH = height.toFloat()
        val m = Matrix()
        m.reset()

        if (scaleMode == ScaleMode.STRETCH) {
            // stretch the crop rect independently in X/Y to fill the view exactly (par ignored)
            val sX = dstW / rectW
            val sY = dstH / rectH
            lastScaleX = sX
            lastScaleY = sY
            lastUniformScale = 1f
            lastEffRectW = rectW
            lastEffRectH = rectH
            m.setTranslate(-left, -top)
            m.postScale(sX, sY)
            // the crop rect maps exactly onto [0,dstW]x[0,dstH], so no centering translation needed
        } else {
            // COVER/FIT: PAR correction, then aspect-preserving uniform scale
            lastEffRectW = rectW * parX
            lastEffRectH = rectH * parY
            val sx = dstW / lastEffRectW
            val sy = dstH / lastEffRectH
            val uniform = if (scaleMode == ScaleMode.COVER) max(sx, sy) else min(sx, sy)
            lastUniformScale = uniform
            lastScaleX = parX * uniform
            lastScaleY = parY * uniform
            m.setTranslate(-left, -top)
            m.postScale(parX * uniform, parY * uniform)
            val tx = (dstW - lastEffRectW * uniform) / 2f
            val ty = (dstH - lastEffRectH * uniform) / 2f
            m.postTranslate(tx, ty)
        }

        // TextureView correction: in the identity state the buffer (src) is stretched non-uniformly
        // to "fill" the view size (X width/srcW, Y height/srcH), so pre-multiply (preScale) the inverse
        // of that fill stretch in front of the "source→view pixel" matrix built above. Without it, the
        // image stretches vertically (vertical ellipse).
        m.preScale(srcW.toFloat() / width.toFloat(), srcH.toFloat() / height.toFloat())

        setTransform(m)

        if (DebugToggles.viewVerbose) {
            Log.d("CropTextureView",
                "view=${width}x${height}, src=${srcW}x${srcH}, rect=${rectW}x${rectH}, " +
                        "mode=$scaleMode, scaleX=${"%.3f".format(lastScaleX)}, scaleY=${"%.3f".format(lastScaleY)}")
        }
    }

    private fun clampCenterToBounds() {
        if (useFullFrame || allowOutOfBounds) return
        val (halfW, halfH) = currentHalfWH()
        // if crop width/height exceeds the source (e.g. aspect expansion), pin the center to the source center
        cropCenterX = if (srcW <= halfW * 2f) srcW / 2f else cropCenterX.coerceIn(halfW, srcW - halfW)
        cropCenterY = if (srcH <= halfH * 2f) srcH / 2f else cropCenterY.coerceIn(halfH, srcH - halfH)
    }

    private fun currentHalfWH(): Pair<Float, Float> {
        return if (isRectCrop()) {
            Pair(cropW / 2f, cropH / 2f)
        } else {
            val s = min(cropSize, min(srcW, srcH))
            Pair(s / 2f, s / 2f)
        }
    }

    private fun clamp(v: Float, minV: Float, maxV: Float) = max(minV, min(v, maxV))
}
