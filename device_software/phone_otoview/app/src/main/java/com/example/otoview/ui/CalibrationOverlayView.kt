package com.example.otoview.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Calibration overlay.
 * - contentRect: the area where the video is actually drawn (inside the FIT letterbox). Adjustments are confined to this area.
 * - An adjustable square + inscribed circle + center cross.
 * - All coordinates/sizes are in view pixels. The caller (Activity) converts to source pixels relative to contentRect.
 */
class CalibrationOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private val density = resources.displayMetrics.density

    private val contentBorder = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = density * 1f
        color = 0x55FFFFFF.toInt()
    }
    private val squarePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = density * 2f
        color = 0xFF12C7D6.toInt() // cyan accent
    }
    private val circlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = density * 1.5f
        color = Color.WHITE
    }
    private val crossPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = density * 1f
        color = 0x88FFFFFF.toInt()
    }
    private val scrimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x66000000
    }

    private val content = RectF()
    private var cx = 0f
    private var cy = 0f
    private var side = 0f
    private var initialized = false

    /** Set the video display area (view px) + initialize the square on first call */
    fun setContentRect(l: Float, t: Float, r: Float, b: Float) {
        content.set(l, t, r, b)
        if (!initialized) {
            cx = content.centerX()
            cy = content.centerY()
            side = min(content.width(), content.height()) * 0.7f
            initialized = true
        }
        clamp()
        invalidate()
    }

    /** Directly set the initial square from the source→view mapping result */
    fun setSquare(centerX: Float, centerY: Float, sidePx: Float) {
        cx = centerX; cy = centerY; side = sidePx
        initialized = true
        clamp()
        invalidate()
    }

    fun moveBy(dx: Float, dy: Float) { cx += dx; cy += dy; clamp(); invalidate() }

    fun scaleBy(factor: Float) {
        val maxSide = if (content.width() > 0) min(content.width(), content.height()) else side
        side = (side * factor).coerceIn(16f * density, maxSide)
        clamp(); invalidate()
    }

    fun centerX() = cx
    fun centerY() = cy
    fun sidePx() = side
    fun contentRect() = RectF(content)

    private fun clamp() {
        if (content.width() <= 0f) return
        val half = side / 2f
        // Keep the square within the content area
        if (side > content.width()) side = content.width()
        if (side > content.height()) side = content.height()
        val h2 = side / 2f
        cx = cx.coerceIn(content.left + h2, content.right - h2)
        cy = cy.coerceIn(content.top + h2, content.bottom - h2)
    }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        if (!initialized || content.width() <= 0f) return

        val half = side / 2f
        val left = cx - half; val top = cy - half
        val right = cx + half; val bottom = cy + half

        // Darken slightly outside the video area (emphasize the letterbox)
        c.drawRect(0f, 0f, width.toFloat(), content.top, scrimPaint)
        c.drawRect(0f, content.bottom, width.toFloat(), height.toFloat(), scrimPaint)

        // Video area border
        c.drawRect(content, contentBorder)

        // Square + inscribed circle + cross
        c.drawRect(left, top, right, bottom, squarePaint)
        c.drawCircle(cx, cy, half, circlePaint)
        c.drawLine(cx - half, cy, cx + half, cy, crossPaint)
        c.drawLine(cx, cy - half, cx, cy + half, crossPaint)

        // Corner handles
        val hl = 14f * density
        for (px in floatArrayOf(left, right)) {
            for (py in floatArrayOf(top, bottom)) {
                c.drawCircle(px, py, 4f * density, squarePaint)
            }
        }
    }
}
