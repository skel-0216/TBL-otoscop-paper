package com.example.otoview.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.example.otoview.R

/**
 * Background pattern view. Draws a very subtle pattern over the base color.
 * Choose the pattern with app:pbgType: dots / grid / diagonal / plus / crosshatch / none
 */
class PatternBackgroundView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private val density = resources.displayMetrics.density
    private fun dp(v: Float) = v * density

    private var baseColor = 0xFFEDF1F6.toInt()
    private var patternColor = 0xFFDFE6EF.toInt()
    private var spacing = dp(26f)
    private var type = TYPE_DOTS

    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = dp(1f)
    }

    init {
        if (attrs != null) {
            val a = context.obtainStyledAttributes(attrs, R.styleable.PatternBackgroundView)
            try {
                baseColor = a.getColor(R.styleable.PatternBackgroundView_pbgBaseColor, baseColor)
                patternColor = a.getColor(R.styleable.PatternBackgroundView_pbgPatternColor, patternColor)
                spacing = a.getDimension(R.styleable.PatternBackgroundView_pbgSpacing, spacing)
                type = a.getInt(R.styleable.PatternBackgroundView_pbgType, type)
            } finally { a.recycle() }
        }
    }

    /** Change the pattern from code */
    fun setPatternType(t: Int) { type = t; invalidate() }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        c.drawColor(baseColor)
        if (type == TYPE_NONE || spacing <= 0f) return

        val w = width.toFloat()
        val h = height.toFloat()
        fill.color = patternColor
        stroke.color = patternColor

        when (type) {
            TYPE_DOTS -> {
                val r = dp(1.7f)
                var y = spacing / 2f
                while (y < h) {
                    var x = spacing / 2f
                    while (x < w) { c.drawCircle(x, y, r, fill); x += spacing }
                    y += spacing
                }
            }
            TYPE_GRID -> {
                var x = 0f
                while (x <= w) { c.drawLine(x, 0f, x, h, stroke); x += spacing }
                var y = 0f
                while (y <= h) { c.drawLine(0f, y, w, y, stroke); y += spacing }
            }
            TYPE_DIAGONAL -> {
                val s = spacing
                var x = -h
                while (x < w) { c.drawLine(x, 0f, x + h, h, stroke); x += s }
            }
            TYPE_PLUS -> {
                val arm = dp(3.2f)
                var y = spacing / 2f
                while (y < h) {
                    var x = spacing / 2f
                    while (x < w) {
                        c.drawLine(x - arm, y, x + arm, y, stroke)
                        c.drawLine(x, y - arm, x, y + arm, stroke)
                        x += spacing
                    }
                    y += spacing
                }
            }
            TYPE_CROSSHATCH -> {
                val s = spacing
                var x = -h
                while (x < w) { c.drawLine(x, 0f, x + h, h, stroke); x += s }
                x = 0f
                while (x < w + h) { c.drawLine(x, 0f, x - h, h, stroke); x += s }
            }
        }
    }

    companion object {
        const val TYPE_NONE = 0
        const val TYPE_DOTS = 1
        const val TYPE_GRID = 2
        const val TYPE_DIAGONAL = 3
        const val TYPE_PLUS = 4
        const val TYPE_CROSSHATCH = 5
    }
}
