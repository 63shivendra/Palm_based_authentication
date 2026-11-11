package com.palmpay.matcher

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View

/**
 * OverlayView: Renders real-time visual feedback:
 * - Center palm guide oval
 * - Hand skeleton and 21 landmark points
 * - Match status ring (Cyan = tracking, Green = Verified Match, Red = Rejected)
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var landmarks: List<PointF>? = null
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var matchStatus: MatchStatus = MatchStatus.SEARCHING

    enum class MatchStatus {
        SEARCHING,
        TRACKING,
        MATCHED,
        NOT_MATCHED
    }

    private val guidePaint = Paint().apply {
        color = Color.parseColor("#4400E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val pointPaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val linePaint = Paint().apply {
        color = Color.parseColor("#8800E5FF")
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }

    private val statusRingPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 12f
        isAntiAlias = true
    }

    fun setResults(landmarks: List<PointF>?, imgW: Int, imgH: Int, status: MatchStatus) {
        this.landmarks = landmarks
        this.imageWidth = imgW
        this.imageHeight = imgH
        this.matchStatus = status
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        // 1. Draw Central Target Guide Oval
        val cx = w / 2f
        val cy = h / 2.2f
        val radiusX = w * 0.32f
        val radiusY = h * 0.22f

        when (matchStatus) {
            MatchStatus.MATCHED -> statusRingPaint.color = Color.parseColor("#00E676") // Green
            MatchStatus.NOT_MATCHED -> statusRingPaint.color = Color.parseColor("#FF1744") // Red
            MatchStatus.TRACKING -> statusRingPaint.color = Color.parseColor("#00E5FF") // Cyan
            MatchStatus.SEARCHING -> statusRingPaint.color = Color.parseColor("#55FFFFFF") // White alpha
        }
        canvas.drawOval(cx - radiusX, cy - radiusY, cx + radiusX, cy + radiusY, statusRingPaint)

        // 2. Draw Hand Landmarks if present
        landmarks?.let { pts ->
            val scaleX = w / imageWidth.toFloat()
            val scaleY = h / imageHeight.toFloat()

            // Draw skeleton connections
            drawHandLines(canvas, pts, scaleX, scaleY)

            // Draw points
            for (p in pts) {
                canvas.drawCircle(p.x * scaleX, p.y * scaleY, 8f, pointPaint)
            }
        }
    }

    private fun drawHandLines(canvas: Canvas, pts: List<PointF>, sx: Float, sy: Float) {
        val connections = arrayOf(
            Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 4),        // Thumb
            Pair(0, 5), Pair(5, 6), Pair(6, 7), Pair(7, 8),        // Index
            Pair(5, 9), Pair(9, 10), Pair(10, 11), Pair(11, 12),   // Middle
            Pair(9, 13), Pair(13, 14), Pair(14, 15), Pair(15, 16), // Ring
            Pair(13, 17), Pair(17, 18), Pair(18, 19), Pair(19, 20),// Pinky
            Pair(0, 17)                                            // Palm base
        )

        for ((i, j) in connections) {
            if (i < pts.size && j < pts.size) {
                canvas.drawLine(
                    pts[i].x * sx, pts[i].y * sy,
                    pts[j].x * sx, pts[j].y * sy,
                    linePaint
                )
            }
        }
    }
}

