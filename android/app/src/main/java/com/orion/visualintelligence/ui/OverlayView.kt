package com.orion.visualintelligence.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.orion.visualintelligence.model.DetectionOverlay

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val textPaint = Paint().apply {
        color = Color.GREEN
        textSize = 48f
        style = Paint.Style.FILL
    }

    private var detections: List<DetectionOverlay> = emptyList()

    var demoMode: Boolean = false

    fun setDetections(list: List<DetectionOverlay>) {
        detections = list
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (d in detections) {
            canvas.drawRect(d.rect, boxPaint)
            canvas.drawText(
                "${d.label} ${(d.confidence * 100).toInt()}%",
                d.rect.left,
                d.rect.top - 10,
                textPaint
            )
        }
    }
}
