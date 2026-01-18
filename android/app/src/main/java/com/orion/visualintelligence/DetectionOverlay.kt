package com.orion.visualintelligence.model

import android.graphics.RectF

data class DetectionOverlay(
    val rect: RectF,
    val label: String,
    val confidence: Float
)
