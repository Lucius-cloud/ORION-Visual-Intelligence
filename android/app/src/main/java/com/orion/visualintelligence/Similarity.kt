package com.orion.visualintelligence

import kotlin.math.sqrt

object Similarity {

    // Normalize vector in-place
    fun normalize(v: FloatArray) {
        var sum = 0f
        for (x in v) sum += x * x

        val norm = sqrt(sum)
        if (norm == 0f) return

        for (i in v.indices) v[i] /= norm
    }

    // Cosine similarity assumes BOTH vectors are normalized
    fun cosine(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0f
        for (i in v1.indices) dot += v1[i] * v2[i]
        return dot
    }
}
