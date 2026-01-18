package com.orion.visualintelligence

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.sqrt

data class ClassificationResult(
    val label: String,
    val confidence: Float,
    val embedding: FloatArray
)

class OrionClassifier(
    private val context: Context,
    private val embeddingStore: EmbeddingStore
) {

    private val interpreter: Interpreter

    init {
        interpreter = Interpreter(loadModelFile(context))
        val out = interpreter.getOutputTensor(0)
        Log.d("ORION", "Model output shape: ${out.shape().contentToString()}")
    }

    // ---------------- MODEL LOADING ----------------

    private fun loadModelFile(context: Context): ByteBuffer {
        val afd = context.assets.openFd("orion_fp16.tflite")
        val inputStream = FileInputStream(afd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
    }

    // ---------------- PREPROCESS ----------------

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        buffer.order(ByteOrder.nativeOrder())

        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val pixels = IntArray(224 * 224)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
            buffer.putFloat((pixel and 0xFF) / 255f)
        }

        buffer.rewind()
        return buffer
    }

    // ---------------- NORMALIZATION ----------------

    private fun l2Normalize(vec: FloatArray) {
        var sum = 0f
        for (v in vec) sum += v * v
        val norm = sqrt(sum)
        if (norm > 0f) {
            for (i in vec.indices) vec[i] /= norm
        }
    }

    // ---------------- CLASSIFICATION ----------------

    fun classify(bitmap: Bitmap): ClassificationResult {

        if (embeddingStore.embeddings.isEmpty()) {
            Log.e("ORION", "❌ No embeddings loaded")
            return ClassificationResult("unknown", 0f, FloatArray(128))
        }

        val input = preprocess(bitmap)
        val embeddingOutput = Array(1) { FloatArray(128) }
        interpreter.run(input, embeddingOutput)
        val embedding = embeddingOutput[0]

        // Normalize live vector
        l2Normalize(embedding)

        val (label, confidence) = findBestMatch(embedding)

        Log.d(
            "ORION",
            "🎯 FINAL → label=$label confidence=${"%.2f".format(confidence * 100)}%"
        )

        return ClassificationResult(label, confidence, embedding)
    }

    // ---------------- CLEAN LABELS (FIXED!) ----------------

    private fun cleanLabel(name: String): String {
        val n = name.substringBefore(".").lowercase()

        return when {
            n.startsWith("capacitor") -> "capacitor"
            n.startsWith("resistor") -> "resistor"
            n.startsWith("transistor") -> "transistor"
            n.startsWith("pcb_board") -> "pcb_board"
            n.startsWith("ic_chip") -> "ic_chip"
            else -> n.substringBefore("_")
        }
    }

    // ---------------- SIMILARITY SEARCH ----------------

    private fun findBestMatch(
        embedding: FloatArray
    ): Pair<String, Float> {

        val scores = mutableListOf<Pair<String, Float>>()

        // Compute similarity with all stored embeddings
        for ((name, storedEmbedding) in embeddingStore.embeddings) {
            val score = Similarity.cosine(embedding, storedEmbedding)
            scores.add(cleanLabel(name) to score)
        }

        // Sort descending by similarity
        val topK = scores.sortedByDescending { it.second }.take(7)

        Log.d("ORION", "Top-K matches:")
        topK.forEach { Log.d("ORION", "  ${it.first} → ${"%.3f".format(it.second)}") }

        // Use BEST match only
        val best = topK.firstOrNull() ?: return "unknown" to 0f

        val threshold = 0.30f  // Realistic for all classes

        if (best.second < threshold) {
            Log.w("ORION", "⚠️ Rejected (score=${best.second})")
            return "unknown" to best.second
        }

        return best.first to best.second
    }
}
