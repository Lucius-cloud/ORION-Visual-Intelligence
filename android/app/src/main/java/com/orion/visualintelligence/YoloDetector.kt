package com.orion.visualintelligence

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val label: String,
    val confidence: Float,
    val box: FloatArray // normalized [x1, y1, x2, y2]
)

class YoloDetector(context: Context) {

    companion object {
        private const val TAG = "YOLO"

        private const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.5f

        private const val MODEL_NAME = "yolov8-electronics-fp16.tflite"
        private const val LABELS_FILE = "labels.txt"

        private const val NUM_PREDICTIONS = 8400
    }

    private val interpreter: Interpreter
    private val labels: List<String>
    private val NUM_CHANNELS: Int

    // Keep delegate references alive
    private var gpuDelegate: GpuDelegate? = null
    private var nnApiDelegate: NnApiDelegate? = null

    init {
        val options = Interpreter.Options()
        var delegateUsed = "CPU"

        // ================= GPU DELEGATE =================
        try {
            gpuDelegate = GpuDelegate() // ✅ CORRECT for TFLite 2.14
            options.addDelegate(gpuDelegate!!)
            delegateUsed = "GPU"
            Log.i(TAG, "🚀 GPU delegate ENABLED")
        } catch (e: Exception) {
            Log.w(TAG, "⚠️ GPU delegate failed, trying NNAPI", e)
        }

        // ================= NNAPI FALLBACK =================
        if (delegateUsed != "GPU") {
            try {
                nnApiDelegate = NnApiDelegate()
                options.addDelegate(nnApiDelegate!!)
                delegateUsed = "NNAPI"
                Log.i(TAG, "⚡ NNAPI delegate ENABLED")
            } catch (e: Exception) {
                Log.w(TAG, "⚠️ NNAPI failed, falling back to CPU", e)
            }
        }

        // ================= CPU FALLBACK =================
        if (delegateUsed == "CPU") {
            options.setNumThreads(4)
            Log.i(TAG, "🧠 Using CPU (4 threads)")
        }

        interpreter = Interpreter(loadModel(context), options)
        labels = loadLabels(context)
        NUM_CHANNELS = 4 + labels.size

        // ================= DEBUG =================
        Log.i(TAG, "✅ YOLOv8 TFLite loaded")
        Log.i(TAG, "📦 Labels (${labels.size}): $labels")
        Log.i(TAG, "📐 Channels = $NUM_CHANNELS")
        Log.i(TAG, "⚙️ Delegate used = $delegateUsed")
        Log.i(
            TAG,
            "📐 Output shape = ${
                interpreter.getOutputTensor(0).shape().contentToString()
            }"
        )
        Log.i(TAG, "⚙️ Input type = ${interpreter.getInputTensor(0).dataType()}")
        Log.i(TAG, "⚙️ Output type = ${interpreter.getOutputTensor(0).dataType()}")
    }

    // ---------------- MODEL LOADING ----------------

    private fun loadModel(context: Context): ByteBuffer {
        val afd = context.assets.openFd(MODEL_NAME)
        val inputStream = FileInputStream(afd.fileDescriptor)
        val channel = inputStream.channel

        return channel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
    }

    // ---------------- LABEL LOADING ----------------

    private fun loadLabels(context: Context): List<String> =
        BufferedReader(
            InputStreamReader(context.assets.open(LABELS_FILE))
        ).readLines().filter { it.isNotBlank() }

    // ---------------- PREPROCESS ----------------

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )

        val buffer = ByteBuffer
            .allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (p in pixels) {
            buffer.putFloat(((p shr 16) and 0xFF) / 255f)
            buffer.putFloat(((p shr 8) and 0xFF) / 255f)
            buffer.putFloat((p and 0xFF) / 255f)
        }

        buffer.rewind()
        resized.recycle()
        return buffer
    }

    // ---------------- INFERENCE ----------------

    fun detect(bitmap: Bitmap): List<Detection> {

        val input = preprocess(bitmap)
        val output =
            Array(1) { Array(NUM_CHANNELS) { FloatArray(NUM_PREDICTIONS) } }

        interpreter.run(input, output)

        val detections = mutableListOf<Detection>()

        for (i in 0 until NUM_PREDICTIONS) {

            val cx = output[0][0][i]
            val cy = output[0][1][i]
            val w = output[0][2][i]
            val h = output[0][3][i]

            var bestScore = 0f
            var bestClass = -1

            for (c in 4 until NUM_CHANNELS) {
                val score = output[0][c][i]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c - 4
                }
            }

            if (bestScore < CONF_THRESHOLD || bestClass !in labels.indices) continue

            val x1 = (cx - w / 2f) / INPUT_SIZE
            val y1 = (cy - h / 2f) / INPUT_SIZE
            val x2 = (cx + w / 2f) / INPUT_SIZE
            val y2 = (cy + h / 2f) / INPUT_SIZE

            detections.add(
                Detection(
                    label = labels[bestClass],
                    confidence = bestScore,
                    box = floatArrayOf(
                        max(0f, x1),
                        max(0f, y1),
                        min(1f, x2),
                        min(1f, y2)
                    )
                )
            )
        }

        return detections
    }

    // ---------------- CLEANUP ----------------

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
        Log.i(TAG, "🧹 Interpreter & delegates closed")
    }
}
