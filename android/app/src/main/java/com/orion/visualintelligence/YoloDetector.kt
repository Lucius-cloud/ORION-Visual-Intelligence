package com.orion.visualintelligence

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

private const val TAG = "ORION_LINKEDIN"

data class Detection(
    val label: String,
    val confidence: Float,
    val box: FloatArray
)

class YoloDetector(context: Context) {

    companion object {
        private const val MODEL_NAME = "best.tflite"
        private const val LABELS_FILE = "labels.txt"

        private const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
    }

    private val interpreter: Interpreter
    private val labels: List<String>
    private val numClasses: Int
    private val numPredictions: Int
    private var gpuDelegate: GpuDelegate? = null

    init {
        val options = Interpreter.Options()
        val compat = CompatibilityList()

        if (compat.isDelegateSupportedOnThisDevice) {
            try {
                gpuDelegate = GpuDelegate(compat.bestOptionsForThisDevice)
                options.addDelegate(gpuDelegate!!)
                Log.i(TAG, "⚡ GPU delegate enabled")
            } catch (e: Exception) {
                options.setNumThreads(4)
                Log.w(TAG, "⚠️ GPU failed, using CPU")
            }
        } else {
            options.setNumThreads(4)
            Log.w(TAG, "⚠️ GPU not supported, using CPU")
        }

        interpreter = Interpreter(loadModel(context), options)
        labels = loadLabels(context)

        val inTensor: Tensor = interpreter.getInputTensor(0)
        val outTensor: Tensor = interpreter.getOutputTensor(0)
        val outShape = outTensor.shape() // [1, 4 + classes, 8400]

        Log.i(TAG, "📥 Input Shape  = ${inTensor.shape().contentToString()}")
        Log.i(TAG, "📤 Output Shape = ${outShape.contentToString()}")

        numPredictions = outShape[2]
        numClasses = outShape[1] - 4

        Log.i(TAG, "🧠 YOLOv8 Head → predictions=$numPredictions classes=$numClasses")
        Log.i(TAG, "🏷 Labels = $labels")

        require(numClasses == labels.size) {
            "❌ Model classes ($numClasses) != labels.txt (${labels.size})"
        }

        Log.i(TAG, "✅ YOLOv8 TFLite initialized")
    }

    private fun loadModel(context: Context): ByteBuffer {
        val afd = context.assets.openFd(MODEL_NAME)
        return FileInputStream(afd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
    }

    private fun loadLabels(context: Context): List<String> =
        BufferedReader(InputStreamReader(context.assets.open(LABELS_FILE)))
            .readLines()
            .filter { it.isNotBlank() }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
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

    fun detect(bitmap: Bitmap): List<Detection> {
        val input = preprocess(bitmap)
        val output = Array(1) { Array(4 + numClasses) { FloatArray(numPredictions) } }

        interpreter.run(input, output)

        val rawDetections = mutableListOf<Detection>()

        for (i in 0 until numPredictions) {
            val cx = output[0][0][i]
            val cy = output[0][1][i]
            val w = output[0][2][i]
            val h = output[0][3][i]

            var bestClass = -1
            var bestScore = 0f

            for (c in 0 until numClasses) {
                val score = output[0][4 + c][i]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            if (bestScore < CONF_THRESHOLD) continue

            val x1 = max(0f, cx - w / 2)
            val y1 = max(0f, cy - h / 2)
            val x2 = min(1f, cx + w / 2)
            val y2 = min(1f, cy + h / 2)

            rawDetections.add(
                Detection(
                    label = labels[bestClass],
                    confidence = bestScore,
                    box = floatArrayOf(x1, y1, x2, y2)
                )
            )
        }

        return nms(rawDetections)
    }

    private fun nms(list: List<Detection>): List<Detection> {
        val results = mutableListOf<Detection>()

        for ((_, group) in list.groupBy { it.label }) {
            val sorted = group.sortedByDescending { it.confidence }.toMutableList()

            while (sorted.isNotEmpty()) {
                val best = sorted.removeAt(0)
                results.add(best)

                val it = sorted.iterator()
                while (it.hasNext()) {
                    if (iou(best.box, it.next().box) > IOU_THRESHOLD) {
                        it.remove()
                    }
                }
            }
        }
        return results
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = max(a[0], b[0])
        val y1 = max(a[1], b[1])
        val x2 = min(a[2], b[2])
        val y2 = min(a[3], b[3])

        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])

        return inter / (areaA + areaB - inter + 1e-6f)
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        Log.i(TAG, "🧹 YOLO resources released")
    }
}
