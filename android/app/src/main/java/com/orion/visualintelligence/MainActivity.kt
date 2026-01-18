package com.orion.visualintelligence

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.orion.visualintelligence.ui.OverlayView
import com.orion.visualintelligence.model.DetectionOverlay
import com.orion.visualintelligence.utils.ImageUtils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "ORION"

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var mlExecutor: ExecutorService

    @Volatile
    private var yoloDetector: YoloDetector? = null

    private lateinit var overlayView: OverlayView

    // ⏱️ Throttle inference → 1 FPS
    @Volatile
    private var lastInferenceTime = 0L

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startPipeline()
            else Log.e(TAG, "❌ Camera permission denied")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 🔹 Log device info (PROVES real phone)
        Log.i(
            "DEVICE",
            "Model=${android.os.Build.MODEL}, " +
                    "Hardware=${android.os.Build.HARDWARE}, " +
                    "SDK=${android.os.Build.VERSION.SDK_INT}"
        )

        cameraExecutor = Executors.newSingleThreadExecutor()
        mlExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startPipeline()
        } else {
            requestPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startPipeline() {
        mlExecutor.execute {
            Log.i(TAG, "🧠 Loading YOLO model…")
            yoloDetector = YoloDetector(applicationContext)

            runOnUiThread {
                startCamera()
            }
        }
    }

    private fun startCamera() {

        val previewView = PreviewView(this)
        overlayView = OverlayView(this).apply {
            demoMode = true // portfolio friendly
        }

        val container = FrameLayout(this).apply {
            addView(previewView)
            addView(overlayView)
        }

        setContentView(container)

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->

                // ⏱️ Throttle to ~1 FPS
                val now = System.currentTimeMillis()
                if (now - lastInferenceTime < 1000) {
                    imageProxy.close()
                    return@setAnalyzer
                }
                lastInferenceTime = now

                val bitmap = ImageUtils.toBitmap(imageProxy)

                mlExecutor.execute {
                    try {
                        val yolo = yoloDetector ?: return@execute

                        // ================= REAL PHONE PERFORMANCE =================
                        val startTime = SystemClock.elapsedRealtime()
                        val detections = yolo.detect(bitmap)
                        val endTime = SystemClock.elapsedRealtime()

                        val inferenceMs = endTime - startTime
                        val fps = if (inferenceMs > 0) 1000f / inferenceMs else 0f

                        Log.i(
                            "PERF",
                            "Inference = ${inferenceMs} ms | FPS ≈ ${"%.2f".format(fps)}"
                        )
                        // ===========================================================

                        if (overlayView.width == 0 || overlayView.height == 0) return@execute

                        val overlays = detections.map {
                            DetectionOverlay(
                                rect = RectF(
                                    it.box[0] * overlayView.width,
                                    it.box[1] * overlayView.height,
                                    it.box[2] * overlayView.width,
                                    it.box[3] * overlayView.height
                                ),
                                label = it.label,
                                confidence = it.confidence
                            )
                        }

                        runOnUiThread {
                            overlayView.setDetections(overlays)
                        }

                    } catch (e: Exception) {
                        Log.e(TAG, "❌ YOLO inference error", e)
                    } finally {
                        imageProxy.close() // ✅ correct place
                    }
                }
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )

            Log.i(TAG, "📷 Camera started")

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        mlExecutor.shutdown()
    }
}
