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
import com.orion.visualintelligence.model.DetectionOverlay
import com.orion.visualintelligence.ui.OverlayView
import com.orion.visualintelligence.utils.ImageUtils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

private const val TAG = "ORION_LINKEDIN"

class MainActivity : ComponentActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var mlExecutor: ExecutorService
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView

    private var yoloDetector: YoloDetector? = null
    private var lastInferenceTime = 0L
    private var frameCount = 0
    private var fpsStartTime = 0L

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) initPipeline()
            else Log.e(TAG, "❌ Camera permission denied")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()
        mlExecutor = Executors.newSingleThreadExecutor()

        Log.i(TAG, "🚀 ORION Visual Intelligence started")
        Log.i(TAG, "📱 YOLOv8 + TensorFlow Lite (On-device inference)")

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initPipeline() {
        mlExecutor.execute {
            Log.i(TAG, "🧠 Initializing YOLO detector…")
            yoloDetector = YoloDetector(applicationContext)
            runOnUiThread { startCamera() }
        }
    }

    private fun startCamera() {
        previewView = PreviewView(this)
        overlayView = OverlayView(this)

        setContentView(
            FrameLayout(this).apply {
                addView(previewView)
                addView(overlayView)
            }
        )

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            fpsStartTime = SystemClock.elapsedRealtime()

            analysis.setAnalyzer(cameraExecutor) { imageProxy ->

                val now = SystemClock.elapsedRealtime()
                if (now - lastInferenceTime < 120) {
                    imageProxy.close()
                    return@setAnalyzer
                }
                lastInferenceTime = now

                val bitmap = ImageUtils.toBitmap(imageProxy)
                imageProxy.close()

                mlExecutor.execute {
                    val detector = yoloDetector ?: return@execute

                    val start = SystemClock.elapsedRealtime()
                    val detections = detector.detect(bitmap)
                    val end = SystemClock.elapsedRealtime()

                    frameCount++
                    val elapsed = end - fpsStartTime
                    val fps = if (elapsed > 0) (frameCount * 1000f / elapsed) else 0f

                    Log.i(
                        TAG,
                        "⏱ Inference=${end - start} ms | FPS=${"%.1f".format(fps)} | Detections=${detections.size}"
                    )

                    detections.forEach {
                        Log.i(
                            TAG,
                            "📦 ${it.label.uppercase()} detected | confidence=${"%.2f".format(it.confidence)}"
                        )
                    }

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
                }
            }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        yoloDetector?.close()
        cameraExecutor.shutdown()
        mlExecutor.shutdown()
        Log.i(TAG, "🧹 ORION pipeline stopped")
    }
}
