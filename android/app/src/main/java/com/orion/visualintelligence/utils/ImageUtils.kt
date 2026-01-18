package com.orion.visualintelligence.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

object ImageUtils {
    fun toBitmap(image: ImageProxy): Bitmap {
        val nv21 = yuv420ToNv21(image)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, image.width, image.height),
            95,
            out
        )

        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val ySize = image.planes[0].buffer.remaining()
        val uSize = image.planes[1].buffer.remaining()
        val vSize = image.planes[2].buffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        var offset = 0

        image.planes[0].buffer.get(nv21, 0, ySize)
        offset += ySize

        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val uPixel = uPlane.pixelStride
        val vPixel = vPlane.pixelStride

        val uvPossibleSwap = uPixel == 2 && vPixel == 2

        if (uvPossibleSwap) {
            vPlane.buffer.get(nv21, offset, vSize)
            offset += vSize
            uPlane.buffer.get(nv21, offset, uSize)
        } else {
            uPlane.buffer.get(nv21, offset, uSize)
            offset += uSize
            vPlane.buffer.get(nv21, offset, vSize)
        }

        return nv21
    }
}
