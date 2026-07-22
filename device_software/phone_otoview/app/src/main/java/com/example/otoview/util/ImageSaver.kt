package com.example.otoview.util

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Single save utility:
 * - API 29+ : MediaStore (RELATIVE_PATH=Pictures/otoview, IS_PENDING flag)
 * - API 28- : save directly to the public Pictures/otoview folder + MediaScanner update
 * - (optional) save with crop applied + write crop metadata JSON to EXIF UserComment
 */
object ImageSaver {

    private const val RELATIVE_DIR = "Pictures/otoview"

    data class CropMeta(
        val srcW: Int,
        val srcH: Int,
        val centerX: Float,
        val centerY: Float,
        val size: Int,
        val scaleMode: String // "COVER" or "FIT" (the saved crop is handled the same way on a square basis)
    )

    /**
     * Default save function (no crop)
     */
    suspend fun saveJpeg(
        context: Context,
        bitmap: Bitmap,
        quality: Int = 95,
        prefix: String = "OTV_",
        exifUserComment: String? = null
    ): Result<Uri> = withContext(Dispatchers.IO) {
        runCatching {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US)
                .format(System.currentTimeMillis())
            val displayName = "${prefix}${ts}.jpg"

            if (Build.VERSION.SDK_INT >= 29) {
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    put(MediaStore.MediaColumns.RELATIVE_PATH, RELATIVE_DIR)
                    put(MediaStore.Images.Media.DATE_TAKEN, System.currentTimeMillis())
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val resolver = context.contentResolver
                val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: error("MediaStore insert failed")

                try {
                    resolver.openOutputStream(uri)?.use { os ->
                        if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, os)) {
                            error("Bitmap compress failed")
                        }
                        os.flush()
                    } ?: error("OpenOutputStream null")

                    exifUserComment?.let { comment ->
                        resolver.openFileDescriptor(uri, "rw")?.use { pfd ->
                            ExifInterface(pfd.fileDescriptor).apply {
                                setAttribute(ExifInterface.TAG_USER_COMMENT, comment)
                                saveAttributes()
                            }
                        }
                    }
                } catch (t: Throwable) {
                    resolver.delete(uri, null, null)
                    throw t
                } finally {
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                }

                uri
            } else {
                val base = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
                val dir = File(base, "otoview").apply { if (!exists()) mkdirs() }
                val file = File(dir, displayName)
                FileOutputStream(file).use { fos ->
                    if (!bitmap.compress(Bitmap.CompressFormat.JPEG, quality, fos)) {
                        error("Bitmap compress failed")
                    }
                    fos.flush()
                }

                exifUserComment?.let { comment ->
                    runCatching {
                        ExifInterface(file.absolutePath).apply {
                            setAttribute(ExifInterface.TAG_USER_COMMENT, comment)
                            saveAttributes()
                        }
                    }
                }

                MediaScannerConnection.scanFile(
                    context,
                    arrayOf(file.absolutePath),
                    arrayOf("image/jpeg"),
                    null
                )

                Uri.fromFile(file)
            }
        }
    }

    /**
     * Save based on crop metadata:
     * - square-crop the actual bitmap, then save
     * - write metadata JSON to EXIF UserComment
     * - if outSize is given, square-resize before saving (e.g. fixed 720)
     */
    suspend fun saveJpeg(
        context: Context,
        bitmap: Bitmap,
        crop: CropMeta?,
        quality: Int = 95,
        prefix: String = "OTV_",
        outSize: Int? = null
    ): Result<Uri> = withContext(Dispatchers.IO) {
        // JSON comment
        val comment = crop?.let {
            """{"srcW":${it.srcW},"srcH":${it.srcH},"centerX":${it.centerX},"centerY":${it.centerY},"size":${it.size},"scaleMode":"${it.scaleMode}"}"""
        }

        // apply crop if needed
        val toSave: Bitmap =
            if (crop != null) cropBitmapToSquare(bitmap, crop, outSize) else bitmap

        // if we created a new bitmap, recycle it after saving
        val needRecycle = (crop != null) || (outSize != null)

        runCatching {
            try {
                saveJpeg(context, toSave, quality, prefix, comment).getOrThrow()
            } finally {
                if (needRecycle && toSave !== bitmap && !toSave.isRecycled) {
                    toSave.recycle()
                }
            }
        }
    }

    /**
     * Crop a square region from the input bitmap based on CropMeta.
     * - map meta.srcW/H coordinate system → actual bitmap coordinate system
     * - boundary correction and keep it square
     * - square-resize if outSize is given
     */
    private fun cropBitmapToSquare(src: Bitmap, meta: CropMeta, outSize: Int? = null): Bitmap {
        // 1) compute the crop rect in the srcW/H coordinate system
        val half = meta.size / 2f
        var left = meta.centerX - half
        var top = meta.centerY - half
        var right = meta.centerX + half
        var bottom = meta.centerY + half

        // boundary correction
        if (left < 0f) { right -= left; left = 0f }
        if (top < 0f) { bottom -= top; top = 0f }
        if (right > meta.srcW) { val d = right - meta.srcW; left -= d; right = meta.srcW.toFloat(); if (left < 0f) left = 0f }
        if (bottom > meta.srcH) { val d = bottom - meta.srcH; top -= d; bottom = meta.srcH.toFloat(); if (top < 0f) top = 0f }

        val sideSrc = min(right - left, bottom - top)

        // 2) map to the actual bitmap coordinate system
        val scaleX = src.width.toFloat() / meta.srcW.toFloat()
        val scaleY = src.height.toFloat() / meta.srcH.toFloat()

        var lPx = (left * scaleX).roundToInt()
        var tPx = (top * scaleY).roundToInt()
        var sidePx = min((sideSrc * scaleX), (sideSrc * scaleY)).roundToInt()

        // safety clamp
        lPx = lPx.coerceIn(0, max(0, src.width - 1))
        tPx = tPx.coerceIn(0, max(0, src.height - 1))
        sidePx = sidePx.coerceAtLeast(1)
        if (lPx + sidePx > src.width) sidePx = src.width - lPx
        if (tPx + sidePx > src.height) sidePx = src.height - tPx

        val cropped = Bitmap.createBitmap(src, lPx, tPx, sidePx, sidePx)

        // 3) square-resize if outSize is given
        return if (outSize != null && outSize > 0 &&
            (cropped.width != outSize || cropped.height != outSize)
        ) {
            val scaled = Bitmap.createScaledBitmap(cropped, outSize, outSize, true)
            cropped.recycle()
            scaled
        } else {
            cropped
        }
    }
}
