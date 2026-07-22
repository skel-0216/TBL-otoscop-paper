package com.example.otoview.gallery

import android.content.ContentUris
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.provider.MediaStore

object GalleryStore {

    private val COLLECTION: Uri =
        if (Build.VERSION.SDK_INT >= 29)
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
        else
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI

    private val PROJECTION = arrayOf(
        MediaStore.Images.Media._ID,
        MediaStore.Images.Media.DISPLAY_NAME,
        if (Build.VERSION.SDK_INT >= 29)
            MediaStore.Images.Media.RELATIVE_PATH
        else
            MediaStore.Images.Media.DATA,             // deprecated but for API 28 and below compatibility
        MediaStore.Images.Media.DATE_ADDED,
        MediaStore.Images.Media.MIME_TYPE
    )

    /**
     * Select only images saved by OtoView (both new/old paths) and return newest-first.
     * - API 29+: RELATIVE_PATH LIKE 'Pictures/otoview%' OR 'Pictures/OtoView%'
     * - API 28-: DATA LIKE '%/Pictures/otoview/%' OR '%/Pictures/OtoView/%'
     * + filename prefix 'OTV_' (optional) and JPEG MIME filter.
     */
    fun queryOtoViewUris(ctx: Context): List<Uri> {
        val resolver = ctx.contentResolver

        val selection: String
        val args: Array<String>

        if (Build.VERSION.SDK_INT >= 29) {
            selection =
                "(" +
                        "${MediaStore.Images.Media.RELATIVE_PATH} LIKE ? OR " +
                        "${MediaStore.Images.Media.RELATIVE_PATH} LIKE ?" +
                        ") AND " +
                        "${MediaStore.Images.Media.MIME_TYPE}=? AND " +
                        "${MediaStore.Images.Media.DISPLAY_NAME} LIKE ?"
            args = arrayOf(
                "Pictures/otoview%",        // New path (recommended)
                "Pictures/OtoView%",        // Old path (uppercase)
                "image/jpeg",
                "OTV_%"
            )
        } else {
            // API 28 and below: based on DATA path
            @Suppress("DEPRECATION")
            selection =
                "(" +
                        "${MediaStore.Images.Media.DATA} LIKE ? OR " +
                        "${MediaStore.Images.Media.DATA} LIKE ?" +
                        ") AND " +
                        "${MediaStore.Images.Media.MIME_TYPE}=? AND " +
                        "${MediaStore.Images.Media.DISPLAY_NAME} LIKE ?"
            @Suppress("DEPRECATION")
            args = arrayOf(
                "%/Pictures/otoview/%",
                "%/Pictures/OtoView/%",
                "image/jpeg",
                "OTV_%"
            )
        }

        val sort = "${MediaStore.Images.Media.DATE_ADDED} DESC"

        val out = ArrayList<Uri>()
        resolver.query(COLLECTION, PROJECTION, selection, args, sort)?.use { c ->
            val idIdx = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            while (c.moveToNext()) {
                val id = c.getLong(idIdx)
                out += ContentUris.withAppendedId(COLLECTION, id)
            }
        }

        // If old files are missed by the filter above, fallback once more here (filename/extension only, no path restriction):
        if (out.isEmpty()) {
            val sel2 = "${MediaStore.Images.Media.MIME_TYPE}=? AND ${MediaStore.Images.Media.DISPLAY_NAME} LIKE ?"
            val args2 = arrayOf("image/jpeg", "OTV_%")
            resolver.query(COLLECTION, PROJECTION, sel2, args2, sort)?.use { c ->
                val idIdx = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                while (c.moveToNext()) {
                    val id = c.getLong(idIdx)
                    out += ContentUris.withAppendedId(COLLECTION, id)
                }
            }
        }

        return out
    }

    /**
     * Delete the given image Uri from MediaStore.
     * @return number of deleted rows (1 on success).
     * On API 29+, a file not owned by the app may throw RecoverableSecurityException;
     * the caller handles it by requesting user consent via IntentSender.
     */
    fun deleteUri(ctx: Context, uri: Uri): Int {
        return ctx.contentResolver.delete(uri, null, null)
    }
}
