package com.example.otoview.gallery

import android.app.AlertDialog
import android.app.RecoverableSecurityException
import android.content.Intent
import android.content.IntentSender
import android.database.ContentObserver
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.DisplayMetrics
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.otoview.R

class GalleryActivity : ComponentActivity() {

    private lateinit var recycler: RecyclerView
    private lateinit var adapter: GalleryAdapter
    private lateinit var tvEmpty: View

    private val mediaObserver = object : ContentObserver(Handler(Looper.getMainLooper())) {
        override fun onChange(selfChange: Boolean) {
            super.onChange(selfChange)
            loadGallery()
        }
    }

    // Launcher to get user consent when deleting a file not owned by the app on API 29+
    private var pendingDeleteUri: Uri? = null
    private val deleteConsentLauncher = registerForActivityResult(
        ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            onDeleteSucceeded()
        }
        pendingDeleteUri = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("GalleryActivity", "onCreate")
        setContentView(R.layout.activity_gallery)

        recycler = findViewById(R.id.rv)
        tvEmpty = findViewById(R.id.tvEmpty)

        recycler.layoutManager = GridLayoutManager(this, calcSpanCount(minCellDp = 110))
        recycler.addItemDecoration(GridSpacingDecoration(spacePx = dp(8)))

        adapter = GalleryAdapter(
            onClick = { uri ->
                val intent = Intent(this, ImageViewerActivity::class.java).apply {
                    data = uri
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(intent)
            },
            onLongClick = { uri -> confirmDelete(uri) }
        )
        recycler.adapter = adapter

        findViewById<View>(R.id.btnClose)?.setOnClickListener { finish() }

        loadGallery()

        contentResolver.registerContentObserver(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            true,
            mediaObserver
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        contentResolver.unregisterContentObserver(mediaObserver)
    }

    private fun loadGallery() {
        // Kept as-is from the existing implementation
        Thread {
            val uris: List<Uri> = GalleryStore.queryOtoViewUris(this@GalleryActivity)
            runOnUiThread {
                adapter.submitList(uris)
                tvEmpty.visibility = if (uris.isEmpty()) View.VISIBLE else View.GONE
            }
        }.start()
    }

    /** Show a delete confirmation dialog on long-press */
    private fun confirmDelete(uri: Uri) {
        AlertDialog.Builder(this)
            .setTitle(R.string.delete_title)
            .setMessage(R.string.delete_message)
            .setNegativeButton(R.string.cancel, null)
            .setPositiveButton(R.string.delete) { _, _ -> deleteImage(uri) }
            .show()
    }

    private fun deleteImage(uri: Uri) {
        Thread {
            try {
                val rows = GalleryStore.deleteUri(this@GalleryActivity, uri)
                runOnUiThread {
                    if (rows > 0) onDeleteSucceeded() else showDeleteFailed()
                }
            } catch (e: SecurityException) {
                // API 29+: file not owned by the app → request user consent (IntentSender)
                val intentSender: IntentSender? =
                    if (Build.VERSION.SDK_INT >= 30) {
                        MediaStore.createDeleteRequest(
                            contentResolver, listOf(uri)
                        ).intentSender
                    } else if (Build.VERSION.SDK_INT >= 29 && e is RecoverableSecurityException) {
                        e.userAction.actionIntent.intentSender
                    } else {
                        null
                    }

                runOnUiThread {
                    if (intentSender != null) {
                        pendingDeleteUri = uri
                        deleteConsentLauncher.launch(
                            IntentSenderRequest.Builder(intentSender).build()
                        )
                    } else {
                        showDeleteFailed()
                    }
                }
            }
        }.start()
    }

    private fun onDeleteSucceeded() {
        Toast.makeText(this, R.string.delete_done, Toast.LENGTH_SHORT).show()
        loadGallery()
    }

    private fun showDeleteFailed() {
        Toast.makeText(this, R.string.delete_failed, Toast.LENGTH_SHORT).show()
    }

    private fun calcSpanCount(minCellDp: Int): Int {
        val dm: DisplayMetrics = resources.displayMetrics
        val wDp = (dm.widthPixels / dm.density)
        val span = (wDp / minCellDp).toInt().coerceAtLeast(2)
        return span
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private class GridSpacingDecoration(private val spacePx: Int) : RecyclerView.ItemDecoration() {
        override fun getItemOffsets(
            outRect: android.graphics.Rect,
            view: android.view.View,
            parent: RecyclerView,
            state: RecyclerView.State
        ) {
            val pos = parent.getChildAdapterPosition(view)
            val spanCount = (parent.layoutManager as? GridLayoutManager)?.spanCount ?: 3
            val col = pos % spanCount
            outRect.left = spacePx - col * spacePx / spanCount
            outRect.right = (col + 1) * spacePx / spanCount
            if (pos < spanCount) outRect.top = spacePx else outRect.top = spacePx / 2
            outRect.bottom = spacePx / 2
        }
    }
}
