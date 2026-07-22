package com.example.otoview

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.*
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import android.widget.ToggleButton
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import com.example.otoview.bench.BenchLog
import com.example.otoview.bench.ResourceSampler
import com.example.otoview.gallery.GalleryActivity
import com.example.otoview.gallery.ImageViewerActivity
import com.example.otoview.net.CommandClient
import com.example.otoview.stream.VideoStreamManager
import com.example.otoview.ui.CropTextureView
import com.example.otoview.util.ImageSaver
import com.example.otoview.util.ViewCapture
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private val defaultIp = "10.42.0.1"
    private val defaultVideoPort = 1234
    private val defaultCmdPort = 4321

    private lateinit var videoView: CropTextureView
    private lateinit var btnStart: Button
    private lateinit var btnGallery: Button
    private lateinit var btnCapture: Button
    private lateinit var toggleBtnLed: ToggleButton
    private lateinit var tvConnStatus: TextView
    private lateinit var brandLogo: View   // tap target to enter developer mode (former UNIST logo)

    // connection/session info card
    private var infoWifiDot: View? = null
    private var infoWifiText: TextView? = null
    private var infoStream: TextView? = null
    private var infoRate: TextView? = null
    private var debugCircleView: View? = null   // alignment circle (circle overlay)

    private var surface: Surface? = null
    private lateinit var streamManager: VideoStreamManager
    private lateinit var cmdClient: CommandClient
    private lateinit var cm: ConnectivityManager

    private var srcW: Int = 0
    private var srcH: Int = 0

    // track ROI state (for the debug HUD)
    private var lastRoiSize: Int = 0
    private var lastRoiCx: Int = 0
    private var lastRoiCy: Int = 0

    // ===================================================================
    // ===== video area (crop) parameters — rectangle =====
    // Crop a "rectangle" from the source (1280x720, IMX708/Module3) and fill a container of the same ratio.
    // Only the actual video is shown, no padding (the container aspect ratio is set automatically to CROP_WIDTH:CROP_HEIGHT).
    // IMX708 has 16:9 square pixels, so no anamorphic correction is needed (matching the ratio gives no distortion).
    //
    //  CROP_CENTER_X : horizontal center (source px). ↓left / ↑right
    //  CROP_WIDTH    : crop width (source px). ↑ to see more left/right (keeps center, expands both sides)
    //  CROP_TOP      : crop top start y (source px). 0 = top of source (removes top padding)
    //  CROP_HEIGHT   : crop height (source px). ↑ to see more downward
    //  → vertical center = CROP_TOP + CROP_HEIGHT/2, left/right = CROP_CENTER_X ± CROP_WIDTH/2
    // ===================================================================
    private var parX: Float = 1.00f   // (unused, legacy)
    private var parY: Float = 1.00f
    private val CROP_CENTER_X = 605    // horizontal center
    private val CROP_WIDTH    = 726    // width (old 720 + 3px on each side)
    private val CROP_TOP      = 0      // top start (padding removed: top of source)
    private val CROP_HEIGHT   = 633    // height (old 630 + 3px below)
    private val DEF_SCALE      = CropTextureView.ScaleMode.STRETCH

    // ===== developer mode (OFF by default, toggled by repeated taps on the UNIST logo) =====
    private var devMode: Boolean = false
    private var eggTapCount = 0
    private var eggStartMs  = 0L
    private val EGG_TAPS_REQUIRED = 7
    private val EGG_TIME_WINDOW_MS = 2500L

    // state management
    private var changingLed = false
    private var currentState: VideoStreamManager.State = VideoStreamManager.State.IDLE
        set(value) {
            field = value
            val status = when (value) {
                VideoStreamManager.State.IDLE -> getString(R.string.stream_status_init)
                is VideoStreamManager.State.CONNECTING -> "Stream: Connecting…"
                is VideoStreamManager.State.PLAYING -> "Stream: Playing (${value.fps} fps)"
                is VideoStreamManager.State.DISCONNECTED -> "Stream: Disconnected (${value.reason})"
                is VideoStreamManager.State.ERROR -> "Stream: Error (${value.message})"
            }
            Log.d(TAG, status)
            applyConnUi(value)
            updateDebugHud()
        }

    // network
    private val netCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            if (surface != null && currentState is VideoStreamManager.State.DISCONNECTED) {
                lifecycleScope.launch { startStreaming() }
            }
        }
    }

    // track Wi-Fi connection state (independent of streaming state) — drives the info card Wi-Fi cell
    private val wifiRequest by lazy {
        NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .build()
    }
    private val wifiCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) { runOnUiThread { updateWifiCell() } }
        override fun onLost(network: Network) { runOnUiThread { updateWifiCell() } }
        override fun onUnavailable() { runOnUiThread { updateWifiCell() } }
    }

    /** Whether any network currently has a Wi-Fi transport (detected even if the default network is cellular) */
    private fun isWifiConnected(): Boolean =
        runCatching {
            cm.allNetworks.any { n ->
                cm.getNetworkCapabilities(n)
                    ?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
            }
        }.getOrDefault(false)

    /** Update the info card Wi-Fi cell to the actual Wi-Fi connection state */
    private fun updateWifiCell() {
        val connected = isWifiConnected()
        val colorRes = if (connected) R.color.status_ok else R.color.status_idle
        infoWifiDot?.backgroundTintList =
            android.content.res.ColorStateList.valueOf(
                androidx.core.content.ContextCompat.getColor(this, colorRes)
            )
        infoWifiText?.text = if (connected) "Connected" else "None"
    }

    // --- deployment benchmark session (only active in developer mode) ---
    private var resourceSampler: ResourceSampler? = null
    private var benchRunNo: Int = 0

    /**
     * Turning on developer mode starts the instrumentation session. The numbers used in the paper
     * are recorded on the real usage path while using the app normally in this state.
     */
    private fun startBenchSession() {
        if (!BenchLog.enabled) return
        benchRunNo++
        BenchLog.startSession("run$benchRunNo")
        resourceSampler = ResourceSampler(this).also { it.start() }
        Toast.makeText(this, "Benchmark session run$benchRunNo started", Toast.LENGTH_SHORT).show()
    }

    private fun exportBenchSession() {
        if (!BenchLog.enabled || !BenchLog.hasSession()) return
        resourceSampler?.stop()
        resourceSampler = null
        val f = BenchLog.export(this)
        Toast.makeText(
            this,
            if (f != null) "Benchmark saved: ${f.name}" else "Benchmark save failed",
            Toast.LENGTH_LONG
        ).show()
    }

    // --- debug HUD (developer mode only) ---
    private var debugHud: TextView? = null
    private var hudVisible: Boolean = true

    private fun ensureDebugHud() {
        if (!devMode) { // do not even create it outside developer mode
            debugHud?.visibility = View.GONE
            return
        }
        if (debugHud != null) {
            debugHud?.visibility = View.VISIBLE
            return
        }
        val tv = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(0x66000000)
            textSize = 11f
            setPadding(12, 8, 12, 8)
            isClickable = true
            isFocusable = false
            text = "debug"
        }
        val lp = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
        addContentView(tv, lp)
        debugHud = tv

        tv.setOnClickListener {
            // brief help
            Toast.makeText(
                this,
                "Gestures: 1-finger drag=move, 2-finger pinch=size, double-tap=FULL/ROI\n" +
                        "Long-press this HUD to save the benchmark session",
                Toast.LENGTH_SHORT
            ).show()
        }
        // save mid-way and start the next run without turning off developer mode.
        tv.setOnLongClickListener {
            exportBenchSession()
            startBenchSession()
            true
        }
    }

    private fun updateDebugHud() {
        if (!devMode) { debugHud?.visibility = View.GONE; return }
        debugHud ?: return
        val vw = videoView.width
        val vh = videoView.height
        val stateStr = when (val s = currentState) {
            is VideoStreamManager.State.PLAYING -> "PLAYING ${s.fps}fps"
            is VideoStreamManager.State.CONNECTING -> "CONNECTING"
            is VideoStreamManager.State.DISCONNECTED -> "DISCONNECTED"
            is VideoStreamManager.State.ERROR -> "ERROR"
            else -> "IDLE"
        }
        val roiStr = if (lastRoiSize == 0) "FULL"
        else "ROI size=$lastRoiSize @ (${lastRoiCx},${lastRoiCy})"
        val parStr = "parX=${"%.2f".format(parX)} parY=${"%.2f".format(parY)}"
        debugHud?.text =
            "state=$stateStr | src=${srcW}x${srcH} | view=${vw}x${vh} | $roiStr | scale=${getScaleMode().name} | $parStr" +
                    "\n${BenchLog.hudLine()}"
        debugHud?.visibility = if (hudVisible) View.VISIBLE else View.GONE
    }

    private fun getScaleMode(): CropTextureView.ScaleMode =
        try {
            val f = CropTextureView::class.java.getDeclaredField("scaleMode")
            f.isAccessible = true
            (f.get(videoView) as? CropTextureView.ScaleMode) ?: CropTextureView.ScaleMode.FIT
        } catch (_: Throwable) { CropTextureView.ScaleMode.FIT }

    // ===== lifecycle =====
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initDebugTogglesSafe()
        BenchLog.init(applicationContext)

        // Keep the screen from dimming or turning off while streaming, even with no touch input.
        // Fixes the screen dimming during real use when the hands are not free while looking through the otoscope.
        // Applies only while this screen is in the foreground and is released automatically when moving to the viewer.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        streamManager = VideoStreamManager(this)
        cmdClient = CommandClient()

        videoView = findViewById(R.id.videoView)
        // fill without distortion by matching the video container aspect ratio to the crop rectangle (width:height)
        findViewById<View>(R.id.videoContainer).let { vc ->
            (vc.layoutParams as? androidx.constraintlayout.widget.ConstraintLayout.LayoutParams)?.let { lp ->
                lp.dimensionRatio = "$CROP_WIDTH:$CROP_HEIGHT"
                vc.layoutParams = lp
            }
        }
        btnStart = findViewById(R.id.btnStart)
        btnGallery = findViewById(R.id.btnGallery)
        btnCapture = findViewById(R.id.btnCaptureAi)
        toggleBtnLed = findViewById(R.id.btn_switchLed)
        tvConnStatus = findViewById(R.id.tvConnStatus)
        brandLogo = findViewById(R.id.brandLogo)
        infoWifiDot = findViewById(R.id.infoWifiDot)
        infoWifiText = findViewById(R.id.infoWifiText)
        infoStream = findViewById(R.id.infoStream)
        infoRate = findViewById(R.id.infoRate)
        debugCircleView = findViewById(R.id.debugCircle)
        calibOverlay = findViewById(R.id.calibOverlay)

        // default UI
        tvConnStatus.text = "Idle"
        tvConnStatus.setBackgroundResource(R.drawable.status_chip_idle)

        // alignment circle: hidden by default (shown only in developer mode)
        debugCircleView?.visibility = View.GONE

        // Texture prepare/release
        videoView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) {
                surface = Surface(st)
                // default display: COVER + PAR + default ROI
                videoView.setScaleMode(DEF_SCALE)
                videoView.setPixelAspectFix(parX, parY)
                applyDefaultRoi() // ROI 720@(465,360)
                ensureDebugHud()
                updateDebugHud()
            }
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) {
                videoView.applyTransform()
                updateDebugHud()
            }
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
                surface?.release()
                surface = null
                streamManager.stop()
                return true
            }
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {
                // Frame metering on the display side. The decoder emits in bursts, so this is the actually visible frame.
                BenchLog.display.onFrameRendered()
            }
        }

        // start/gallery/capture
        btnStart.setOnClickListener { lifecycleScope.launch { startStreaming() } }
        btnGallery.setOnClickListener { startActivity(Intent(this, GalleryActivity::class.java)) }
        btnCapture.setOnClickListener {
            if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                    android.content.pm.PackageManager.PERMISSION_GRANTED
                ) {
                    requestPermissions(arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), REQ_WRITE_EXTERNAL)
                    return@setOnClickListener
                }
            }
            lifecycleScope.launch { captureSaveAndOpen() }
        }

        // LED toggle
        toggleBtnLed.setOnCheckedChangeListener { _, isChecked ->
            if (changingLed) return@setOnCheckedChangeListener
            lifecycleScope.launch {
                val prev = !isChecked
                if (!cmdClient.isConnected) {
                    val res = cmdClient.connectSmart(cm, defaultIp, defaultCmdPort, timeoutMs = 8000)
                    if (res.isFailure) {
                        Log.w(TAG, "Cmd connect failed: ${res.exceptionOrNull()?.message}")
                        revertSwitch(prev); return@launch
                    }
                }
                val send = cmdClient.sendLight(isChecked)
                if (send.isFailure) {
                    Log.w(TAG, "Cmd send failed: ${send.exceptionOrNull()?.message}")
                    revertSwitch(prev)
                }
            }
        }

        // ===== developer mode: toggle by repeated taps on the logo mark =====
        brandLogo.setOnClickListener { handleEggTap() }

        // status chip tap/long-press → only works in developer mode
        tvConnStatus.setOnClickListener {
            if (!devMode) return@setOnClickListener
            // cycle COVER → FIT → STRETCH → COVER
            val newMode = when (getScaleMode()) {
                CropTextureView.ScaleMode.COVER -> CropTextureView.ScaleMode.FIT
                CropTextureView.ScaleMode.FIT -> CropTextureView.ScaleMode.STRETCH
                CropTextureView.ScaleMode.STRETCH -> CropTextureView.ScaleMode.COVER
            }
            videoView.setScaleMode(newMode)
            videoView.applyTransform()
            Toast.makeText(this, "ScaleMode = $newMode", Toast.LENGTH_SHORT).show()
            updateDebugHud()
        }
        tvConnStatus.setOnLongClickListener {
            if (!devMode) return@setOnLongClickListener false
            if (srcW > 0 && srcH > 0) {
                if (lastRoiSize == 0) applyDefaultRoi() else {
                    videoView.setFullFrame(true)
                    videoView.applyTransform()
                    lastRoiSize = 0
                }
                updateDebugHud()
                true
            } else false
        }
    }

    override fun onResume() {
        super.onResume()
        cm.registerDefaultNetworkCallback(netCallback)
        runCatching { cm.registerNetworkCallback(wifiRequest, wifiCallback) }
        updateWifiCell() // reflect the current Wi-Fi state immediately on entry
    }

    override fun onPause() {
        super.onPause()
        runCatching { cm.unregisterNetworkCallback(netCallback) }
        runCatching { cm.unregisterNetworkCallback(wifiCallback) }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Save so no session is left behind even if the app is just closed.
        exportBenchSession()
        cmdClient.sendQuitOnDestroy()
        streamManager.stop()
        cmdClient.close()
        surface?.release()
        surface = null
    }

    // ===== start streaming =====
    private suspend fun startStreaming() {
        val s = surface ?: run {
            Toast.makeText(this, "Surface not ready", Toast.LENGTH_SHORT).show()
            return
        }
        currentState = VideoStreamManager.State.CONNECTING
        try {
            streamManager.start(
                host = defaultIp,
                port = defaultVideoPort,
                surface = s,
                autoRetry = true,
                onVideoSize = { w, h ->
                    srcW = w; srcH = h
                    infoStream?.text = "${h}p"
                    Log.d(TAG, "onVideoSize: src=${w}x${h}, view=${videoView.width}x${videoView.height}")

                    // apply source size + PAR + default ROI + COVER
                    videoView.setSourceSize(w, h)
                    videoView.setScaleMode(DEF_SCALE)
                    videoView.setPixelAspectFix(parX, parY)
                    applyDefaultRoi() // this also calls applyTransform

                    updateDebugHud()
                },
                onState = { st, detail ->
                    currentState = st
                    if (detail != null && st is VideoStreamManager.State.ERROR) {
                        Log.e(TAG, "Stream ERROR detail:\n$detail")
                    }
                }
            )
        } catch (t: Throwable) {
            currentState = VideoStreamManager.State.ERROR(t.message ?: "Unknown")
        }
    }

    // ===== apply default crop (rectangle → fill a same-ratio container with STRETCH, no distortion) =====
    private fun applyDefaultRoi() {
        val cropW = CROP_WIDTH.coerceAtLeast(16)
        val cropH = CROP_HEIGHT.coerceAtLeast(16)
        val cx = CROP_CENTER_X.toFloat()
        val cy = CROP_TOP + cropH / 2f   // vertical center = top start + height/2

        videoView.setFullFrame(false)
        videoView.setAllowOutOfBounds(true) // safe even if set out of bounds (current values are within the source)
        videoView.setScaleMode(DEF_SCALE)   // STRETCH
        videoView.setCropRect(cx, cy, cropW, cropH)
        videoView.applyTransform()

        lastRoiSize = cropH
        lastRoiCx = cx.toInt()
        lastRoiCy = cy.toInt()
        Log.d(TAG, "applyDefaultRoi: crop=${cropW}x${cropH} center=($cx,$cy) scale=${DEF_SCALE.name}")
    }

    // ===== capture save / open viewer =====
    private suspend fun captureSaveAndOpen() {
        // From the moment the button is pressed until the viewer opens. Analysis is started by the user
        // tapping in the viewer, so record it cut here to avoid mixing in the human-in-the-loop segment.
        val clickNs = BenchLog.nowNs()
        val bmp: Bitmap = withContext(Dispatchers.Main) {
            BenchLog.time("capture_grab") {
                ViewCapture.captureTextureView(this@MainActivity, videoView)
            }
        } ?: run {
            BenchLog.event("capture_failed")
            Toast.makeText(this@MainActivity, "Frame capture failed", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val saved = BenchLog.time("capture_save_jpeg", mapOf("w" to bmp.width, "h" to bmp.height)) {
                ImageSaver.saveJpeg(this, bmp, quality = 95)
            }
            val uri: Uri = saved.getOrNull() ?: run {
                val cacheFile = File(cacheDir, "capture_${System.currentTimeMillis()}.jpg")
                withContext(Dispatchers.IO) {
                    FileOutputStream(cacheFile).use { fos ->
                        bmp.compress(Bitmap.CompressFormat.JPEG, 95, fos)
                        fos.flush()
                    }
                }
                androidx.core.content.FileProvider.getUriForFile(
                    this@MainActivity,
                    "${applicationContext.packageName}.fileprovider",
                    cacheFile
                )
            }
            BenchLog.since("capture_to_viewer", clickNs)
            startActivity(
                Intent(this, ImageViewerActivity::class.java)
                    .setData(uri)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            )
        } catch (t: Throwable) {
            Toast.makeText(this@MainActivity, "Capture/save failed: ${t.message}", Toast.LENGTH_LONG).show()
        } finally {
            bmp.recycle()
        }
    }

    private fun revertSwitch(checked: Boolean) { toggleBtnLed.isChecked = checked }

    private fun applyConnUi(state: VideoStreamManager.State) {
        if (!::tvConnStatus.isInitialized) return
        when (state) {
            VideoStreamManager.State.IDLE -> {
                tvConnStatus.text = "Idle"
                tvConnStatus.setBackgroundResource(R.drawable.status_chip_idle)
            }
            is VideoStreamManager.State.CONNECTING -> {
                tvConnStatus.text = "Connecting…"
                tvConnStatus.setBackgroundResource(R.drawable.status_chip_warn)
            }
            is VideoStreamManager.State.PLAYING -> {
                tvConnStatus.text = "Playing (${state.fps} fps)"
                tvConnStatus.setBackgroundResource(R.drawable.status_chip_ok)
            }
            is VideoStreamManager.State.DISCONNECTED -> {
                tvConnStatus.text = "Disconnected"
                tvConnStatus.setBackgroundResource(R.drawable.status_chip_warn)
            }
            is VideoStreamManager.State.ERROR -> {
                tvConnStatus.text = "Error"
                tvConnStatus.setBackgroundResource(R.drawable.status_chip_err)
            }
        }
        updateInfoCard(state)
    }

    /** Sync the info card Rate cell with the stream state (the Wi-Fi cell is handled separately in updateWifiCell) */
    private fun updateInfoCard(state: VideoStreamManager.State) {
        infoRate?.text = when (state) {
            is VideoStreamManager.State.PLAYING -> "${state.fps} fps"
            else -> "— fps"
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_WRITE_EXTERNAL) {
            lifecycleScope.launch { captureSaveAndOpen() }
        }
    }

    // ===== toggle developer mode (repeated taps on the UNIST logo) =====
    private fun handleEggTap() {
        val now = System.currentTimeMillis()
        if (eggStartMs == 0L || now - eggStartMs > EGG_TIME_WINDOW_MS) {
            eggStartMs = now
            eggTapCount = 0
        }
        eggTapCount++

        val remain = (EGG_TAPS_REQUIRED - eggTapCount).coerceAtLeast(0)
        if (!devMode && remain in 1..3) {
            Toast.makeText(this, "$remain more tap(s) to Developer mode", Toast.LENGTH_SHORT).show()
        }

        if (eggTapCount >= EGG_TAPS_REQUIRED) {
            devMode = !devMode
            eggTapCount = 0
            eggStartMs = 0L
            onDevModeChanged()
        }
    }

    private fun onDevModeChanged() {
        if (devMode) {
            Toast.makeText(this, "Developer mode ON", Toast.LENGTH_SHORT).show()
            // start the benchmark session (developer mode = instrumentation mode)
            startBenchSession()
            // enable the HUD
            ensureDebugHud()
            updateDebugHud()
            // show the alignment circle
            debugCircleView?.visibility = View.VISIBLE
            // enable gestures
            attachGestureControllers()
        } else {
            Toast.makeText(this, "Developer mode OFF", Toast.LENGTH_SHORT).show()
            // auto-save when the session ends
            exportBenchSession()
            // force-exit calibration
            calibMode = false
            calibOverlay?.visibility = View.GONE
            // hide the HUD
            debugHud?.visibility = View.GONE
            // hide the alignment circle
            debugCircleView?.visibility = View.GONE
            // disable gestures
            detachGestureControllers()
            // pin to the user-mode default state (keep the requested values)
            videoView.setScaleMode(DEF_SCALE)
            videoView.setPixelAspectFix(parX, parY)
            applyDefaultRoi()
        }
    }

    // ===== gestures (only active in developer mode) =====
    private var scaleDetector: ScaleGestureDetector? = null
    private var gestureDetector: GestureDetector? = null

    // ===== calibration mode =====
    private var calibOverlay: com.example.otoview.ui.CalibrationOverlayView? = null
    private var calibMode: Boolean = false

    private fun attachGestureControllers() {
        // reuse if already attached
        if (scaleDetector == null) {
            scaleDetector = ScaleGestureDetector(this,
                object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    override fun onScale(detector: ScaleGestureDetector): Boolean {
                        if (calibMode) {
                            calibOverlay?.scaleBy(detector.scaleFactor)
                            updateCalibReadout()
                            return true
                        }
                        videoView.scaleCropBy(detector.scaleFactor)
                        val (cx, cy) = videoView.getCropCenter()
                        lastRoiCx = cx; lastRoiCy = cy
                        lastRoiSize = if (videoView.isRectCrop())
                            videoView.getRectSize().first else videoView.getSquareSize()
                        updateDebugHud()
                        return true
                    }
                }
            )
        }
        if (gestureDetector == null) {
            gestureDetector = GestureDetector(this,
                object : GestureDetector.SimpleOnGestureListener() {
                    override fun onDown(e: MotionEvent): Boolean = true
                    override fun onLongPress(e: MotionEvent) {
                        if (!devMode) return
                        if (calibMode) exitCalibration() else enterCalibration()
                    }
                    override fun onScroll(
                        e1: MotionEvent?, e2: MotionEvent,
                        distanceX: Float, distanceY: Float
                    ): Boolean {
                        if (calibMode) {
                            calibOverlay?.moveBy(-distanceX, -distanceY)
                            updateCalibReadout()
                            return true
                        }
                        videoView.moveCropByViewDelta(-distanceX, -distanceY)
                        val (cx, cy) = videoView.getCropCenter()
                        lastRoiCx = cx; lastRoiCy = cy
                        lastRoiSize = if (videoView.isRectCrop())
                            videoView.getRectSize().first else videoView.getSquareSize()
                        updateDebugHud()
                        return true
                    }
                    override fun onDoubleTap(e: MotionEvent): Boolean {
                        if (calibMode) return true
                        if (lastRoiSize == 0) {
                            applyDefaultRoi()
                        } else {
                            videoView.setFullFrame(true)
                            videoView.applyTransform()
                            lastRoiSize = 0
                            updateDebugHud()
                        }
                        return true
                    }
                }
            )
        }
        // attach the touch listener (only in developer mode)
        videoView.setOnTouchListener { _, ev ->
            if (!devMode) return@setOnTouchListener false
            scaleDetector?.onTouchEvent(ev)
            gestureDetector?.onTouchEvent(ev)
            true
        }
    }

    private fun detachGestureControllers() {
        // remove the touch listener
        videoView.setOnTouchListener(null)
        // keep the instances (reuse on re-entry), but they do nothing
    }

    // ===== calibration mode =====
    // Show the video at native aspect ratio (FIT+FULL) without distortion, and set the ROI directly with
    // a square+inscribed-circle overlay on top. Overlay position/size → shown live as source pixels (CROP_CENTER/SIZE).
    private fun enterCalibration() {
        val sw = if (srcW > 0) srcW else 1280
        val sh = if (srcH > 0) srcH else 720
        calibMode = true

        // native aspect ratio: FIT with no crop (letterbox, zero distortion)
        videoView.setFullFrame(true)
        videoView.setScaleMode(CropTextureView.ScaleMode.FIT)
        videoView.applyTransform()
        debugCircleView?.visibility = View.GONE

        videoView.post {
            val vw = videoView.width.toFloat()
            val vh = videoView.height.toFloat()
            val r = fitContentRect(vw, vh, sw, sh)   // [l,t,r,b]
            calibOverlay?.setContentRect(r[0], r[1], r[2], r[3])
            // use the current crop values as the initial square
            val contentW = r[2] - r[0]
            val s = contentW / sw                    // uniform scale
            val curSize = if (lastRoiSize > 0) lastRoiSize else sh
            val curCx = if (lastRoiCx > 0) lastRoiCx else sw / 2
            val curCy = if (lastRoiCy > 0) lastRoiCy else sh / 2
            calibOverlay?.setSquare(r[0] + curCx * s, r[1] + curCy * s, curSize * s)
            calibOverlay?.visibility = View.VISIBLE
            updateCalibReadout()
        }
        Toast.makeText(this, "Calibration ON · drag=move, pinch=size, long-press=apply/exit", Toast.LENGTH_LONG).show()
    }

    private fun exitCalibration() {
        calibMode = false
        calibOverlay?.visibility = View.GONE
        debugCircleView?.visibility = View.VISIBLE

        val t = calibToSrc()
        if (t == null) { applyDefaultRoi(); return }
        val (cx, cy, size) = t

        // apply the chosen square as the crop (equivalent to RATIO_X=1.0 = square crop)
        videoView.setFullFrame(false)
        videoView.setScaleMode(DEF_SCALE)
        videoView.setCrop(cx.toFloat(), cy.toFloat(), size)
        videoView.applyTransform()
        lastRoiCx = cx; lastRoiCy = cy; lastRoiSize = size

        val msg = "Applied (square) · center=($cx,$cy) size=$size"
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        Log.d(TAG, "CALIB $msg")
        updateDebugHud()
    }

    /** overlay square (view px) → source pixels (cx, cy, size) */
    private fun calibToSrc(): Triple<Int, Int, Int>? {
        val ov = calibOverlay ?: return null
        val rect = ov.contentRect()
        if (rect.width() <= 0f) return null
        val sw = if (srcW > 0) srcW else 1280
        val sh = if (srcH > 0) srcH else 720
        val s = rect.width() / sw                    // uniform scale
        if (s <= 0f) return null
        val cx = ((ov.centerX() - rect.left) / s).toInt().coerceIn(0, sw)
        val cy = ((ov.centerY() - rect.top) / s).toInt().coerceIn(0, sh)
        val size = (ov.sidePx() / s).toInt().coerceIn(16, minOf(sw, sh))
        return Triple(cx, cy, size)
    }

    private fun updateCalibReadout() {
        val t = calibToSrc() ?: return
        ensureDebugHud()
        debugHud?.visibility = View.VISIBLE
        debugHud?.text =
            "CALIB  CENTER=(${t.first},${t.second})  SIZE=${t.third}  · native ratio · long-press to apply"
    }

    /** the video display rect [left,top,right,bottom] when FITting source (sw,sh) into the view (vw,vh) */
    private fun fitContentRect(vw: Float, vh: Float, sw: Int, sh: Int): FloatArray {
        val s = minOf(vw / sw, vh / sh)
        val w = sw * s; val h = sh * s
        val l = (vw - w) / 2f; val t = (vh - h) / 2f
        return floatArrayOf(l, t, l + w, t + h)
    }

    // ===== volume keys: fine-tune PAR only in developer mode (keep if wanted) =====
    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (!devMode) return super.onKeyDown(keyCode, event)
        val step = 0.02f
        val rangeMin = 0.25f
        val rangeMax = 4.00f
        fun applyPar() {
            videoView.setPixelAspectFix(parX, parY)
            videoView.applyTransform()
            Toast.makeText(
                this,
                "PAR X=${"%.3f".format(parX)}  Y=${"%.3f".format(parY)}",
                Toast.LENGTH_SHORT
            ).show()
            updateDebugHud()
        }
        return when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP -> {
                parX = (parX + step).coerceIn(rangeMin, rangeMax)
                applyPar()
                true
            }
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                parX = (parX - step).coerceIn(rangeMin, rangeMax)
                applyPar()
                true
            }
            else -> super.onKeyDown(keyCode, event)
        }
    }

    // ===== Debug toggles init (safe if absent) =====
    private fun initDebugTogglesSafe() {
        val isDebuggable = (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        runCatching {
            val clazz = Class.forName("com.example.otoview.debug.DebugToggles")
            val method = clazz.getMethod("setDefaults", Boolean::class.javaPrimitiveType)
            method.invoke(null, isDebuggable)
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val REQ_WRITE_EXTERNAL = 1001
    }
}
