package com.example.glasstcptest

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.StrictMode
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.WindowManager
import android.widget.FrameLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.isVisible
import com.example.glasstcptest.bench.BenchLog
import com.example.glasstcptest.bench.ResourceSampler
import com.example.glasstcptest.databinding.ActivityMainBinding
import com.example.testforglassguesture.GlassGestureDetector
import java.io.IOException
import java.io.PrintWriter
import java.net.Socket


class MainActivity : AppCompatActivity(), GlassGestureDetector.OnGestureListener  {
    private val binding by lazy { ActivityMainBinding.inflate(layoutInflater) }

    companion object {
        const val FEATURE_VOICE_COMMANDS = 14
        const val REQUEST_PERMISSION_CODE = 200
        val PERMISSIONS = arrayOf(Manifest.permission.RECORD_AUDIO)
    }

    private lateinit var glassGestureDetector: GlassGestureDetector

    lateinit var frameLayout: FrameLayout
    private lateinit var videoFragment: VideoFragment

    // for command socket
    private lateinit var commandSocket : Socket
    private val SERVERIP_COMMAND = IpAddress.ipAddress
    private val SERVERPORT_COMMAND = IpAddress.port_command
    private lateinit var threadCommand: Thread

    private var STATE_QUIT = false
    private val COMMAND_QUIT = "command_quit"
    private val COMMAND_LIGHT_ON = "command_light_on"
    private val COMMAND_LIGHT_OFF = "command_light_off"

    private val modelProcessor by lazy { ModelProcessor(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.requestFeature(FEATURE_VOICE_COMMANDS)
        setContentView(binding.root)

        Log.d(":::DEBUG:::", "Main init")
        glassGestureDetector = GlassGestureDetector(this, this)

        // Requesting permissions to enable voice commands menu
        ActivityCompat.requestPermissions(
            this,
            PERMISSIONS,
            REQUEST_PERMISSION_CODE
        )
        // FLAG_KEEP_SCREEN_ON: keeps the screen from dimming or turning off while streaming
        // is being viewed, even without touch. Fixes the screen dimming during real use,
        // when the hands are not free while looking through the otoscope. It applies only
        // while the activity is on screen and clears automatically when it goes to the
        // background, so no separate wake lock or permission is needed.
        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN or
                    WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            WindowManager.LayoutParams.FLAG_FULLSCREEN or
                    WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        )

        val policy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)

        try{
            commandSocket = Socket(SERVERIP_COMMAND, SERVERPORT_COMMAND)
        } catch (e : IOException) {
            e.printStackTrace()
        }

        frameLayout = binding.frameLayoutVideo

        BenchLog.init(applicationContext)
        startBenchSession()

        // Preload the model in the background at startup so the first capture does not carry
        // the model load (about 1.7 s on XR1). Inference itself is hardware-bound and does not
        // shrink, but the first capture feels much faster.
        Thread { modelProcessor.labelsList() }.start()
    }

    // ===== Deployment benchmark (only active in debuggable builds) =====
    private var resourceSampler: ResourceSampler? = null
    private var benchRunNo = 0

    private fun startBenchSession() {
        if (!BenchLog.enabled) return
        benchRunNo++
        BenchLog.startSession("run$benchRunNo")
        resourceSampler = ResourceSampler(this).also { it.start() }
        Log.d("bench", "session run$benchRunNo started")
    }

    private fun exportBenchSession() {
        if (!BenchLog.enabled || !BenchLog.hasSession()) return
        resourceSampler?.stop()
        resourceSampler = null
        val f = BenchLog.export(this)
        Log.d("bench", "exported ${f?.absolutePath}")
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_PERMISSION_CODE) {
            for (result in grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    Log.d("DEBUG ::: VoiceCommandsActivity", "Permission denied. Voice commands menu is disabled.")
                }
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    override fun onCreatePanelMenu(featureId: Int, menu: Menu): Boolean {
        menuInflater.inflate(R.menu.voice_commands_menu, menu)
        return true
    }

    override fun onContextItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            // Handle selected menu item
            R.id.capture, R.id.analyze, R.id.scan, R.id.takePicture, R.id.takeAPicture, R.id.process -> {
                Log.d("Voice Recognition", "capture called")
                capture("voice")
                true
            }
            R.id.quit, R.id.exit, R.id.terminate -> {
                Log.d("Voice Recognition", "Quit called")
                finish()
                true
            }
            R.id.video, R.id.back, R.id.goBack, R.id.goMain, R.id.goToMain -> {
                Log.d("Voice Recognition", "video called")
                hideCaptureShowTextureView()
                true
            }
            R.id.lightOn -> {
                Log.d("Voice Recognition", "light on called")
                ledControl(COMMAND_LIGHT_ON)
                true
            }
            R.id.lightOff -> {
                Log.d("Voice Recognition", "light off called")
                ledControl(COMMAND_LIGHT_OFF)
                true
            }
            else -> {
                Log.d("Voice Recognition", "else called")
                super.onContextItemSelected(item)
            }
        }
    }

    override fun onStart() {
        super.onStart()

        videoFragment = VideoFragment.newInstance()

        val fragTran = supportFragmentManager.beginTransaction()
        fragTran.add(R.id.frameLayout_video, videoFragment)
        fragTran.commit()
    }

    override fun onStop(){
        super.onStop()

        Log.d("main-onStop", "onStop called")

        exportBenchSession()
        quitStreaming()
        // stop socket send to server client
    }

    // This Glass EE2 sends touchpad gestures as key events, not MotionEvents:
    //   tap = DPAD_CENTER, swipe forward = TAB, swipe back = Shift+TAB, swipe down = BACK.
    // So GlassGestureDetector, which handles MotionEvents, did not catch them. Handle the keys directly.
    // (The dispatchTouchEvent path is kept too, so devices/emulators that deliver touch still work.)
    override fun dispatchTouchEvent(ev: MotionEvent?): Boolean {
        return ev?.let { glassGestureDetector.onTouchEvent(it) } == true || super.dispatchTouchEvent(ev)
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_TAB -> {
                if (event?.isShiftPressed == true) {
                    hideCaptureShowTextureView()   // swipe back: return to video
                } else {
                    capture("swipe_forward")       // swipe forward: capture
                }
                return true
            }
            KeyEvent.KEYCODE_DPAD_CENTER, KeyEvent.KEYCODE_ENTER -> {
                capture("tap")                     // tap: capture (easiest when hands are not free)
                return true
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onGesture(gesture: GlassGestureDetector.Gesture): Boolean {
        return when (gesture) {
            // tap gesture has many problems.
            GlassGestureDetector.Gesture.TAP -> {
                Log.d("DEBUG ::: onGesture", "Gesture.Tap is run")
                // Instrumented builds only: save the session so far and start the next run.
                if (BenchLog.enabled) {
                    exportBenchSession()
                    startBenchSession()
                }
                true
            }

            GlassGestureDetector.Gesture.SWIPE_DOWN -> {
                Log.d("DEBUG ::: onGesture", "Gesture.SWIPE_DOWN is run")
                finish()
                true
            }

            GlassGestureDetector.Gesture.SWIPE_FORWARD -> {
                Log.d("DEBUG ::: onGesture", "Gesture.SWIPE_FORWARD is run")
                capture("gesture")
                true
            }

            GlassGestureDetector.Gesture.SWIPE_BACKWARD -> {
                Log.d("DEBUG ::: onGesture", "Gesture.SWIPE_BACKWARD is run")
                hideCaptureShowTextureView()
                true
            }

            else -> {
                Log.d("DEBUG ::: onGesture", "else was run")
                false
            }
        }
    }

    override fun onScroll(
        e1: MotionEvent,
        e2: MotionEvent,
        distanceX: Float,
        distanceY: Float
    ): Boolean {
        return true
    }

    override fun onTouchEnded() { }

    private fun ledControl(command : String) {
        threadCommand = Thread {
            val printWriter = PrintWriter(commandSocket.getOutputStream()) // plain socket for button comms
            printWriter.flush()
            when(command) {
                COMMAND_LIGHT_ON -> printWriter.write(COMMAND_LIGHT_ON)
                COMMAND_LIGHT_OFF -> printWriter.write(COMMAND_LIGHT_OFF)
            }
            printWriter.flush()
        }
        threadCommand.start()
    }

    private fun quitStreaming() {
        threadCommand = Thread {
            while (!STATE_QUIT) {
                try {
                    val printWriter = PrintWriter(commandSocket.getOutputStream()) // plain socket for button comms
                    printWriter.write(COMMAND_QUIT)
                    printWriter.flush()
                    STATE_QUIT = true
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
            STATE_QUIT = false
        }
        threadCommand.start()
    }

    /**
     * [trigger] is "voice" or "gesture". The time spent on voice recognition itself is
     * handled by the Glass OS and is not measured. What is measured here is the time after
     * the command reaches the app.
     */
    @SuppressLint("DiscouragedApi")
    private fun capture(trigger: String) {
        Log.d("capture", "capture called")
        val triggerNs = BenchLog.nowNs()

        val fullFrame = videoFragment.takeSnapshot()

        binding.imageViewMain.setImageBitmap(fullFrame)
        binding.frameLayoutVideo.isVisible = false
        binding.imageViewMain.isVisible = true
        binding.captureBorder.setBackgroundResource(R.drawable.border_capture_analyzing)
        binding.captureBorder.isVisible = true
        binding.textView.text = getString(R.string.analyzing)
        binding.textView.isVisible = true

        Thread {
            try {
                val aiInput = BenchLog.time("roi_crop") { cropRoiForAI(fullFrame) }
                val result = modelProcessor.analyze(aiInput)
                Log.d("capture", "result=${result.topLabel} ${result.topProb}")
                runOnUiThread {
                    binding.captureBorder.setBackgroundResource(R.drawable.border_capture_done)
                    binding.textView.text =
                        "${result.topLabel}  ${"%.1f".format(result.topProb * 100f)}%"
                    // from command received to result shown on screen (the latency the user feels)
                    BenchLog.since("capture_e2e", triggerNs, mapOf("trigger" to trigger))
                }
            } catch (e: Throwable) {
                Log.e("capture", "classify failed", e)
                BenchLog.event("capture_failed", null, mapOf("trigger" to trigger))
                runOnUiThread { binding.textView.text = getString(R.string.analyze_failed) }
            }
        }.start()
    }

    private fun hideCaptureShowTextureView() {
        runOnUiThread {
            binding.imageViewMain.isVisible = false
            binding.imageViewMain.setImageResource(0)
            binding.captureBorder.isVisible = false
            binding.frameLayoutVideo.isVisible = true
            binding.textView.isVisible = false
        }
    }

    private fun cropRoiForAI(full: Bitmap): Bitmap {
        val refW = 1280f
        val refH = 720f
        val roiCenterX = 605f
        val roiWidth = 726f
        val roiTop = 0f
        val roiHeight = 633f

        val sx = full.width / refW
        val sy = full.height / refH

        var left = ((roiCenterX - roiWidth / 2f) * sx).toInt().coerceAtLeast(0)
        var top = (roiTop * sy).toInt().coerceAtLeast(0)
        var w = (roiWidth * sx).toInt()
        var h = (roiHeight * sy).toInt()
        if (left + w > full.width) w = full.width - left
        if (top + h > full.height) h = full.height - top
        w = w.coerceAtLeast(1)
        h = h.coerceAtLeast(1)

        val roi = Bitmap.createBitmap(full, left, top, w, h)
        return Bitmap.createScaledBitmap(roi, 256, 256, true)
    }
}
