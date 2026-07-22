package com.example.glasstcptest

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaCodec
import android.media.MediaFormat
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.LayoutInflater
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.glasstcptest.bench.BenchLog
import com.example.glasstcptest.databinding.FragmentVideoBinding
import java.nio.ByteBuffer
import kotlin.experimental.and

class VideoFragment : Fragment(), TextureView.SurfaceTextureListener {
    private val binding by lazy { FragmentVideoBinding.inflate(layoutInflater) }

    companion object {
        fun newInstance(): VideoFragment {
            return VideoFragment()
        }
    }

    // instance variables
    private lateinit var updCamera: CameraData
    private lateinit var decoderThread: DecoderThread
    private lateinit var textureView: StreamingView
    private lateinit var messageView: TextView
    private var finishHandler = Handler()
    private var startVideoHandler = Handler()
    private lateinit var finishRunner: Runnable
    private lateinit var startVideoRunner: Runnable
    private var networkname = "Endoscope"
    private var devicename = "Endoscope_device"
    private var ipaddress = IpAddress.ipAddress
    private var port = IpAddress.port_stream

    private var width = 0
    private var height = 0

    lateinit var image : Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // create finish handler and runnable
        finishHandler = Handler()
        finishRunner = Runnable {
            requireActivity().finish()
        }

        // create start video handler and runnable
        startVideoHandler = Handler()
        startVideoRunner = Runnable {
            val format = decoderThread.getMediaFormat()
            val videoWidth = format?.getInteger(MediaFormat.KEY_WIDTH)
            val videoHeight = format?.getInteger(MediaFormat.KEY_HEIGHT)
            width = videoWidth!!
            height = videoHeight!!
            textureView.setSourceSize(videoWidth, videoHeight)
        }
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?,
                              savedInstanceState: Bundle?): View {

        // configure Camera Date
        updCamera = CameraData(networkname, devicename, ipaddress, port)

        // configure the name
        // initialize the message
        // sets the initial message shown
        Log.d("DEBUG", "done-init")
        messageView = binding.videoMessage
        messageView.setTextColor(ContextCompat.getColor(requireContext(), R.color.white))
        messageView.text = getString(R.string.initializing_video)

        // set the texture listener
        // where the streaming video is shown
        textureView = binding.videoSurface
        textureView.surfaceTextureListener = this
        textureView.setScaleMode(StreamingView.ScaleMode.FIT)


        // Inflate the layout for this fragment
        return binding.root
    }

    override fun onDestroy() {
        super.onDestroy()
        finishHandler.removeCallbacks(finishRunner)
    }

    override fun onStart() {
        super.onStart()
        decoderThread = DecoderThread()
        decoderThread.start()
    }

    override fun onStop() {
        super.onStop()
        decoderThread.interrupt()
    }

    override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
        decoderThread.setSurface(Surface(surface), startVideoHandler, startVideoRunner)
    }

    override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) { }

    override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
        // MyEdit WARNNING
        decoderThread.setSurface(null, null, null)
        return true
    }

    override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
        // Display-side frame metering. The decoder emits in bursts, so this is the frame actually seen.
        BenchLog.display.onFrameRendered()
    }

    fun sourceWidth(): Int = if (width > 0) width else 1280
    fun sourceHeight(): Int = if (height > 0) height else 720

    fun takeSnapshot(): Bitmap {
        image = BenchLog.time("snapshot_grab") {
            textureView.getBitmap(sourceWidth(), sourceHeight())!!
        }
        return image
    }

    private inner class DecoderThread : Thread() {
        // local constants
        private val FINISH_TIMEOUT = 5000
        private val BUFFER_SIZE = 16384
        private val NAL_SIZE_INC = 4096
        private val MAX_READ_ERRORS = 300

        // instance variables
        private var mediadecoder: MediaCodec? = null
        private var format: MediaFormat? = null
        private var decoding = false
        private var surface: Surface? = null
        private var buffer: ByteArray? = null
        private var inputBuffers: Array<ByteBuffer>? = null
        private var presentationTime: Long = 0
        private var presentationTimeInc: Long = 66666
        private var reader: TcpIpReader? = null
        // Stream startup latency: from decoder thread start to first rendered frame
        private var threadStartNs: Long = 0
        private var firstFrameLogged = false
        private lateinit var startVideoHandler: Handler
        private lateinit var startVideoRunner: Runnable

        //******************************************************************************
        // setSurface
        //******************************************************************************
        fun setSurface(surface: Surface?, handler: Handler?, runner: Runnable?) {
            this.surface = surface
            this.startVideoHandler = handler ?: Handler() // if handler is null, create an empty Handler
            this.startVideoRunner = runner ?: Runnable {} // if runner is null, create an empty Runnable
            if (mediadecoder != null) {
                if (surface != null) {
                    var newDecoding = decoding
                    if (decoding) {
                        setDecodingState(false)
                    }
                    if (format != null) {
                        try {
                            mediadecoder!!.configure(format!!, surface, null, 0)
                        } catch (_: Exception) { }
                        if (!newDecoding) {
                            newDecoding = true
                        }
                    }
                    if (newDecoding) {
                        setDecodingState(newDecoding)
                    }
                } else if (decoding) {
                    setDecodingState(false)
                }
            }
        }

        //******************************************************************************
        // getMediaFormat
        //******************************************************************************
        fun getMediaFormat(): MediaFormat? {
            Log.d(":::DEBUG:::", "Format : $format")
            return format
        }

        //******************************************************************************
        // setDecodingState
        //******************************************************************************
        @Synchronized
        private fun setDecodingState(newDecoding: Boolean) {
            try {
                if (newDecoding != decoding && mediadecoder != null) {
                    if (newDecoding) {
                        mediadecoder!!.start()
                    } else {
                        mediadecoder!!.stop()
                    }
                    decoding = newDecoding
                }
            } catch (_: Exception) {
            }
        }

        //******************************************************************************
        // run
        //******************************************************************************
        override fun run() {
            var nal = ByteArray(NAL_SIZE_INC)
            var nalLen = 0
            var numZeroes = 0
            var numReadErrors = 0
            threadStartNs = BenchLog.nowNs()
            firstFrameLogged = false
            try {
                // create the decoder
                mediadecoder = MediaCodec.createDecoderByType("video/avc")

                // create the reader
                buffer = ByteArray(BUFFER_SIZE)
                // TcpIpReader Class is used here!
                reader = BenchLog.time("stream_connect") { TcpIpReader(updCamera) }
                if (!reader!!.isConnected()) {
                    throw Exception()
                }

                // read until we're interrupted
                while (!isInterrupted) {
                    // read from the stream
//                    Log.d("DEBUG::", "read from the stream")
                    val len = reader!!.read(buffer!!)
                    if (isInterrupted) break

                    // process the input buffer
                    if (len > 0) {
                        // when incoming data is not 0!
                        numReadErrors = 0
                        var i = 0
                        while (i < len && !isInterrupted) {
                            // add the byte to the NAL
                            if (nalLen == nal.size) {
                                nal = nal.copyOf(nal.size + NAL_SIZE_INC)
                            }
                            nal[nalLen++] = buffer!![i]

                            // look for a header
                            if (buffer!![i] == 0.toByte()) {
                                numZeroes++
                            } else {
                                if (buffer!![i] == 1.toByte() && numZeroes == 3) {
                                    if (nalLen > 4) {
                                        val nalType = processNal(nal, nalLen - 4)
                                        if (isInterrupted) break
                                        if (nalType == -1) {
                                            nal[0] = 0
                                            nal[1] = 0
                                            nal[2] = 0
                                            nal[3] = 1
                                        }
                                    }
                                    nalLen = 4
                                }
                                numZeroes = 0
                            }
                            i++
                        }
                    } else {
                        numReadErrors++
                        if (numReadErrors >= MAX_READ_ERRORS) {
                            setMessage(R.string.lost_the_connection_to_the_camera)
                            break
                        }
                    }

                    // send an output buffer to the surface
                    if (format != null && decoding) {
                        if (isInterrupted) break
                        val info = MediaCodec.BufferInfo()
                        var index: Int
                        do {
                            index = mediadecoder!!.dequeueOutputBuffer(info, 0)
                            if (isInterrupted) break
                            if (index >= 0) {
                                mediadecoder!!.releaseOutputBuffer(index, true)
                                // Count only the frames that actually go to the screen.
                                BenchLog.frames.onFrameRendered()
                                if (!firstFrameLogged) {
                                    firstFrameLogged = true
                                    BenchLog.since("stream_first_frame", threadStartNs)
                                }
                            }
                            //Log.info(String.format("dequeueOutputBuffer index = %d", index));
                        } while (index >= 0)
                    }
                }
            } catch (ex: Exception) {
                if (reader == null || !reader!!.isConnected()) {
                    setMessage(R.string.error_couldnt_connect)
                } else {
                    setMessage(R.string.error_lost_connection)
                    Log.d("ERROR LOST THE CONNECTION 2", "$reader, ${reader!!.isConnected()}")
                }
                ex.printStackTrace()
            }

            // close the reader
            if (reader != null) {
                try {
                    reader!!.close()
                } catch (_: Exception) {
                }
                reader = null
            }

            // stop the decoder
            if (mediadecoder != null) {
                try {
                    setDecodingState(false)
                    mediadecoder!!.release()
                } catch (_: Exception) {
                }
                mediadecoder = null
            }
        }

        //******************************************************************************
        // processNal
        //******************************************************************************
        private fun processNal(nal: ByteArray, nalLen: Int): Int {
            // get the NAL type
            val nalType = if (nalLen > 4 && nal[0] == 0.toByte() && nal[1] == 0.toByte() && nal[2] == 0.toByte() && nal[3] == 1.toByte()) {
                (nal[4] and 0x1F).toInt()
            } else {
                -1
            }
            //Log.info(String.format("NAL: type = %d, len = %d", nalType, nalLen));

            // process the first SPS record we encounter
            if (nalType == 7 && !decoding) {
                // SpsParser Class is used here! ★★★★★★★★★★★★★★★★★★★★
                // SpsReader Class is inside SpsParser
                val parser = SpsParser(nal, nalLen)
                format = MediaFormat.createVideoFormat("video/avc", parser.width, parser.height)
                presentationTimeInc = 66666
                presentationTime = System.nanoTime() / 1000
                // Log.info(String.format("SPS: %02X, %d x %d, %d", nal[4], parser.width, parser.height, presentationTimeInc));
                mediadecoder!!.configure(format!!, surface, null, 0)
                setDecodingState(true)
                inputBuffers = mediadecoder!!.inputBuffers
                hideMessage()
                startVideoHandler.post(startVideoRunner)
            }

            // queue the frame
            if (nalType > 0 && decoding) {
                val index = mediadecoder!!.dequeueInputBuffer(0)
                if (index >= 0) {
                    val inputBuffer = inputBuffers!![index]
                    //ByteBuffer inputBuffer = decoder.getInputBuffer(index);
                    inputBuffer.put(nal, 0, nalLen)
                    mediadecoder!!.queueInputBuffer(index, 0, nalLen, presentationTime, 0)
                    presentationTime += presentationTimeInc
                }
                //Log.info(String.format("dequeueInputBuffer index = %d", index));
            }
            return nalType
        }

        //******************************************************************************
        // hideMessage
        //******************************************************************************
        private fun hideMessage() {
            // remove the shown message once streaming starts
            requireActivity().runOnUiThread {
                messageView.visibility = View.GONE
            }
        }

        //******************************************************************************
        // setMessage
        //******************************************************************************
        private fun setMessage(id: Int) {
            requireActivity().runOnUiThread {
                messageView.setText(id)
                messageView.setTextColor(ContextCompat.getColor(requireContext(), R.color.white))
                messageView.visibility = View.VISIBLE
            }
        }
    }
}