package com.example.otoview.gallery

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.otoview.R
import com.example.otoview.ai.AiAnalyzer
import com.example.otoview.bench.BenchLog
import kotlinx.coroutines.*

class ImageViewerActivity : ComponentActivity() {

    companion object {
        const val KEY_URI = "key_image_uri"
        /** Artificial wait so the loading indicator shows for at least one frame. Subtracted out in the benchmark. */
        private const val UI_LOADING_DELAY_MS = 80L
    }

    private var imageUri: Uri? = null

    // Top image + summary card
    private lateinit var imgPreview: ImageView
    private lateinit var summaryCard: View
    private lateinit var tvTopLabel: TextView
    private lateinit var tvTopProb: TextView
    private lateinit var tvAnalysisHint: TextView

    // Details
    private lateinit var btnToggleDetails: Button
    private lateinit var detailsContent: LinearLayout
    private lateinit var rvProbs: RecyclerView

    // Bottom buttons
    private lateinit var btnClose: Button
    private lateinit var btnAnalyze: Button

    private lateinit var analyzer: AiAnalyzer
    private lateinit var adapter: ProbAdapter

    private val uiScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_image_viewer)

        imgPreview       = findViewById(R.id.imgPreview)
        summaryCard      = findViewById(R.id.summaryCard)
        tvTopLabel       = findViewById(R.id.tvTopLabel)
        tvTopProb        = findViewById(R.id.tvTopProb)
        tvAnalysisHint   = findViewById(R.id.tvAnalysisHint)

        btnToggleDetails = findViewById(R.id.btnToggleDetails)
        detailsContent   = findViewById(R.id.detailsContent)
        rvProbs          = findViewById(R.id.rvProbs)

        btnClose         = findViewById(R.id.btnClose)
        btnAnalyze       = findViewById(R.id.btnAnalyze)

        // Initial state: make clear that analysis has not run yet
        tvTopLabel.text = "Not analyzed"
        tvTopProb.text  = ""
        tvAnalysisHint.text = "Waiting for analysis"
        tvAnalysisHint.visibility = View.VISIBLE

        detailsContent.visibility = View.GONE
        btnToggleDetails.text = "Show details"
        btnToggleDetails.isEnabled = false

        adapter = ProbAdapter()
        rvProbs.layoutManager = LinearLayoutManager(this)
        rvProbs.adapter = adapter

        analyzer = AiAnalyzer(applicationContext)

        imageUri = savedInstanceState?.getParcelable(KEY_URI)
            ?: intent?.data
                    ?: intent?.getParcelableExtra(KEY_URI)
                    ?: intent?.getParcelableExtra(Intent.EXTRA_STREAM)

        if (imageUri == null) {
            Toast.makeText(this, "There is no Image URI", Toast.LENGTH_SHORT).show()
            finish(); return
        }
        imgPreview.setImageURI(imageUri)

        btnAnalyze.setOnClickListener { runAnalysis() }
        btnClose.setOnClickListener { finish() }

        btnToggleDetails.setOnClickListener {
            val opening = detailsContent.visibility != View.VISIBLE
            detailsContent.visibility = if (opening) View.VISIBLE else View.GONE
            btnToggleDetails.text = if (opening) "Hide details" else "Show details"
            // Hide the Top-1 summary card when details open to free up space
            summaryCard.visibility = if (opening) View.GONE else View.VISIBLE
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        imageUri?.let { outState.putParcelable(KEY_URI, it) }
    }

    override fun onDestroy() {
        super.onDestroy(); uiScope.cancel()
    }

    private fun runAnalysis() {
        val uri = imageUri ?: return
        uiScope.launch {
            // From the moment the Analyze button is tapped until the result appears (user-perceived latency).
            // The delay(80) below is an artificial wait for the loading indicator, so it is recorded together.
            val tapNs = BenchLog.nowNs()
            try {
                setLoading(true)
                delay(UI_LOADING_DELAY_MS)
                val result = withContext(Dispatchers.Default) { analyzer.analyze(uri) }

                // Update Top-1
                tvTopLabel.text = result.topLabel
                tvTopProb.text  = "(${(result.topProb * 100f).format1()}%)"
                tvAnalysisHint.visibility = View.GONE
                BenchLog.since(
                    "analyze_tap_to_result", tapNs,
                    mapOf("ui_delay_ms" to UI_LOADING_DELAY_MS)
                )

                // Update list
                val labels = analyzer.labelsList()
                val items = result.probs.indices
                    .sortedByDescending { i -> result.probs[i] }
                    .map { i -> ProbRow(labels.getOrElse(i) { "CLASS $i" }, result.probs[i]*100f) }
                adapter.submit(items)

                // Enable the details button (closed by default)
                btnToggleDetails.isEnabled = true
                btnToggleDetails.text = "Show details"
                detailsContent.visibility = View.GONE
                summaryCard.visibility = View.VISIBLE

            } catch (e: Throwable) {
                Toast.makeText(this@ImageViewerActivity, "AI error: ${e.message}", Toast.LENGTH_SHORT).show()
                tvTopLabel.text = "Not analyzed"
                tvTopProb.text  = ""
                tvAnalysisHint.visibility = View.VISIBLE
                tvAnalysisHint.text = "Waiting for analysis"
                btnToggleDetails.isEnabled = false
                detailsContent.visibility = View.GONE
                summaryCard.visibility = View.VISIBLE
            } finally {
                setLoading(false)
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        btnAnalyze.isEnabled = !loading
        btnAnalyze.text = if (loading) "Analyzing…" else "Analyze"
        if (loading) {
            tvAnalysisHint.visibility = View.VISIBLE
            tvAnalysisHint.text = "Analyzing…"
        }
    }

    private fun Float.format1(): String = String.format("%.1f", this)
}
