package com.example.otoview.gallery

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import androidx.constraintlayout.widget.Guideline
import com.example.otoview.R

data class ProbRow(val label: String, val percent: Float) // 0~100

class ProbAdapter : RecyclerView.Adapter<ProbAdapter.VH>() {

    private val items = mutableListOf<ProbRow>()

    fun submit(newItems: List<ProbRow>) {
        items.clear()
        items.addAll(newItems)
        notifyDataSetChanged()
    }

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val tvLabel: TextView   = v.findViewById(R.id.tvLabel)
        val tvPercent: TextView = v.findViewById(R.id.tvPercent)
        val glPercent: Guideline = v.findViewById(R.id.glPercent)
        val barFill: View       = v.findViewById(R.id.barFill)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context).inflate(R.layout.item_prob_row, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val it = items[position]
        holder.tvLabel.text = it.label
        holder.tvPercent.text = "${String.format("%.1f", it.percent)}%"
        // convert 0~100 → 0.0~1.0 to set the guideline position
        val p = (it.percent / 100f).coerceIn(0f, 1f)
        holder.glPercent.setGuidelinePercent(p)

        // top-1 (first row) uses the cyan accent, the rest are dimmed (design spec)
        val ctx = holder.itemView.context
        val isTop = position == 0
        holder.tvLabel.setTextColor(
            ContextCompat.getColor(ctx, if (isTop) R.color.text_secondary else R.color.muted)
        )
        holder.tvPercent.setTextColor(
            ContextCompat.getColor(ctx, if (isTop) R.color.text_primary else R.color.muted)
        )
        holder.barFill.setBackgroundResource(
            if (isTop) R.drawable.prob_progress else R.drawable.prob_progress_dim
        )
    }

    override fun getItemCount(): Int = items.size
}
