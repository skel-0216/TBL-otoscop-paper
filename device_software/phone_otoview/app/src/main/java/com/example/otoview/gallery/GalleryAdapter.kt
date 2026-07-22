package com.example.otoview.gallery

import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.otoview.R

class GalleryAdapter(
    private val onClick: (Uri) -> Unit,
    private val onLongClick: (Uri) -> Unit
) : ListAdapter<Uri, GalleryAdapter.VH>(Diff) {

    object Diff : DiffUtil.ItemCallback<Uri>() {
        override fun areItemsTheSame(oldItem: Uri, newItem: Uri) = oldItem == newItem
        override fun areContentsTheSame(oldItem: Uri, newItem: Uri) = true
    }

    inner class VH(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val img: ImageView = itemView.findViewById(R.id.imageThumb)
        fun bind(uri: Uri) {
            img.setImageURI(uri) // simple render (swap for Glide etc. if needed)
            itemView.setOnClickListener { onClick(uri) }
            itemView.setOnLongClickListener {
                onLongClick(uri)
                true
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val v = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_image, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        holder.bind(getItem(position))
    }
}
