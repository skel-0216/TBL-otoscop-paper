package com.example.glasstcptest

class SpsParser(nal: ByteArray, len: Int) {
    // local constants
    private val TAG = "SpsParser"

    // instance variables
    private var reader: SpsReader
    var width = 0
    var height = 0
    var fps = 0f
    var num_units_in_tick = 0
    var time_scale = 0

    init {
        // SpsReader Class is used here
        reader = SpsReader(nal, len)
        reader.skipBits(if (nal[2] == 0.toByte()) 40 else 36)

        var frame_crop_left_offset = 0
        var frame_crop_right_offset = 0
        var frame_crop_top_offset = 0
        var frame_crop_bottom_offset = 0

        val profile_idc = reader.readBits(8)
        val constraint_set0_flag = reader.readBit()
        val constraint_set1_flag = reader.readBit()
        val constraint_set2_flag = reader.readBit()
        val constraint_set3_flag = reader.readBit()
        val constraint_set4_flag = reader.readBit()
        val constraint_set5_flag = reader.readBit()
        val reserved_zero_2bits = reader.readBits(2)
        val level_idc = reader.readBits(8)
        val seq_parameter_set_id = reader.readExpGolombCode()

        if (profile_idc == 100 || profile_idc == 110 ||
            profile_idc == 122 || profile_idc == 244 ||
            profile_idc == 44 || profile_idc == 83 ||
            profile_idc == 86 || profile_idc == 118
        ) {
            val chroma_format_idc = reader.readExpGolombCode()

            if (chroma_format_idc == 3) {
                val residual_colour_transform_flag = reader.readBit()
            }
            val bit_depth_luma_minus8 = reader.readExpGolombCode()
            val bit_depth_chroma_minus8 = reader.readExpGolombCode()
            val qpprime_y_zero_transform_bypass_flag = reader.readBit()
            val seq_scaling_matrix_present_flag = reader.readBit()

            if (seq_scaling_matrix_present_flag != 0) {
                var i = 0
                while (i < 8) {
                    val seq_scaling_list_present_flag = reader.readBit()
                    if (seq_scaling_list_present_flag != 0) {
                        val sizeOfScalingList = if (i < 6) 16 else 64
                        var lastScale = 8
                        var nextScale = 8
                        var j = 0
                        while (j < sizeOfScalingList) {
                            if (nextScale != 0) {
                                val delta_scale = reader.readSignedExpGolombCode()
                                nextScale = (lastScale + delta_scale + 256) % 256
                            }
                            lastScale = if (nextScale == 0) lastScale else nextScale
                            j++
                        }
                    }
                    i++
                }
            }
        }

        val log2_max_frame_num_minus4 = reader.readExpGolombCode()
        val pic_order_cnt_type = reader.readExpGolombCode()
        if (pic_order_cnt_type == 0) {
            val log2_max_pic_order_cnt_lsb_minus4 = reader.readExpGolombCode()
        } else if (pic_order_cnt_type == 1) {
            val delta_pic_order_always_zero_flag = reader.readBit()
            val offset_for_non_ref_pic = reader.readSignedExpGolombCode()
            val offset_for_top_to_bottom_field = reader.readSignedExpGolombCode()
            val num_ref_frames_in_pic_order_cnt_cycle = reader.readExpGolombCode()
            var i = 0
            while (i < num_ref_frames_in_pic_order_cnt_cycle) {
                reader.readSignedExpGolombCode()
                i++
            }
        }
        val max_num_ref_frames = reader.readExpGolombCode()
        val gaps_in_frame_num_value_allowed_flag = reader.readBit()
        val pic_width_in_mbs_minus1 = reader.readExpGolombCode()
        val pic_height_in_map_units_minus1 = reader.readExpGolombCode()
        val frame_mbs_only_flag = reader.readBit()
        if (frame_mbs_only_flag == 0) {
            val mb_adaptive_frame_field_flag = reader.readBit()
        }
        val direct_8x8_inference_flag = reader.readBit()
        val frame_cropping_flag = reader.readBit()
        if (frame_cropping_flag != 0) {
            frame_crop_left_offset = reader.readExpGolombCode()
            frame_crop_right_offset = reader.readExpGolombCode()
            frame_crop_top_offset = reader.readExpGolombCode()
            frame_crop_bottom_offset = reader.readExpGolombCode()
        }
        val vui_parameters_present_flag = reader.readBit()
        if (vui_parameters_present_flag != 0) {
            val aspect_ratio_info_present_flag = reader.readBit()
            if (aspect_ratio_info_present_flag != 0) {
                val aspect_ratio = reader.readBits(8)
            }
            val overscan_info_present_flag = reader.readBit()
            if (overscan_info_present_flag != 0) {
                val overscan_appropriate_flag = reader.readBit()
            }
            val video_signal_type_present_flag = reader.readBit()
            if (video_signal_type_present_flag != 0) {
                val video_format = reader.readBits(3)
                val video_full_range_flag = reader.readBit()
                val colour_description_present_flag = reader.readBit()
                if (colour_description_present_flag != 0) {
                    val colour_primaries = reader.readBits(8)
                    val transfer_characteristics = reader.readBits(8)
                    val matrix_coefficients = reader.readBits(8)
                }
            }
            val chroma_loc_info_present_flag = reader.readBit()
            if (chroma_loc_info_present_flag != 0) {
                val chroma_sample_loc_type_top_field = reader.readExpGolombCode()
                val chroma_sample_loc_type_bottom_field = reader.readExpGolombCode()
            }
            val timing_info_present_flag = reader.readBit()
            if (timing_info_present_flag != 0) {
                num_units_in_tick = reader.readBits(32)
                time_scale = reader.readBits(32)
                val fixed_frame_rate_flag = reader.readBit()
            }
            val nal_hrd_parameters_present_flag = reader.readBit()
            if (nal_hrd_parameters_present_flag != 0) {
                readHRDParameters()
            }
            val vcl_hrd_parameters_present_flag = reader.readBit()
            if (vcl_hrd_parameters_present_flag != 0) {
                readHRDParameters()
            }
            if (nal_hrd_parameters_present_flag != 0 || vcl_hrd_parameters_present_flag != 0) {
                val low_delay_hrd_flag = reader.readBit()
            }
            val pic_struct_present_flag = reader.readBit()
            val bitstream_restriction_flag = reader.readBit()
            if (bitstream_restriction_flag != 0) {
                val motion_vectors_over_pic_boundaries_flag = reader.readBit()
                val max_bytes_per_pic_denom = reader.readExpGolombCode()
                val max_bits_per_mb_denom = reader.readExpGolombCode()
                val log2_max_mv_length_horizontal = reader.readExpGolombCode()
                val log2_max_mv_length_vertical = reader.readExpGolombCode()
                val num_reorder_frames = reader.readExpGolombCode()
                val max_dec_frame_buffering = reader.readExpGolombCode()
            }
        }

        width = ((pic_width_in_mbs_minus1 + 1) * 16) - frame_crop_bottom_offset * 2 - frame_crop_top_offset * 2
        height =
            ((2 - frame_mbs_only_flag) * (pic_height_in_map_units_minus1 + 1) * 16) - (frame_crop_right_offset * 2) - (frame_crop_left_offset * 2)
        if (time_scale != 0) {
            fps = num_units_in_tick.toFloat() / time_scale.toFloat() / 2
        }
    }

    private fun readHRDParameters() {
        val cpb_cnt_minus1 = reader.readExpGolombCode()
        val bit_rate_scale = reader.readBits(4)
        val cpb_size_scale = reader.readBits(4)

        for (SchedSelIdx in 0..cpb_cnt_minus1) {
            reader.readExpGolombCode()
            reader.readExpGolombCode()
            reader.readBit()
        }
        val initial_cpb_removal_delay_length_minus1 = reader.readBits(5)
        val cpb_removal_delay_length_minus1 = reader.readBits(5)
        val dpb_output_delay_length_minus1 = reader.readBits(5)
        val time_offset_length = reader.readBits(5)
    }
}

