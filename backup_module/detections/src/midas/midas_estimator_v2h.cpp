/*******************************************************************************
 * midas_estimator_v2h.cpp
 * MiDaS depth estimator using DRP-AI TVM runtime (Renesas RZ/V2H)
 * Core inference logic adapted from app_midas_cam/src/main_midas.cpp
 ******************************************************************************/

#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)

#include <detections/midas/midas_estimator_v2h.h>
#include <detections/drp_queue.h>
#include <detections/ai.h>
#include <logger/logger.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>

/*---------------------------------------------------------------------------*/
/* Constructor / Destructor                                                   */
/*---------------------------------------------------------------------------*/

MidasEstimatorV2H::MidasEstimatorV2H()
{
    /* Allocate DMA buffer for one camera frame (WxH, BGR) */
    drpai_buf = static_cast<dma_buffer*>(malloc(sizeof(dma_buffer)));
    if (!drpai_buf) {
        LOGR_ERROR("MidasEstimatorV2H: Failed to allocate dma_buffer struct");
        return;
    }

    int ret = buffer_alloc_dmabuf(drpai_buf, IMAGE_WIDTH * IMAGE_HEIGHT * 2);
    if (ret == -1) {
        LOGR_ERROR("MidasEstimatorV2H: Failed to allocate DMA buffer");
        free(drpai_buf);
        drpai_buf = nullptr;
    }
}

MidasEstimatorV2H::~MidasEstimatorV2H()
{
    if (drpai_buf) {
        free(drpai_buf);
        drpai_buf = nullptr;
    }
}

/*---------------------------------------------------------------------------*/
/* Private helpers                                                            */
/*---------------------------------------------------------------------------*/

float MidasEstimatorV2H::float16_to_float32(uint16_t a)
{
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

int8_t MidasEstimatorV2H::get_result()
{
    int8_t  ret         = 0;
    int32_t output_num  = 0;
    int64_t output_size = 0;

    std::tuple<InOutDataType, void*, int64_t> output_buffer;

    output_num = runtime->GetNumOutput();
    if (output_num != 1) {
        fprintf(stderr, "[MiDaS V2H] ERROR: unexpected output count %d (expected 1)\n",
                output_num);
        return -1;
    }

    output_buffer = runtime->GetOutput(0);
    output_size   = std::get<2>(output_buffer);

    const int32_t expected_size =
        MIDAS_MODEL_OUT_W * MIDAS_MODEL_OUT_H * MIDAS_NUM_CLASS;

    if (output_size != expected_size) {
        fprintf(stderr,
                "[MiDaS V2H] ERROR: output size mismatch (%lld vs %d)\n",
                (long long)output_size, expected_size);
        return -1;
    }

    if (InOutDataType::FLOAT16 == std::get<0>(output_buffer)) {
        uint16_t* data_ptr =
            reinterpret_cast<uint16_t*>(std::get<1>(output_buffer));
        for (int j = 0; j < output_size; ++j) {
            drpai_output_buf[j] = float16_to_float32(data_ptr[j]);
        }
    } else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer)) {
        float* data_ptr =
            reinterpret_cast<float*>(std::get<1>(output_buffer));
        for (int j = 0; j < output_size; ++j) {
            drpai_output_buf[j] = data_ptr[j];
        }
    } else {
        fprintf(stderr, "[MiDaS V2H] ERROR: output data type not floating point\n");
        ret = -1;
    }

    return ret;
}

void MidasEstimatorV2H::R_Post_Proc_MiDaS(float* floatarr)
{
    /* Convert raw depth buffer (H x W x 1) to normalized float depth map */
    cv::Mat depth_org(MIDAS_MODEL_OUT_H, MIDAS_MODEL_OUT_W, CV_32FC1, floatarr);
    cv::Mat depth_norm;
    cv::normalize(depth_org, depth_norm, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);

    /* Store result — will be transferred out in inference() */
    depth_norm.copyTo(prev_prediction);  // temporary re-use; overwritten below
}

/*---------------------------------------------------------------------------*/
/* Public API                                                                 */
/*---------------------------------------------------------------------------*/

cv::Mat MidasEstimatorV2H::inference(const cv::Mat& frame)
{
    auto future = DRPQueue::get_instance().enqueue([this, &frame]() -> cv::Mat {
        if (!runtime || !preruntime || !drpai_buf) {
            return cv::Mat::zeros(MIDAS_MODEL_OUT_H, MIDAS_MODEL_OUT_W, CV_32F);
        }

        /* 1. Resize ke resolusi input DRP-AI (640x480) jika perlu */
        cv::Mat frame_for_drp;
        if (frame.cols != IMAGE_WIDTH || frame.rows != IMAGE_HEIGHT) {
            cv::resize(frame, frame_for_drp, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        } else {
            frame_for_drp = frame;
        }

        /* 2. Convert BGR → YUYV (Manual Packing) karena model dikompilasi untuk YUYV */
        cv::Mat yuv;
        cv::cvtColor(frame_for_drp, yuv, cv::COLOR_BGR2YUV); // Y, U, V
        
        cv::Mat frame_yuyv(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC2);
        for (int r = 0; r < IMAGE_HEIGHT; ++r) {
            const uint8_t* src = yuv.ptr<uint8_t>(r);
            uint8_t* dst = frame_yuyv.ptr<uint8_t>(r);
            for (int c = 0; c < IMAGE_WIDTH; c += 2) {
                dst[c*2 + 0] = src[c*3 + 0]; // Y0
                dst[c*2 + 1] = (src[c*3 + 1] + src[(c+1)*3 + 1]) / 2; // U
                dst[c*2 + 2] = src[(c+1)*3 + 0]; // Y1
                dst[c*2 + 3] = (src[c*3 + 2] + src[(c+1)*3 + 2]) / 2; // V
            }
        }

        size_t yuyv_size = (size_t)IMAGE_WIDTH * IMAGE_HEIGHT * 2;
        memcpy(drpai_buf->mem, frame_yuyv.data, yuyv_size);
        int ret = buffer_flush_dmabuf(drpai_buf->idx, yuyv_size);
        if (ret < 0) {
            LOGR_ERROR("MidasEstimatorV2H: buffer flush failed");
            return cv::Mat::zeros(MIDAS_MODEL_OUT_H, MIDAS_MODEL_OUT_W, CV_32F);
        }

        /* 2. Pre-processing on DRP-AI ---------------------------------------- */
        void*    output_ptr = nullptr;
        uint32_t out_size   = 0;

        in_param.pre_in_shape_w = IMAGE_WIDTH;
        in_param.pre_in_shape_h = IMAGE_HEIGHT;
        in_param.pre_in_format  = FORMAT_YUYV_422;
        in_param.pre_out_format = FORMAT_RGB;

        /* Reload MiDaS Pre-processing efficiently via cache */
        AI::get_instance()->ReloadPre(c::midas::pre_dir);

        float mean[3] = {0.485f, 0.456f, 0.406f};
        float std_v[3] = {0.229f, 0.224f, 0.225f};
        in_param.cof_add[0] = -mean[2] * 255.0f;
        in_param.cof_add[1] = -mean[1] * 255.0f;
        in_param.cof_add[2] = -mean[0] * 255.0f;
        in_param.cof_mul[0] = 1.0f / (std_v[2] * 255.0f);
        in_param.cof_mul[1] = 1.0f / (std_v[1] * 255.0f);
        in_param.cof_mul[2] = 1.0f / (std_v[0] * 255.0f);

        in_param.pre_in_addr = (uintptr_t)drpai_buf->phy_addr;

        ret = preruntime->Pre(&in_param, &output_ptr, &out_size);
        if (ret < 0) {
            LOGR_ERROR("MidasEstimatorV2H: PreRuntime::Pre() failed");
            return cv::Mat::zeros(MIDAS_MODEL_OUT_H, MIDAS_MODEL_OUT_W, CV_32F);
        }

        /* 3. AI inference ----------------------------------------------------- */
        runtime->SetInput(0, reinterpret_cast<float*>(output_ptr));
        runtime->Run(DRPAI_FREQ);

        /* 4. Read output ------------------------------------------------------ */
        ret = get_result();
        if (ret != 0) {
            LOGR_ERROR("MidasEstimatorV2H: get_result() failed");
            return cv::Mat::zeros(MIDAS_MODEL_OUT_H, MIDAS_MODEL_OUT_W, CV_32F);
        }

        /* 5. Post-process: normalize to [0, 255] float map ------------------- */
        cv::Mat depth_org(MIDAS_MODEL_OUT_H, MIDAS_MODEL_OUT_W, CV_32FC1,
                          drpai_output_buf);
        cv::Mat depth_norm;
        cv::normalize(depth_org, depth_norm, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);

        /* 6. EMA temporal smoothing ------------------------------------------ */
        cv::Mat result;
        if (prev_prediction.empty() ||
            prev_prediction.size() != depth_norm.size()) {
            result = depth_norm.clone();
        } else {
            result = (ema_alpha * depth_norm) +
                     ((1.0f - ema_alpha) * prev_prediction);
        }
        prev_prediction = result.clone();

        return result;
    });
    return future.get();
}

/*---------------------------------------------------------------------------*/

cv::Mat MidasEstimatorV2H::get_standardized_depth(const cv::Mat& depth_map)
{
    /* MiDaS output is already [0,255] from inference().
     * Re-scale to a nominal [0, 1000] range for compatibility with the
     * volume_math / calibration_routines which expect that range. */
    constexpr float SATURATION_POINT = 255.0f;
    constexpr float scale            = 1000.0f / SATURATION_POINT;

    cv::Mat standardized;
    depth_map.convertTo(standardized, CV_32F, scale);

    /* Clamp to [0, 1000] */
    cv::threshold(standardized, standardized, 1000.0, 1000.0,
                  cv::THRESH_TRUNC);
    cv::max(standardized, cv::Mat::zeros(standardized.size(), CV_32F),
            standardized);

    return standardized;
}

float MidasEstimatorV2H::get_tray_depth(const cv::Mat& depth_map,
                                        const cv::Rect& roi_coords)
{
    cv::Mat standardized = depth_map;
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    if (maxVal > 1000.0) {
        standardized = get_standardized_depth(depth_map);
    }

    /* Clamp ROI to image bounds */
    cv::Rect safe_roi = roi_coords & cv::Rect(0, 0, standardized.cols,
                                               standardized.rows);
    cv::Mat roi = standardized(safe_roi);
    if (roi.empty()) return 0.0f;

    /* Median computation (robust for non-continuous ROI) */
    std::vector<float> vals;
    vals.reserve(roi.rows * roi.cols);
    for (int r = 0; r < roi.rows; ++r) {
        const float* ptr = roi.ptr<float>(r);
        vals.insert(vals.end(), ptr, ptr + roi.cols);
    }

    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

float MidasEstimatorV2H::get_rim_depth(const cv::Mat& depth_map,
                                       const cv::Rect& bbox)
{
    cv::Mat standardized = depth_map;
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    if (maxVal > 1000.0) {
        standardized = get_standardized_depth(depth_map);
    }

    int thickness_inward = std::max(4, bbox.height / 10);
    int px1 = std::max(bbox.x, bbox.x + bbox.width / 4);
    int px2 = std::min(bbox.x + bbox.width,
                       bbox.x + bbox.width - bbox.width / 4);
    int py1 = bbox.y;
    int py2 = std::min(bbox.y + bbox.height, bbox.y + thickness_inward);

    cv::Rect patch_rect(px1, py1, px2 - px1, py2 - py1);
    /* Clamp to image */
    patch_rect &= cv::Rect(0, 0, standardized.cols, standardized.rows);
    cv::Mat patch = standardized(patch_rect);
    if (patch.empty()) return 0.0f;

    /* Median computation (robust for non-continuous ROI) */
    std::vector<float> vals;
    vals.reserve(patch.rows * patch.cols);
    for (int r = 0; r < patch.rows; ++r) {
        const float* ptr = patch.ptr<float>(r);
        vals.insert(vals.end(), ptr, ptr + patch.cols);
    }

    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

#endif  // V2H && DRP_AI_TVM_RUNTIME == 1
