/*******************************************************************************
 * midas_estimator_v2h.h
 * MiDaS depth estimator using DRP-AI TVM runtime (Renesas RZ/V2H)
 * Follows app_midas_cam/src/main_midas.cpp pattern, packaged as a class.
 ******************************************************************************/
#pragma once

#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)

#include <MeraDrpRuntimeWrapper.h>
#include <PreRuntime.h>
#include <builtin_fp16.h>
#include <constants/define_midas.h>
#include <constants/define_drpai.h>
#include <dmabuf.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/*******************************************************************************
 * MidasEstimatorV2H
 *
 * Usage:
 *   // Runtime and preruntime are injected from AI singleton (owner = AI)
 *   auto* midas = ai->midas_estimator.get();
 *   cv::Mat depth = midas->inference(frame);       // full-frame depth
 *   float  rim    = midas->get_rim_depth(depth, bbox_rect);
 *   float  tray   = midas->get_tray_depth(depth, tray_rect);
 ******************************************************************************/
class MidasEstimatorV2H {
public:
    MidasEstimatorV2H();
    ~MidasEstimatorV2H();

    /* ----------------------------------------------------------------
     * Injected external runtime pointers (owned by AI singleton)
     * ----------------------------------------------------------------*/
    MeraDrpRuntimeWrapper* runtime    = nullptr;
    PreRuntime*            preruntime = nullptr;

    /* ----------------------------------------------------------------
     * Public API — mirrors midas_volumecup/depth.hpp interface
     * ----------------------------------------------------------------*/

    /**
     * Run DRP-AI inference on a BGR frame.
     * @param frame  Input BGR frame (any resolution; will be pre-processed by PreRuntime)
     * @return       Float depth map (same spatial size as MIDAS_MODEL_OUT_WxH),
     *               normalized to [0, 255] and temporally smoothed.
     */
    cv::Mat inference(const cv::Mat& frame);

    /**
     * Standardize a raw depth map to [0, 1000] range.
     */
    cv::Mat get_standardized_depth(const cv::Mat& depth_map);

    /**
     * Compute median depth value within a tray ROI rectangle.
     * Returns depth in [0, 1000] (standardized) units.
     */
    float get_tray_depth(const cv::Mat& depth_map, const cv::Rect& roi_coords);

    /**
     * Compute rim depth from top-strip of a bounding box.
     * Returns depth in [0, 1000] (standardized) units.
     */
    float get_rim_depth(const cv::Mat& depth_map, const cv::Rect& bbox);

private:
    /* Raw DRP-AI output buffer (W*H floats) */
    float drpai_output_buf[MIDAS_MODEL_OUT_W * MIDAS_MODEL_OUT_H * MIDAS_NUM_CLASS];

    /* Pre-process parameter struct */
    s_preproc_param_t in_param;

    /* DMA buffer for camera frame copy */
    dma_buffer* drpai_buf = nullptr;

    /* EMA smoother state */
    cv::Mat prev_prediction;
    float   ema_alpha = 0.4f;

    /* FP16 → FP32 utility */
    float float16_to_float32(uint16_t a);

    /* Read DRP-AI output into drpai_output_buf */
    int8_t get_result();

    /* Post-process: normalize float output → depth CV_32F */
    void R_Post_Proc_MiDaS(float* floatarr);
};

#endif  // V2H && DRP_AI_TVM_RUNTIME == 1
