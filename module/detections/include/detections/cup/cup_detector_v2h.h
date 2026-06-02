#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
#pragma once

#include <MeraDrpRuntimeWrapper.h>
#include <PreRuntime.h>
#include <builtin_fp16.h>
#include <constants/define_cup.h>
#include <constants/define_drpai.h>
#include <detections/box.h>
#include <detections/cup/dfl_proc.h>
#include <dmabuf.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>
#include <logger/logger.h>

class CupDetectorV2H {
public:
    CupDetectorV2H();
    ~CupDetectorV2H();

    // Runtime injected from outside
    MeraDrpRuntimeWrapper* runtime = nullptr;
    PreRuntime* preruntime = nullptr;

    dma_buffer* drpai_buf = nullptr;

    // Main Detection Function
    std::tuple<std::vector<Detection>> detect(cv::Mat& frame);

    bool check_cup_in_center(const std::vector<Detection>& detections,
                             const cv::Point& center_point);

private:
    /* ===== Internal Buffers ===== */
    float drpai_output_buf[c::yolov8::num_inf_out];

    std::vector<Detection> det;

    DFL dfl;
    s_preproc_param_t in_param;

    const float TOLERANCE_FACTOR = 0.1f;

    /* ===== Internal Processing ===== */
    void R_Post_Proc(float* floatarr);
    int8_t get_result();
    float float16_to_float32(uint16_t a);

    //     /* new post process */
    //     int model_w;
    //     int model_h;
    //     int num_reg;
    //     int num_class;
    //     float thresh_pre_nms;
    //     float thresh_nms;

    //     void decode(float *cls, float *bb, int stride, std::vector<Detection>
    //     &dets); std::vector<std::vector<float>>
    //     dist2bbox(std::vector<std::vector<float>> distance,
    //     std::vector<std::vector<float>> anchor_points, bool xywh = true, int
    //     dim = -1); std::vector<std::vector<float>> make_anchor(int h, int w,
    //     float grid_cell_offset = 0.5); std::vector<std::vector<float>>
    //     dfl(float *bb, int h, int w); std::vector<float> dfl_target(float
    //     *bb, int target_idx); std::vector<float>
    //     dist2bbox_target(std::vector<float> bbox, std::vector<float>
    //     anchor_points, bool xywh = true, int dim = -1); std::vector<float>
    //     conv;

    //     std::vector<Detection> postprocess(float *cls_8, float *cls_16, float
    //     *cls_32, float *bb_8, float *bb_16, float *bb_32);
    //
};

#endif  // V2H