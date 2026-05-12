#ifdef V2H

#pragma once

#include <MeraDrpRuntimeWrapper.h>
#include <builtin_fp16.h>
#include <constants/define_drpai.h>
#include <constants/define_face.h>
#include <detections/box.h>
#include <logger/logger.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <tuple>

class FaceDetectorV2H {
public:
    FaceDetectorV2H();
    ~FaceDetectorV2H();

    MeraDrpRuntimeWrapper* runtime;
    float drpai_output_buf[c::yolov3::num_inf_out];

    std::vector<std::string> class_names = {"face"};
    std::vector<Detection> det;

    double sigmoid(double x);
    int32_t yolo_index(uint8_t n, int32_t offs, int32_t channel);
    int32_t yolo_offset(uint8_t n, int32_t b, int32_t y, int32_t x);
    void R_Post_Proc(float* floatarr);

    float float16_to_float32(uint16_t a);

    std::tuple<std::vector<Detection>> detect(cv::Mat& frame);
};

#endif  // V2H