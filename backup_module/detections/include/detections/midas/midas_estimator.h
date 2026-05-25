#pragma once

#if defined(LINUX64) || DRP_AI_TVM_RUNTIME == 0
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <memory>
#include <string>

class MidasEstimator {
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver op_resolver;

    int input_height;
    int input_width;
    int input_channels;
    int output_height;
    int output_width;

    cv::Mat prev_prediction;
    float ema_alpha = 0.4f;

    void getModelInputDetails();
    void getModelOutputDetails();
    void prepareInputForInference(cv::Mat &image);

public:
    MidasEstimator();
    ~MidasEstimator();

    std::string model_path = "weights/midas_small_256.tflite";
    void initializeModel(size_t thread_num = 1);
    cv::Mat inference(const cv::Mat& frame);

    cv::Mat get_standardized_depth(const cv::Mat& depth_map);
    float get_tray_depth(const cv::Mat& depth_map, const cv::Rect& roi_coords);
    float get_rim_depth(const cv::Mat& depth_map, const cv::Rect& bbox);
};
#endif
