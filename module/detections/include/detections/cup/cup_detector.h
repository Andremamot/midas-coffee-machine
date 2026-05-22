#pragma once

#if defined(LINUX64) || DRP_AI_TVM_RUNTIME == 0

#include <detections/box.h>
#include <onnxruntime_cxx_api.h>
#include <ultralytics/yolov8/preprocess.h>

#include <atomic>
#include <opencv2/opencv.hpp>
#include <tuple>

class CupDetector {
public:
    CupDetector();
    ~CupDetector();

    std::string model_path = "weights/best.onnx";
    std::vector<std::string> class_names = {"cup_rim", "cup_body"};

    // onnx
    std::unique_ptr<Ort::Session> session = nullptr;
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    // constants
    const char* INPUT_NAME = "images";
    const char* OUTPUT_NAME = "output0";
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    const float THRESHOLD_SCORE = 0.3;
    const float THRESHOLD_NMS = 0.3;
    const int NUM_CLASSES = 2;
    const float TOLERANCE_FACTOR = 0.1f;

    std::vector<int64_t> input_dims = {1, 3, 640, 640};
    std::vector<int64_t> output_dims;

    std::atomic<bool> running{false};

    std::tuple<std::vector<Detection>> detect(cv::Mat& frame);

    bool check_cup_in_center(const std::vector<Detection>& detections,
                             const cv::Point& center_point);
};
#endif  // LINUX64