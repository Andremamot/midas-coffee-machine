#pragma once

#include <darknet/preprocess.h>
#include <detections/box.h>
#include <logger/logger.h>
#include <onnxruntime_cxx_api.h>

#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>

class FaceDetector {
public:
    FaceDetector();
    ~FaceDetector();

    std::string model_path = "model/face/face-detector.onnx";
    std::vector<std::string> class_names = {"face"};

    // onnx
    std::unique_ptr<Ort::Session> session = nullptr;
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    // constants
    const char* INPUT_NAME = "input";
    const char* OUTPUT_NAME = "output";
    const int INPUT_WIDTH = 416;
    const int INPUT_HEIGHT = 416;
    const float THRESHOLD_SCORE = 0.3;
    const float THRESHOLD_NMS = 0.3;
    const int NUM_CLASSES = 1;

    std::vector<int64_t> input_dims = {1, 3, 416, 416};
    std::vector<int64_t> output_dims;

    std::atomic<bool> running{false};

    std::tuple<std::vector<Detection>> detect(cv::Mat& frame);
};
