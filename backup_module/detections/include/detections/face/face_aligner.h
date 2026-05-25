#pragma once

#include <logger/logger.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>
#include <vector>

class FaceAligner {
public:
    FaceAligner();
    ~FaceAligner() = default;

    std::tuple<bool> align(const cv::Mat& image, const cv::Rect& face_rect, cv::Mat& aligned_face);

private:
    // onnx
    std::unique_ptr<Ort::Session> session = nullptr;
    Ort::Env env{nullptr};
    Ort::SessionOptions session_options{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    // constants
    const char* INPUT_NAME = "input";
    const char* OUTPUT_NAME = "output";
    const int INPUT_WIDTH = 56;
    const int INPUT_HEIGHT = 56;

    std::vector<int64_t> input_dims = {1, 3, 56, 56};
    std::vector<int64_t> output_dims;

    const std::string BASE_PATH = "model/face/";
    const std::string MODEL_FILE = "landmark_detection_56_se_external.onnx";

    std::vector<cv::Point2f> extract_landmarks(const std::vector<float>& output,
                                               int face_w,
                                               int face_h,
                                               const cv::Rect& face_rect);
    bool is_face_landmark_ratio_valid(const std::vector<cv::Point2f>& landmarks);
};
