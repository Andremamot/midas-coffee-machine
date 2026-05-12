#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

class MidasDepthEstimator {
public:
    // Requires ONNX export of MiDaS
    MidasDepthEstimator(const std::string& weights_path = "weights/midas_v21_small_256.onnx", const std::string& model_type = "midas_v21_small_256");
    cv::Mat process(const cv::Mat& image);
    cv::Mat get_standardized_depth(const cv::Mat& depth_map);
    float get_tray_depth(const cv::Mat& depth_map, const cv::Rect& roi_coords);
    float get_rim_depth(const cv::Mat& depth_map, const cv::Rect& bbox);
    
private:
    cv::dnn::Net net;
    cv::Mat prev_prediction;
    float ema_alpha;
};
