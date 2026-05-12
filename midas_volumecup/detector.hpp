#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

struct BBox {
    int x1, y1, x2, y2;
    float conf;
};

class YoloDetector {
public:
    // We expect ONNX for OpenCV DNN
    YoloDetector(const std::string& weights_path = "weights/cup_detection_v3_12_s_best.onnx");
    std::vector<BBox> detect(const cv::Mat& frame);
private:
    cv::dnn::Net net;
};
