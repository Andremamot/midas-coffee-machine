#include "depth.hpp"
#include <iostream>
#include <algorithm>

MidasDepthEstimator::MidasDepthEstimator(const std::string& weights_path, const std::string& model_type) : ema_alpha(0.4f) {
    std::cout << "Loading MiDaS ONNX model from " << weights_path << "..." << std::endl;
    try {
        net = cv::dnn::readNetFromONNX(weights_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "Successfully loaded MiDaS with OpenCV DNN." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Exception loading ONNX MiDaS: " << e.what() << "\nMake sure you have exported your MiDaS .pt to .onnx!" << std::endl;
    }
}

cv::Mat MidasDepthEstimator::process(const cv::Mat& image) {
    // 1. Apply CLAHE
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab, lab_planes);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab);
    cv::Mat clahe_image;
    cv::cvtColor(lab, clahe_image, cv::COLOR_Lab2BGR);
    
    // 2. RGB for MiDaS
    cv::Mat rgb;
    cv::cvtColor(clahe_image, rgb, cv::COLOR_BGR2RGB);
    
    // scale to 1/255.0, Mean and std for midas small
    cv::Mat blob;
    // MiDaS small uses resolution 256x256
    cv::dnn::blobFromImage(rgb, blob, 1.0/255.0, cv::Size(256, 256), cv::Scalar(123.675, 116.28, 103.53), true, false);
    // actually, it's (img/255 - mean)/std. Let's use simple scaling without mean subtraction as closest to python code:
    // Python code: img_input = self.transform({"image": original_image_rgb / 255.0})["image"]
    // We'll leave the exact normalize to simple 1/255 for now if that works.
    
    cv::dnn::blobFromImage(rgb, blob, 1.0/255.0, cv::Size(256, 256), cv::Scalar(), true, false);
    
    if (net.empty()) {
        std::cerr << "Warning: MiDaS network is empty. Returning blank depth map.\n";
        return cv::Mat::zeros(image.size(), CV_32F);
    }

    net.setInput(blob);
    cv::Mat prediction = net.forward();
    
    // shape is typically [1, 1, 256, 256] or similar
    // Reshape and resize
    int H = prediction.size[2];
    int W = prediction.size[3];
    cv::Mat depth(H, W, CV_32F, prediction.ptr<float>());
    
    cv::Mat resized;
    cv::resize(depth, resized, image.size(), 0, 0, cv::INTER_CUBIC);
    
    // Temporal smoothing
    if (prev_prediction.empty() || prev_prediction.size() != resized.size()) {
        prev_prediction = resized.clone();
    } else {
        resized = (ema_alpha * resized) + ((1.0f - ema_alpha) * prev_prediction);
        prev_prediction = resized.clone();
    }
    
    return resized;
}

cv::Mat MidasDepthEstimator::get_standardized_depth(const cv::Mat& depth_map) {
    float SATURATION_POINT = 2500.0f;
    float scale = 1000.0f / SATURATION_POINT;
    
    cv::Mat standardized;
    depth_map.convertTo(standardized, CV_32F, scale);
    
    // clip to [0, 1000]
    cv::threshold(standardized, standardized, 1000.0, 1000.0, cv::THRESH_TRUNC);
    cv::Mat zeros = cv::Mat::zeros(standardized.size(), CV_32F);
    cv::max(standardized, zeros, standardized);
    
    return standardized;
}

float MidasDepthEstimator::get_tray_depth(const cv::Mat& depth_map, const cv::Rect& roi_coords) {
    cv::Mat standardized = depth_map;
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    if (maxVal > 1000) {
        standardized = get_standardized_depth(depth_map);
    }
    
    cv::Mat roi = standardized(roi_coords);
    if (roi.empty()) return 0.0f;
    
    // median
    std::vector<float> vals;
    vals.assign((float*)roi.datastart, (float*)roi.dataend);
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

float MidasDepthEstimator::get_rim_depth(const cv::Mat& depth_map, const cv::Rect& bbox) {
    cv::Mat standardized = depth_map;
    double minVal, maxVal;
    cv::minMaxLoc(depth_map, &minVal, &maxVal);
    if (maxVal > 1000) {
        standardized = get_standardized_depth(depth_map);
    }
    
    int thickness_inward = std::max(4, bbox.height / 10);
    int px1 = std::max(bbox.x, bbox.x + bbox.width / 4);
    int px2 = std::min(bbox.x + bbox.width, bbox.x + bbox.width - bbox.width / 4);
    
    int py1 = bbox.y;
    int py2 = std::min(bbox.y + bbox.height, bbox.y + thickness_inward);
    
    cv::Rect patch_rect(px1, py1, px2 - px1, py2 - py1);
    cv::Mat patch = standardized(patch_rect);
    if (patch.empty()) return 0.0f;
    
    std::vector<float> vals;
    vals.assign((float*)patch.datastart, (float*)patch.dataend);
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}
