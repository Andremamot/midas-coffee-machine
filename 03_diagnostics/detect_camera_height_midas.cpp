#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <yaml-cpp/yaml.h>

#include <detections/ai.h>
#include "../midas_volumecup/volume_math.hpp"

int main() {
    std::string config_file = "../midas_calibration.yaml";
    float manual_h = 29.0f, focal = 846.0f, alpha = 1.0f;
    float c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0;
    int cam_id = 0;
    cv::Rect roi(10, 400, 90, 70); 

    try {
        YAML::Node config = YAML::LoadFile(config_file);
        if (config["focal_length"]) focal = config["focal_length"].as<float>();
        if (config["alpha"]) alpha = config["alpha"].as<float>();
        if (config["c1"]) c1 = config["c1"].as<float>();
        if (config["c2"]) c2 = config["c2"].as<float>();
        if (config["c3"]) c3 = config["c3"].as<float>();
        if (config["c4"]) c4 = config["c4"].as<float>();
        if (config["camera_index"]) cam_id = config["camera_index"].as<int>();
        if (config["tray_roi"]) {
            auto r = config["tray_roi"].as<std::vector<int>>();
            if (r.size() == 4) roi = cv::Rect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
        }
    } catch (...) {}

    std::cout << "Loading AI Models... Please wait.\n";
    std::cout << "Loading AI Models... Please wait.\n";
    AI* ai = AI::get_instance();
    std::cout << "Models Ready.\n";

    cv::VideoCapture cap(cam_id);
    if (!cap.isOpened()) {
        std::cerr << "Camera failed to open.\n";
        return 1;
    }
    
    std::cout << "Press ESC to exit.\n";

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) continue;
        
        cv::Mat depth_map = ai->midas_estimator->inference(frame);
        auto [boxes] = ai->cup_detector->detect(frame);
        
        cv::Mat depth_norm;
        cv::normalize(depth_map, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        float m_tray = ai->midas_estimator->get_tray_depth(depth_norm, roi);
        cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);
        
        float floor_z_multi = (c1 * m_tray) + (c2 * m_tray) + (c3 * manual_h) + c4;
        float floor_z_alpha = VolumeMath::calculate_z_rim_alpha(m_tray, m_tray, manual_h, alpha);
        
        std::string debug_text = "M_tray: " + std::to_string(m_tray) + "\n";
        debug_text += "Floor Multi: " + std::to_string(floor_z_multi) + "cm\n";
        debug_text += "Floor Alpha: " + std::to_string(floor_z_alpha) + "cm\n";
        
        cv::Mat debug_frame = frame.clone();
        
        if (!boxes.empty() && m_tray > 0) {
            cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
            float m_rim = ai->midas_estimator->get_rim_depth(depth_norm, bbox);
            float final_z_multi = (c1 * m_rim) + (c2 * m_tray) + (c3 * manual_h) + c4;
            float final_h_multi = manual_h - final_z_multi;
            
            debug_text += "\nM_rim: " + std::to_string(m_rim) + "\n";
            debug_text += "Cup Z Multi: " + std::to_string(final_z_multi) + "cm\n";
            debug_text += "Cup H Multi: " + std::to_string(final_h_multi) + "cm";
            
            cv::rectangle(debug_frame, bbox, cv::Scalar(0, 255, 0), 2);
        }
        
        cv::Mat heatmap;
        cv::applyColorMap(depth_norm, heatmap, cv::COLORMAP_INFERNO);
        cv::Mat combined;
        cv::hconcat(debug_frame, heatmap, combined);
        
        int y = 30;
        size_t pos = 0;
        std::string token;
        std::string text_copy = debug_text;
        while ((pos = text_copy.find("\n")) != std::string::npos) {
            token = text_copy.substr(0, pos);
            cv::putText(combined, token, cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
            y += 30;
            text_copy.erase(0, pos + 1);
        }
        cv::putText(combined, text_copy, cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
        
        cv::imshow("Debug Visualizer", combined);
        if (cv::waitKey(1) == 27) break;
    }
    
    cap.release();
    return 0;
}
