#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <yaml-cpp/yaml.h>

#include <detections/ai.h>
#include "../midas_volumecup/volume_math.hpp"
#include "../midas_volumecup/camera_config.hpp"

int main() {
    std::string config_file = "../midas_calibration.yaml";
    float focal_length = 800.0f;
    float m = 1.0f, c = 0.0f, mb = 1.0f, cb = 0.0f;
    float a = 0.0f, b = 10.0f, coeff_c = 0.0f;
    int cam_idx = 0;
    cv::Rect roi(10, 400, 90, 70); 

    try {
        YAML::Node config = YAML::LoadFile(config_file);
        if (config["focal_length"]) focal_length = config["focal_length"].as<float>();
        if (config["a"]) a = config["a"].as<float>();
        if (config["b"]) b = config["b"].as<float>();
        if (config["c"]) coeff_c = config["c"].as<float>();
        
        if (config["signal_m"]) m = config["signal_m"].as<float>();
        if (config["signal_c"]) c = config["signal_c"].as<float>();
        if (config["signal_mb"]) mb = config["signal_mb"].as<float>();
        
        if (config["camera_index"]) cam_idx = config["camera_index"].as<int>();
        if (config["tray_roi"]) {
            auto r = config["tray_roi"].as<std::vector<int>>();
            if (r.size() == 4) roi = cv::Rect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
        }
    } catch (...) {}

    std::cout << "Loading models...\n";
    AI* ai = AI::get_instance();
    
    cv::VideoCapture cap(cam_idx);
    if (!cap.isOpened()) {
        std::cerr << "Camera failed to open.\n";
        return 1;
    }
    
    float z_rim_smooth = -1.0f;
    CameraConfig* cam_config = nullptr;
    
    std::cout << "Running logic loop...\nPress ESC to exit.\n";

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) continue;
        
        if (!cam_config) {
            cam_config = new CameraConfig(frame.cols, frame.rows);
        }
        
        cv::Mat depth_map = ai->midas_estimator->inference(frame);
        auto [boxes] = ai->cup_detector->detect(frame);
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        bool valid_h = false;
        double H_nozzle = VolumeMath::measure_nozzle_height(gray, *cam_config, m, c, mb, cb, valid_h);
        
        std::string volume_text = "Volume: N/A";
        cv::Mat depth_norm;
        cv::normalize(depth_map, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        if (valid_h && !boxes.empty()) {
            cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            
            float w_pixels = std::max(bbox.width, bbox.height);
            
            float m_rim = ai->midas_estimator->get_rim_depth(depth_norm, bbox);
            float m_tray = ai->midas_estimator->get_tray_depth(depth_norm, roi);
            
            if (m_tray > 0) {
                float z_rim_raw = VolumeMath::calculate_z_rim(m_rim, m_tray, a, b, coeff_c, true);
                
                if (z_rim_smooth < 0) z_rim_smooth = z_rim_raw;
                else z_rim_smooth = 0.8f * z_rim_smooth + 0.2f * z_rim_raw;
                
                auto [h_cup, w_real, volume] = VolumeMath::calculate_volume(z_rim_smooth, H_nozzle, w_pixels, focal_length);
                
                char buffer[128];
                snprintf(buffer, sizeof(buffer), "H_cam: %.1fcm | Z_rim: %.1fcm | Vol: %.0fmL", H_nozzle, z_rim_smooth, volume);
                volume_text = std::string(buffer);
            }
        }
        
        cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, "Tray ROI", cv::Point(roi.x, roi.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
        cv::putText(frame, volume_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::Mat depth_vis;
        cv::applyColorMap(depth_norm, depth_vis, cv::COLORMAP_INFERNO);
        cv::Mat combined;
        cv::hconcat(frame, depth_vis, combined);
        
        cv::imshow("MiDaS Volume Runner", combined);
        if (cv::waitKey(1) == 27) break;
    }
    
    delete cam_config;
    cap.release();
    return 0;
}
