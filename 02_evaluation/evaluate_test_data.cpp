#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <detections/ai.h>
#include "../midas_volumecup/volume_math.hpp"

struct ReportData {
    std::string filename;
    std::string debug_image;
    float m_rim;
    float m_tray;
    float ratio;
    float true_z_tray;
    float true_z;
    float pred_z;
    float error;
    float error_pct;
    float pred_floor_z;
};

std::vector<std::string> get_jpg_files(const std::string& dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (dir) {
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            if (name.length() >= 4 && name.substr(name.length() - 4) == ".jpg") {
                files.push_back(name);
            }
        }
        closedir(dir);
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main() {
    std::string root_dir = "..";
    std::string snapshot_dir = root_dir + "/01_calibration/calibration_snapshots";
    
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp_str = oss.str();
    
    std::string output_dir = "evaluation_results/eval_" + timestamp_str;
    std::string config_file = root_dir + "/midas_calibration.yaml";
    std::string report_path = output_dir + "/validation_report.md";
    
    mkdir("evaluation_results", 0777);
    mkdir(output_dir.c_str(), 0777);

    float alpha = 1.0;
    cv::Rect tray_roi(10, 400, 90, 70);
    
    try {
        YAML::Node config = YAML::LoadFile(config_file);
        if (config["alpha"]) alpha = config["alpha"].as<float>();
        if (config["tray_roi"]) {
            auto r = config["tray_roi"].as<std::vector<int>>();
            if (r.size() == 4) tray_roi = cv::Rect(r[0], r[1], r[2]-r[0], r[3]-r[1]);
        }
        std::cout << "Loaded config: Alpha=" << alpha << "\n";
    } catch (...) {
        std::cerr << "Failed to load config.\n"; return 1;
    }

    auto images = get_jpg_files(snapshot_dir);
    if (images.empty()) { std::cerr << "No snapshots found.\n"; return 1; }

    AI* ai = AI::get_instance();
    
    std::vector<ReportData> report_data;
    float total_error = 0.0;
    int valid_count = 0;

    for (const auto& img_name : images) {
        std::string filepath = snapshot_dir + "/" + img_name;
        
        float true_z_tray = 0.0, true_z_rim = 0.0;
        try {
            size_t p1 = img_name.find("tray") + 4, p2 = img_name.find("cm_rim");
            true_z_tray = std::stof(img_name.substr(p1, p2 - p1));
            p1 = p2 + 6; p2 = img_name.find("cm", p1);
            true_z_rim = std::stof(img_name.substr(p1, p2 - p1));
        } catch (...) { continue; }

        cv::Mat frame = cv::imread(filepath);
        if (frame.empty()) continue;

        cv::Mat depth_map = ai->midas_estimator->inference(frame);
        cv::Mat depth_norm;
        cv::normalize(depth_map, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        auto [boxes] = ai->cup_detector->detect(frame);
        float m_tray = ai->midas_estimator->get_tray_depth(depth_norm, tray_roi);
        
        if (!boxes.empty() && m_tray > 0) {
            cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
            float m_rim = ai->midas_estimator->get_rim_depth(depth_norm, bbox);
            float ratio = m_rim / m_tray;
            float pred_z = VolumeMath::calculate_z_rim_alpha(m_rim, m_tray, true_z_tray, alpha);
            float pred_floor_z = VolumeMath::calculate_z_rim_alpha(m_tray, m_tray, true_z_tray, alpha);

            float error = std::abs(pred_z - true_z_rim);
            float error_percent = true_z_rim > 0 ? (error / true_z_rim) * 100.0f : 0.0f;
            
            cv::Mat debug_frame = frame.clone();
            cv::rectangle(debug_frame, bbox, cv::Scalar(0, 255, 0), 2);
            
            cv::Mat heatmap;
            cv::applyColorMap(depth_norm, heatmap, cv::COLORMAP_INFERNO);
            cv::rectangle(heatmap, bbox, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(heatmap, tray_roi, cv::Scalar(255, 0, 0), 2);
            
            cv::Mat combined_debug;
            cv::hconcat(debug_frame, heatmap, combined_debug);
            
            std::string debug_name = "debug_" + img_name;
            cv::imwrite(output_dir + "/" + debug_name, combined_debug);
            
            report_data.push_back({img_name, debug_name, m_rim, m_tray, ratio, true_z_tray, true_z_rim, pred_z, error, error_percent, pred_floor_z});
            total_error += error;
            valid_count++;
        }
    }
    
    float mae = valid_count > 0 ? total_error / valid_count : 0;
    
    std::ofstream out(report_path);
    if(out.is_open()) {
        out << "# Evaluation Report\nMAE: " << mae << " cm\n\n";
        out << "| Snapshot | Pred Z | Error |\n| :--- | :--- | :--- |\n";
        for (const auto& d : report_data) {
            out << "| " << d.filename << " | " << d.pred_z << " | " << d.error << " |\n";
            out << "\n![](" << d.debug_image << ")\n";
        }
    }
    
    std::cout << "Success! Report written to: " << report_path << std::endl;
    return 0;
}
