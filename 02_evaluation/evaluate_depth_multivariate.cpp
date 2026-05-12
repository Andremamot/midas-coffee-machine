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
#include <matplot/matplot.h>

#include <detections/ai.h>

struct ReportData {
    std::string filename;
    std::string debug_image;
    float m_rim;
    float m_tray;
    float true_z_tray;
    float true_z;
    float pred_z;
    float error;
    float error_pct;
    float pred_h_cup;
    float pred_inner_diam;
    float true_inner_diam;
    float inner_err_pct;
    float true_outer_diam;
};

float calculate_z_rim_multivariate(float m_rim, float m_tray, float true_z_tray, float c1, float c2, float c3, float c4) {
    return (c1 * m_rim) + (c2 * m_tray) + (c3 * true_z_tray) + c4;
}

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
    std::string snapshot_dir = root_dir + "/01_calibration/calibration_snapshots_multivariate";
    
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp_str = oss.str();
    
    std::string output_dir = "evaluation_results_multivariate/eval_" + timestamp_str;
    std::string config_file = root_dir + "/midas_calibration.yaml";
    std::string report_path = output_dir + "/validation_report.md";
    
    mkdir("evaluation_results_multivariate", 0777);
    mkdir(output_dir.c_str(), 0777);

    float c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0, f_len = 846.0;
    cv::Rect tray_roi(10, 400, 90, 70);
    
    try {
        YAML::Node config = YAML::LoadFile(config_file);
        if (config["c1"]) c1 = config["c1"].as<float>();
        if (config["c2"]) c2 = config["c2"].as<float>();
        if (config["c3"]) c3 = config["c3"].as<float>();
        if (config["c4"]) c4 = config["c4"].as<float>();
        if (config["focal_length"]) f_len = config["focal_length"].as<float>();
        if (config["tray_roi"]) {
            auto r = config["tray_roi"].as<std::vector<int>>();
            if (r.size() == 4) tray_roi = cv::Rect(r[0], r[1], r[2]-r[0], r[3]-r[1]);
        }
        std::cout << "Loaded config: C1=" << c1 << " C2=" << c2 << " C3=" << c3 << " C4=" << c4 << "\n";
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
        
        float true_z_tray = 0.0, true_z_rim = 0.0, true_inner_diam = 0.0, true_outer_diam = 0.0;
        try {
            size_t p1 = img_name.find("tray") + 4, p2 = img_name.find("cm_rim");
            true_z_tray = std::stof(img_name.substr(p1, p2 - p1));
            
            p1 = p2 + 6; p2 = img_name.find("cm", p1);
            true_z_rim = std::stof(img_name.substr(p1, p2 - p1));
            
            if (img_name.find("diam") != std::string::npos) {
                p1 = img_name.find("diam") + 4; p2 = img_name.find("cm", p1);
                true_inner_diam = std::stof(img_name.substr(p1, p2 - p1));
            }
            if (img_name.find("outer") != std::string::npos) {
                p1 = img_name.find("outer") + 5; p2 = img_name.find("cm", p1);
                true_outer_diam = std::stof(img_name.substr(p1, p2 - p1));
            }
        } catch (...) { continue; }

        cv::Mat frame = cv::imread(filepath);
        if (frame.empty()) continue;

        cv::Mat depth_map = ai->midas_estimator->inference(frame);
        auto [boxes] = ai->cup_detector->detect(frame);
        
        float m_tray = ai->midas_estimator->get_tray_depth(depth_map, tray_roi);
        if (!boxes.empty() && m_tray > 0) {
            cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
            float m_rim = ai->midas_estimator->get_rim_depth(depth_map, bbox);
            
            float pred_z = calculate_z_rim_multivariate(m_rim, m_tray, true_z_tray, c1, c2, c3, c4);
            float pred_h_cup = true_z_tray - pred_z;
            float pred_inner_diam = (bbox.width * pred_z) / f_len;

            float error = std::abs(pred_z - true_z_rim);
            float error_percent = true_z_rim > 0 ? (error / true_z_rim) * 100.0f : 0.0f;
            
            cv::Mat debug_frame = frame.clone();
            cv::rectangle(debug_frame, bbox, cv::Scalar(0, 255, 0), 2);
            cv::putText(debug_frame, "Z_rim: " + std::to_string(pred_z) + "cm", cv::Point(bbox.x, bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            
            cv::Mat depth_std = ai->midas_estimator->get_standardized_depth(depth_map);
            cv::Mat depth_norm_vis;
            depth_std.convertTo(depth_norm_vis, CV_8U, 255.0 / 1000.0);
            cv::Mat heatmap;
            cv::applyColorMap(depth_norm_vis, heatmap, cv::COLORMAP_INFERNO);
            cv::rectangle(heatmap, bbox, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(heatmap, tray_roi, cv::Scalar(255, 0, 0), 2);
            
            cv::Mat combined_debug;
            cv::hconcat(debug_frame, heatmap, combined_debug);
            
            int y_offset = combined_debug.rows - 200;
            std::vector<std::string> info = {
                "True Z_rim: " + std::to_string(true_z_rim) + " cm",
                "Pred Z_rim: " + std::to_string(pred_z) + " cm",
                "Error: " + std::to_string(error) + " cm",
                "Pred Cup H: " + std::to_string(pred_h_cup) + " cm"
            };
            for (size_t i = 0; i < info.size(); ++i) {
                cv::putText(combined_debug, info[i], cv::Point(20, y_offset + i*30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,0), 4);
                cv::putText(combined_debug, info[i], cv::Point(20, y_offset + i*30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
            }
            
            std::string debug_name = "debug_" + img_name;
            cv::imwrite(output_dir + "/" + debug_name, combined_debug);
            
            float inner_err_pct = true_inner_diam > 0 ? (std::abs(pred_inner_diam - true_inner_diam) / true_inner_diam)*100.0f : 0;
            report_data.push_back({img_name, debug_name, m_rim, m_tray, true_z_tray, true_z_rim, pred_z, error, error_percent, pred_h_cup, pred_inner_diam, true_inner_diam, inner_err_pct, true_outer_diam});
            total_error += error;
            valid_count++;
        }
    }
    
    // Write MD
    float mae = valid_count > 0 ? total_error / valid_count : 0;
    std::string chart_filename = "eval_chart.png";
    if (valid_count > 0) {
        std::vector<double> true_zs, pred_zs;
        for (const auto& d : report_data) {
            true_zs.push_back(d.true_z);
            pred_zs.push_back(d.pred_z);
        }
        auto f = matplot::figure(true);
        f->size(800, 600);
        matplot::scatter(true_zs, pred_zs);
        matplot::hold(matplot::on);
        double min_z = *std::min_element(true_zs.begin(), true_zs.end()) - 2;
        double max_z = *std::max_element(true_zs.begin(), true_zs.end()) + 2;
        matplot::plot(std::vector<double>{min_z, max_z}, std::vector<double>{min_z, max_z}, "r--");
        matplot::title("Multivariate Regression Fit");
        matplot::save(output_dir + "/" + chart_filename);
    }
    
    std::ofstream out(report_path);
    if(out.is_open()) {
        out << "# MiDaS Depth Calibration: Multivariate Validation Report\n";
        out << "MAE: " << mae << " cm\n";
        out << "C1: " << c1 << "  C2: " << c2 << "  C3: " << c3 << " C4: " << c4 << "\n\n";
        if (valid_count > 0) out << "![Chart](" << chart_filename << ")\n\n";
        out << "| Snapshot | Pred Z | True Z | Error |\n";
        out << "| :--- | :--- | :--- | :--- |\n";
        for (const auto& d : report_data) {
            out << "| " << d.filename << " | " << d.pred_z << " | " << d.true_z << " | " << d.error << " |\n";
        }
        for (const auto& d : report_data) {
            out << "\n### " << d.filename << "\n![](" << d.debug_image << ")\n";
        }
    }
    
    std::cout << "Success! Report written to: " << report_path << std::endl;
    return 0;
}
