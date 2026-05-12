#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <detections/ai.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(x) _mkdir(x)
#else
#include <sys/stat.h>
#define MKDIR(x) mkdir(x, 0777)
#endif
#include <dirent.h>

void load_calibration(const std::string& params_path, cv::Mat& K, cv::Mat& D) {
    try {
        YAML::Node data = YAML::LoadFile(params_path);
        if (data["camera_matrix_right"]) {
            auto km = data["camera_matrix_right"].as<std::vector<std::vector<float>>>();
            K = cv::Mat(3, 3, CV_32F);
            for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) K.at<float>(i,j) = km[i][j];
        }
        if (data["dist_coeff_right"]) {
            auto dm = data["dist_coeff_right"].as<std::vector<std::vector<float>>>();
            D = cv::Mat(1, 5, CV_32F);
            if (!dm.empty() && dm[0].size() >= 5) {
                for(int i=0; i<5; ++i) D.at<float>(0,i) = dm[0][i];
            }
        }
    } catch (...) {
        std::cerr << "Failed to load calibration array from config.\n";
    }
}

std::string detect_tray_pattern(const std::string& image_path, const std::string& output_dir, AI* ai, const cv::Mat& K, const cv::Mat& D) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << "\n";
        return "";
    }
    
    if (!K.empty() && !D.empty()) {
        cv::Mat temp;
        cv::undistort(img, temp, K, D);
        img = temp;
    }

    cv::Mat display_img = img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    int h = gray.rows, w = gray.cols;
    
    auto [boxes] = ai->cup_detector->detect(img);
    
    cv::Rect left_roi(10, 150, 190, 200);
    cv::Rect right_roi(440, 150, 190, 200);
    
    if (!boxes.empty()) {
        int cx1 = boxes[0].box.x, cy1 = boxes[0].box.y, cx2 = boxes[0].box.x + boxes[0].box.w, cy2 = boxes[0].box.y + boxes[0].box.h;
        cv::rectangle(display_img, cv::Point(cx1, cy1), cv::Point(cx2, cy2), cv::Scalar(0, 255, 255), 2);
        
        int cup_w = cx2 - cx1;
        int cup_h = cy2 - cy1;
        int cy_center = (cy1 + cy2) / 2;
        
        int target_h = cup_w / 2;
        int ry1 = std::max(0, cy_center - target_h / 2);
        int ry2 = std::min(h, cy_center + target_h / 2);
        
        int target_w = cup_w * 0.3;
        left_roi = cv::Rect(std::max(0, cx1 - target_w), ry1, (cx1 - 5) - std::max(0, cx1 - target_w), ry2 - ry1);
        right_roi = cv::Rect(cx2 + 5, ry1, std::min(w, cx2 + target_w) - (cx2 + 5), ry2 - ry1);
    }
    
    std::vector<cv::Rect> rois;
    if (left_roi.width > 5 && left_roi.height > 5) rois.push_back(left_roi);
    if (right_roi.width > 5 && right_roi.height > 5) rois.push_back(right_roi);
    
    for (const auto& roi : rois) {
        cv::rectangle(display_img, roi, cv::Scalar(255, 0, 0), 2);
        cv::Mat roi_gray = gray(roi);
        cv::Mat roi_blurred;
        cv::GaussianBlur(roi_gray, roi_blurred, cv::Size(5, 5), 0);
        
        cv::Mat edges;
        cv::Canny(roi_blurred, edges, 50, 150);
        
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180.0, 25, 25, 12);
        
        for (const auto& line : lines) {
            int glx1 = line[0] + roi.x, gly1 = line[1] + roi.y;
            int glx2 = line[2] + roi.x, gly2 = line[3] + roi.y;
            
            float angle = std::abs(std::atan2(gly2 - gly1, glx2 - glx1) * 180.0 / CV_PI);
            if (angle < 3 || angle > 177) {
                cv::line(display_img, cv::Point(glx1, gly1), cv::Point(glx2, gly2), cv::Scalar(0, 255, 0), 2);
            }
        }
        
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(roi_blurred, corners, 25, 0.02, 10);
        for (const auto& c : corners) {
            cv::circle(display_img, cv::Point(c.x + roi.x, c.y + roi.y), 3, cv::Scalar(0, 0, 255), -1);
        }
    }
    
    size_t last_slash = image_path.find_last_of("/\\");
    std::string base_name = (last_slash == std::string::npos) ? image_path : image_path.substr(last_slash + 1);
    std::string output_path = output_dir + "/v4_detected_" + base_name;
    
    cv::imwrite(output_path, display_img);
    std::cout << "Results saved to: " << output_path << "\n";
    return output_path;
}

std::vector<std::string> get_image_files(const std::string& dir_path) {
    std::vector<std::string> files;
    DIR* dir = opendir(dir_path.c_str());
    if (dir) {
        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            std::string name = ent->d_name;
            if (name.length() >= 4) {
                std::string ext = name.substr(name.length() - 4);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".jpg" || ext == "jpeg" || ext == ".png") {
                    files.push_back(dir_path + "/" + name);
                }
            }
        }
        closedir(dir);
    }
    return files;
}

int main(int argc, char** argv) {
    std::string root_dir = "..";
    std::string output_dir = root_dir + "/05_tray_pattern_recog/results";
    std::string params_file = root_dir + "/calibration_params.yml";
    std::string weights_path = root_dir + "/weights/cup_detection_v3_12_s_best.onnx";
    
    std::string image_arg = "";
    std::string input_dir_arg = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--image" && i + 1 < argc) image_arg = argv[++i];
        else if (arg == "--input_dir" && i + 1 < argc) input_dir_arg = argv[++i];
        else if (arg == "--output_dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--params" && i + 1 < argc) params_file = argv[++i];
    }
    
    MKDIR(output_dir.c_str());
    
    cv::Mat K, D;
    load_calibration(params_file, K, D);
    AI* ai = AI::get_instance();
    
    if (!image_arg.empty()) {
        detect_tray_pattern(image_arg, output_dir, ai, K, D);
    } else {
        std::string snapshots_dir = input_dir_arg.empty() ? root_dir + "/01_calibration/calibration_snapshots" : input_dir_arg;
        auto files = get_image_files(snapshots_dir);
        if (files.empty()) {
            std::cerr << "Error: No images found in " << snapshots_dir << "\n";
        } else {
            for (const auto& f : files) detect_tray_pattern(f, output_dir, ai, K, D);
        }
    }
    return 0;
}
