#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <detections/ai.h>

using json = nlohmann::json;

struct MultiDataPoint {
    float m_rim;
    float m_tray;
    float z_tray;
    float z_rim;
    float diam_rim;
    float diam_outer;
    long long timestamp;
};

void run_cli(std::mutex& mtx, bool& running, bool& capture_flag, bool& fit_flag, bool& save_flag, float& tz_tray, float& tz_rim, float& diam_rim, float& diam_out, int& cam_id) {
    while (running) {
        std::cout << "\nMenu:\n[1] Capture Point\n[2] Fit Multivariate\n[3] Save to YAML\n[4] Change Camera\n[5] Exit\nChoice: " << std::flush;
        int choice; 
        if (!(std::cin >> choice)) {
            std::cin.clear(); std::cin.ignore(10000, '\n'); continue;
        }
        
        std::lock_guard<std::mutex> lock(mtx);
        if (choice == 1) {
            std::cout << "True Z Tray (cm): "; std::cin >> tz_tray;
            std::cout << "True Z Rim (cm): "; std::cin >> tz_rim;
            std::cout << "True Inner Diam (cm): "; std::cin >> diam_rim;
            std::cout << "True Outer Diam (cm) [0 for none]: "; std::cin >> diam_out;
            capture_flag = true;
        } else if (choice == 2) {
            fit_flag = true;
        } else if (choice == 3) {
            save_flag = true;
        } else if (choice == 4) {
            std::cout << "Camera ID: "; std::cin >> cam_id;
        } else if (choice == 5) {
            running = false;
        }
    }
}

int main() {
    std::string config_file = "../midas_calibration.yaml";
    std::string points_file = "calibration_points_multivariate.json";
    std::string snapshots_dir = "calibration_snapshots_multivariate";
    
    mkdir(snapshots_dir.c_str(), 0777);

    float focal_length = 846.0f;
    int camera_index = 0;
    cv::Rect roi(10, 400, 90, 70); 

    try {
        YAML::Node config = YAML::LoadFile(config_file);
        if (config["focal_length"]) focal_length = config["focal_length"].as<float>();
        if (config["camera_index"]) camera_index = config["camera_index"].as<int>();
        if (config["tray_roi"]) {
            auto r = config["tray_roi"].as<std::vector<int>>();
            if (r.size() == 4) roi = cv::Rect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
        }
    } catch (...) {}

    std::vector<MultiDataPoint> points;
    try {
        std::ifstream f(points_file);
        if (f.good()) {
            json j; f >> j;
            for (auto& item : j) {
                float dr = item.contains("Diam_rim") ? (float)item["Diam_rim"] : 0.0f;
                float dou = item.contains("Diam_outer") ? (float)item["Diam_outer"] : 0.0f;
                points.push_back({item["M_rim"], item["M_tray"], item["Z_tray"], item["Z_rim"], dr, dou, item["timestamp"]});
            }
            std::cout << "Loaded " << points.size() << " multivariate points.\n";
        }
    } catch (...) {}

    AI* ai = AI::get_instance();
    
    std::mutex mtx;
    bool running = true, capture_flag = false, fit_flag = false, save_flag = false;
    float tz_tray = 35.0f, tz_rim = 15.0f, diam_rim = 7.2f, diam_out = 0.0f;
    cv::Mat current_coeffs; // C1..C4
    int current_cam_id = camera_index;

    std::thread cli_thread(run_cli, std::ref(mtx), std::ref(running), std::ref(capture_flag), std::ref(fit_flag), std::ref(save_flag), std::ref(tz_tray), std::ref(tz_rim), std::ref(diam_rim), std::ref(diam_out), std::ref(current_cam_id));

    cv::VideoCapture cap(current_cam_id);
    
    while (running) {
        mtx.lock();
        int requested_cam = current_cam_id;
        bool do_cap = capture_flag; capture_flag = false;
        bool do_fit = fit_flag; fit_flag = false;
        bool do_save = save_flag; save_flag = false;
        mtx.unlock();

        if (!cap.isOpened() || static_cast<int>(cap.get(cv::CAP_PROP_POS_MSEC)) == -1) { 
            cap.release();
            cap.open(requested_cam);
        }

        cv::Mat frame;
        if (cap.isOpened() && cap.read(frame)) {
            cv::Mat depth = ai->midas_estimator->inference(frame);
            auto [boxes] = ai->cup_detector->detect(frame);

            if (do_cap) {
                if (!boxes.empty()) {
                    cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
                    float mr = ai->midas_estimator->get_rim_depth(depth, bbox); // depth_estimator now handles standardization internally
                    float mt = ai->midas_estimator->get_tray_depth(depth, roi);
                    if (mt > 0) {
                        long long ts = std::chrono::system_clock::now().time_since_epoch().count();
                        points.push_back({mr, mt, tz_tray, tz_rim, diam_rim, diam_out, ts});
                        
                        json j = json::array();
                        for (auto& p : points) {
                            j.push_back({{"M_rim", p.m_rim}, {"M_tray", p.m_tray}, {"Z_tray", p.z_tray}, {"Z_rim", p.z_rim}, {"Diam_rim", p.diam_rim}, {"Diam_outer", p.diam_outer}, {"timestamp", p.timestamp}});
                        }
                        std::ofstream f(points_file);
                        f << j.dump(4);
                        
                        std::string img_path = snapshots_dir + "/calib_tray" + std::to_string(tz_tray) + "cm_rim" + std::to_string(tz_rim) + "cm_diam" + std::to_string(diam_rim) + "cm_" + std::to_string(ts) + ".jpg";
                        cv::imwrite(img_path, frame);
                        std::cout << "\nCaptured & Saved multivariate point (" << points.size() << ")!\n";
                    }
                }
            }

            if (do_fit && points.size() >= 4) {
                // X: [M_rim, M_tray, Z_tray, 1]
                cv::Mat X(points.size(), 4, CV_32F);
                cv::Mat Y(points.size(), 1, CV_32F);
                for (size_t i = 0; i < points.size(); ++i) {
                    X.at<float>(i, 0) = points[i].m_rim;
                    X.at<float>(i, 1) = points[i].m_tray;
                    X.at<float>(i, 2) = points[i].z_tray;
                    X.at<float>(i, 3) = 1.0f;
                    Y.at<float>(i, 0) = points[i].z_rim;
                }
                
                // Solve Y = X * C  => C = (X^T * X)^-1 * X^T * Y
                cv::solve(X, Y, current_coeffs, cv::DECOMP_SVD);
                
                std::cout << "\nFitted Multivariate Coefficients:\n";
                if (!current_coeffs.empty()) {
                    std::cout << "C1: " << current_coeffs.at<float>(0, 0) << "\n";
                    std::cout << "C2: " << current_coeffs.at<float>(1, 0) << "\n";
                    std::cout << "C3: " << current_coeffs.at<float>(2, 0) << "\n";
                    std::cout << "C4: " << current_coeffs.at<float>(3, 0) << "\n";
                }
            } else if (do_fit) {
                std::cout << "\nNeed at least 4 points.\n";
            }

            if (do_save && !current_coeffs.empty()) {
                YAML::Node out_config;
                try { out_config = YAML::LoadFile(config_file); } catch(...) {}
                out_config["focal_length"] = focal_length;
                out_config["c1"] = current_coeffs.at<float>(0, 0);
                out_config["c2"] = current_coeffs.at<float>(1, 0);
                out_config["c3"] = current_coeffs.at<float>(2, 0);
                out_config["c4"] = current_coeffs.at<float>(3, 0);
                out_config["tray_roi"] = std::vector<int>{roi.x, roi.y, roi.x + roi.width, roi.y + roi.height};
                out_config["camera_index"] = requested_cam;
                
                std::ofstream fout(config_file);
                fout << out_config;
                std::cout << "\nSaved to YAML!\n";
            }

            if (!boxes.empty()) {
                cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            }
            cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);
            
            cv::Mat depth_std = ai->midas_estimator->get_standardized_depth(depth);
            cv::Mat depth_norm;
            depth_std.convertTo(depth_norm, CV_8U, 255.0/1000.0);
            
            cv::Mat dv;
            cv::applyColorMap(depth_norm, dv, cv::COLORMAP_INFERNO);
            
            cv::Mat comb;
            cv::hconcat(frame, dv, comb);
            cv::imshow("Multivariate Calibration", comb);
            if (cv::waitKey(1) == 27) running = false; // ESC
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    cli_thread.join();
    return 0;
}
