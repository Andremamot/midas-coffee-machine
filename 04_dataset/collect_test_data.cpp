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

struct TestDataPoint {
    float m_rim;
    float m_tray;
    float z_tray;
    float z_rim;
    long long timestamp;
};

void run_cli(std::mutex& mtx, bool& running, bool& capture_flag, float& tz_tray, float& tz_rim, int& cam_id) {
    while (running) {
        std::cout << "\nMenu:\n[1] Capture Dataset Point\n[2] Change Camera\n[3] Exit\nChoice: " << std::flush;
        int choice; 
        if (!(std::cin >> choice)) {
            std::cin.clear(); std::cin.ignore(10000, '\n'); continue;
        }
        
        std::lock_guard<std::mutex> lock(mtx);
        if (choice == 1) {
            std::cout << "True Z Tray (cm): "; std::cin >> tz_tray;
            std::cout << "True Z Rim (cm): "; std::cin >> tz_rim;
            capture_flag = true;
        } else if (choice == 2) {
            std::cout << "Camera ID: "; std::cin >> cam_id;
        } else if (choice == 3) {
            running = false;
        }
    }
}

int main() {
    std::string config_file = "../midas_calibration.yaml";
    std::string points_file = "test_points.json";
    std::string snapshots_dir = "test_snapshots";
    
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

    std::vector<TestDataPoint> points;
    try {
        std::ifstream f(points_file);
        if (f.good()) {
            json j; f >> j;
            for (auto& item : j) {
                points.push_back({item["M_rim"], item["M_tray"], item["Z_tray"], item["Z_rim"], item["timestamp"]});
            }
            std::cout << "Loaded " << points.size() << " test points.\n";
        }
    } catch (...) {}

    AI* ai = AI::get_instance();
    
    std::mutex mtx;
    bool running = true, capture_flag = false;
    float tz_tray = 35.0f, tz_rim = 15.0f;
    int current_cam_id = camera_index;

    std::thread cli_thread(run_cli, std::ref(mtx), std::ref(running), std::ref(capture_flag), std::ref(tz_tray), std::ref(tz_rim), std::ref(current_cam_id));

    cv::VideoCapture cap;
    
    while (running) {
        mtx.lock();
        int requested_cam = current_cam_id;
        bool do_cap = capture_flag; capture_flag = false;
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
                cv::Mat dn;
                cv::normalize(depth, dn, 0, 255, cv::NORM_MINMAX, CV_8U);
                if (!boxes.empty()) {
                    cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
                    float mr = ai->midas_estimator->get_rim_depth(dn, bbox);
                    float mt = ai->midas_estimator->get_tray_depth(dn, roi);
                    if (mt > 0) {
                        long long ts = std::chrono::system_clock::now().time_since_epoch().count();
                        points.push_back({mr, mt, tz_tray, tz_rim, ts});
                        
                        json j = json::array();
                        for (auto& p : points) {
                            j.push_back({{"M_rim", p.m_rim}, {"M_tray", p.m_tray}, {"Z_tray", p.z_tray}, {"Z_rim", p.z_rim}, {"timestamp", p.timestamp}});
                        }
                        std::ofstream f(points_file);
                        f << j.dump(4);
                        
                        std::string img_path = snapshots_dir + "/test_tray" + std::to_string(tz_tray) + "cm_rim" + std::to_string(tz_rim) + "cm_" + std::to_string(ts) + ".jpg";
                        cv::imwrite(img_path, frame);
                        std::cout << "\nCaptured & Saved test point (" << points.size() << ")!\n";
                    }
                }
            }

            if (!boxes.empty()) {
                cv::Rect bbox(boxes[0].box.x, boxes[0].box.y, boxes[0].box.w, boxes[0].box.h);
                cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);
            }
            cv::rectangle(frame, roi, cv::Scalar(255, 0, 0), 2);
            
            cv::Mat depth_norm;
            cv::normalize(depth, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::Mat dv;
            cv::applyColorMap(depth_norm, dv, cv::COLORMAP_INFERNO);
            
            cv::Mat comb;
            cv::hconcat(frame, dv, comb);
            cv::imshow("Test Dataset Collection", comb);
            if (cv::waitKey(1) == 27) running = false; // ESC
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    cli_thread.join();
    return 0;
}
