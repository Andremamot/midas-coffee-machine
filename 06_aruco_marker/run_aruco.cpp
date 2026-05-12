#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "aruco_detector.hpp"

using json = nlohmann::json;

#ifdef _WIN32
#include <direct.h>
#define MKDIR(x) _mkdir(x)
#else
#define MKDIR(x) mkdir(x, 0777)
#endif

void print_result(const std::vector<ArucoResult>& results, const std::string& source = "") {
    std::string prefix = source.empty() ? "" : "[" + source + "] ";
    std::cout << "\n" << prefix << std::string(50, '=') << "\n";
    for (const auto& r : results) {
        json output = {
            {"id", r.id},
            {"distance_cm", r.distance_cm},
            {"center", {r.center.x, r.center.y}},
            {"euler_deg", {{"roll", r.euler_roll}, {"pitch", r.euler_pitch}, {"yaw", r.euler_yaw}}},
            {"reprojection_error", r.reprojection_error}
        };
        std::cout << output.dump(2) << "\n";
    }
    if (results.empty()) {
        std::cout << "  No ArUco marker detected\n";
    }
    std::cout << std::string(50, '=') << "\n\n";
}

void process_single_image(ArucoDetector& detector, const std::string& image_path, const std::string& output_dir) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Cannot read image: " << image_path << "\n";
        return;
    }
    std::cout << "Processing: " << image_path << "\n";
    auto results = detector.detect(img);
    size_t last_slash = image_path.find_last_of("/\\");
    std::string basename = (last_slash == std::string::npos) ? image_path : image_path.substr(last_slash + 1);
    print_result(results, basename);
    
    cv::Mat annotated = detector.annotate_frame(img, results);
    size_t last_dot = basename.find_last_of(".");
    if (last_dot != std::string::npos) basename = basename.substr(0, last_dot);
    
    std::string out_path = output_dir + "/aruco_" + basename + ".jpg";
    cv::imwrite(out_path, annotated);
    std::cout << "Visualization saved: " << out_path << "\n";
}

void run_live_camera(ArucoDetector& detector, int camera_index, bool lock_focus, int focus_value) {
    std::string screenshot_dir = "results/live_cam";
    MKDIR("results");
    MKDIR(screenshot_dir.c_str());
    
    cv::VideoCapture cap(camera_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera " << camera_index << "\n";
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    if (lock_focus) {
        cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
        cap.set(cv::CAP_PROP_FOCUS, focus_value);
    }
    
    std::cout << "Warming up...\n";
    for (int i = 0; i < 30; ++i) {
        cv::Mat f; cap.read(f);
    }
    
    std::cout << "\nLive camera started\nPress 'q' to exit | 's' screenshot\n";
    
    int stats_total = 0, stats_detected = 0;
    float stats_d_sum = 0, stats_d_min = 9999, stats_d_max = -9999;
    
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        if (frame.cols > 640) cv::resize(frame, frame, cv::Size(640, 480));
        
        auto results = detector.detect(frame);
        stats_total++;
        if (!results.empty()) {
            stats_detected++;
            auto best = detector.get_best_distance(results);
            if (best.used_count > 0) {
                stats_d_sum += best.distance_cm;
                stats_d_min = std::min(stats_d_min, best.distance_cm);
                stats_d_max = std::max(stats_d_max, best.distance_cm);
            }
        }
        
        cv::Mat annotated = detector.annotate_frame(frame, results);
        float session_avg = stats_detected > 0 ? stats_d_sum / stats_detected : 0;
        
        char buf[128];
        snprintf(buf, sizeof(buf), "Avg: %.2f cm | Frames: %d", session_avg, stats_total);
        cv::putText(annotated, buf, cv::Point(10, 460), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        
        cv::imshow("ArUco Target", annotated);
        int k = cv::waitKey(1) & 0xFF;
        if (k == 'q' || k == 27) break;
        else if (k == 's') {
            auto t = std::time(nullptr);
            auto tm = *std::localtime(&t);
            std::ostringstream oss; oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
            std::string ss_path = screenshot_dir + "/aruco_capture_" + oss.str() + ".jpg";
            cv::imwrite(ss_path, annotated);
            print_result(results, "screenshot");
            std::cout << "Saved: " << ss_path << "\n";
        }
    }
    cap.release();
    float avg = stats_detected > 0 ? stats_d_sum / stats_detected : 0;
    std::cout << "\n========== ARUCO REPORT ==========\n";
    std::cout << "Frames     : " << stats_total << "\n";
    std::cout << "Detected   : " << stats_detected << "\n";
    std::cout << "Avg Dist   : " << avg << " cm\n";
    std::cout << "==================================\n";
}

int main(int argc, char** argv) {
    bool gen_marker = false;
    std::string image_path;
    int camera_index = -1;
    float marker_size = 5.0f;
    std::string dictionary = "DICT_4X4_50";
    bool lock_focus = false;
    int focus_val = 0;
    std::string params_file = "../calibration_params.yml";
    std::string out_dir = "results";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--generate-marker") gen_marker = true;
        else if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--camera") {
            camera_index = 0;
            if (i + 1 < argc && isdigit(argv[i+1][0])) camera_index = std::stoi(argv[++i]);
        }
        else if (arg == "--marker-size" && i + 1 < argc) marker_size = std::stof(argv[++i]);
        else if (arg == "--dictionary" && i + 1 < argc) dictionary = argv[++i];
        else if (arg == "--lock-focus") lock_focus = true;
        else if (arg == "--focus-value" && i + 1 < argc) focus_val = std::stoi(argv[++i]);
        else if (arg == "--params" && i + 1 < argc) params_file = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) out_dir = argv[++i];
    }

    if (gen_marker) {
        std::cout << "Please use generate_marker executable directly.\n";
        return 0;
    }

    MKDIR(out_dir.c_str());
    std::cout << "Initializing ArUco detector...\n";
    ArucoDetector detector(marker_size, dictionary, params_file);

    if (!image_path.empty()) {
        process_single_image(detector, image_path, out_dir);
    } else if (camera_index >= 0) {
        run_live_camera(detector, camera_index, lock_focus, focus_val);
    } else {
        std::cerr << "Usage: --image <path> OR --camera [index] OR --generate-marker\n";
    }

    return 0;
}
