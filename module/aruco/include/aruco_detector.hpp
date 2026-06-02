#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <string>
#include <vector>
#include <map>

struct ArucoResult {
    int id;
    std::vector<cv::Point2f> corners;
    float distance_cm;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    float euler_roll;
    float euler_pitch;
    float euler_yaw;
    cv::Point2f center;
    float reprojection_error;
};

struct BestDistanceResult {
    float distance_cm;
    int used_count;
    int rejected_count;
    std::vector<float> all_distances;
};

class ArucoDetector {
public:
    ArucoDetector(float marker_size_cm = 5.0f, const std::string& dictionary_name = "DICT_4X4_50", const std::string& params_path = "../calibration_params.yml");
    ~ArucoDetector() = default;

    std::vector<ArucoResult> detect(const cv::Mat& frame);
    cv::Mat annotate_frame(const cv::Mat& frame, const std::vector<ArucoResult>& results);
    BestDistanceResult get_best_distance(const std::vector<ArucoResult>& results, float max_reproj_error = 0.5f);

    std::string dictionary_name;
    float marker_size_cm;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;  // Exposed to allow zeroing after Moildev undistortion

private:
    cv::Ptr<cv::aruco::Dictionary> aruco_dict;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params;
    float f_pixel;

    void load_camera_calibration(const std::string& params_path);
    void rvec_to_euler(const cv::Vec3d& rvec, float& roll, float& pitch, float& yaw);
    float compute_reprojection_error(const std::vector<cv::Point2f>& corners_2d, const cv::Vec3d& rvec, const cv::Vec3d& tvec);
    
    std::map<std::string, cv::aruco::PREDEFINED_DICTIONARY_NAME> dict_map = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250}
    };
};
