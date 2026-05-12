#include "aruco_detector.hpp"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <numeric>
#include <algorithm>

ArucoDetector::ArucoDetector(float marker_size_cm, const std::string& dictionary_name, const std::string& params_path) 
    : marker_size_cm(marker_size_cm), dictionary_name(dictionary_name) {
    
    if (dict_map.find(dictionary_name) == dict_map.end()) {
        std::cerr << "Unknown dictionary: " << dictionary_name << "\n";
        aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    } else {
        aruco_dict = cv::aruco::getPredefinedDictionary(dict_map[dictionary_name]);
    }
    
    aruco_params = cv::aruco::DetectorParameters::create();
    aruco_params->adaptiveThreshConstant = 7;
    aruco_params->adaptiveThreshWinSizeMin = 3;
    aruco_params->adaptiveThreshWinSizeMax = 23;
    aruco_params->adaptiveThreshWinSizeStep = 10;
    aruco_params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    aruco_params->cornerRefinementWinSize = 5;
    aruco_params->cornerRefinementMaxIterations = 30;
    aruco_params->cornerRefinementMinAccuracy = 0.1;
    
    load_camera_calibration(params_path);
}

void ArucoDetector::load_camera_calibration(const std::string& params_path) {
    try {
        YAML::Node data = YAML::LoadFile(params_path);
        
        auto K_vec = data["camera_matrix_left"].as<std::vector<std::vector<float>>>();
        camera_matrix = cv::Mat(3, 3, CV_64F);
        for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) camera_matrix.at<double>(i,j) = K_vec[i][j];
        
        auto D_vec = data["dist_coeff_left"].as<std::vector<std::vector<float>>>();
        dist_coeffs = cv::Mat(1, 5, CV_64F);
        for(int i=0; i<5; ++i) dist_coeffs.at<double>(0,i) = D_vec[0][i];
        
        f_pixel = (camera_matrix.at<double>(0,0) + camera_matrix.at<double>(1,1)) / 2.0;
    } catch (...) {
        std::cerr << "Failed to load camera calibration from " << params_path << "\n";
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
        f_pixel = 800.0;
    }
}

std::vector<ArucoResult> ArucoDetector::detect(const cv::Mat& frame) {
    std::vector<ArucoResult> results;
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame.clone();
    }
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    cv::aruco::detectMarkers(gray, aruco_dict, corners, ids, aruco_params);
    
    if (ids.empty()) return results;
    
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, marker_size_cm, camera_matrix, dist_coeffs, rvecs, tvecs);
    
    for (size_t i = 0; i < ids.size(); ++i) {
        ArucoResult res;
        res.id = ids[i];
        res.corners = corners[i];
        res.rvec = rvecs[i];
        res.tvec = tvecs[i];
        res.distance_cm = static_cast<float>(res.tvec[2]);
        
        float sum_x = 0, sum_y = 0;
        for (const auto& pt : res.corners) { sum_x += pt.x; sum_y += pt.y; }
        res.center = cv::Point2f(sum_x / 4.0f, sum_y / 4.0f);
        
        rvec_to_euler(res.rvec, res.euler_roll, res.euler_pitch, res.euler_yaw);
        res.reprojection_error = compute_reprojection_error(res.corners, res.rvec, res.tvec);
        
        results.push_back(res);
    }
    
    return results;
}

cv::Mat ArucoDetector::annotate_frame(const cv::Mat& frame, const std::vector<ArucoResult>& results) {
    cv::Mat annotated = frame.clone();
    if (results.empty()) {
        cv::putText(annotated, "No ArUco marker detected", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        return annotated;
    }
    
    for (const auto& r : results) {
        for (int j = 0; j < 4; ++j) {
            cv::line(annotated, r.corners[j], r.corners[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        }
        cv::drawFrameAxes(annotated, camera_matrix, dist_coeffs, r.rvec, r.tvec, marker_size_cm * 0.5f);
        
        int cx = r.center.x, cy = r.center.y;
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "ID:%d", r.id);
        cv::putText(annotated, buffer, cv::Point(cx - 40, cy - 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        snprintf(buffer, sizeof(buffer), "D:%.1fcm", r.distance_cm);
        cv::putText(annotated, buffer, cv::Point(cx - 40, cy), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        snprintf(buffer, sizeof(buffer), "err:%.2fpx", r.reprojection_error);
        cv::putText(annotated, buffer, cv::Point(cx - 40, cy + 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        
        snprintf(buffer, sizeof(buffer), "R:%.0f P:%.0f Y:%.0f", r.euler_roll, r.euler_pitch, r.euler_yaw);
        cv::putText(annotated, buffer, cv::Point(cx - 60, cy + 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 255), 1);
    }
    return annotated;
}

void ArucoDetector::rvec_to_euler(const cv::Vec3d& rvec, float& roll, float& pitch, float& yaw) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    
    double sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) + R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6;
    
    if (!singular) {
        roll = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        yaw = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        roll = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
        pitch = std::atan2(-R.at<double>(2,0), sy);
        yaw = 0;
    }
    roll *= 180.0 / CV_PI;
    pitch *= 180.0 / CV_PI;
    yaw *= 180.0 / CV_PI;
}

float ArucoDetector::compute_reprojection_error(const std::vector<cv::Point2f>& corners_2d, const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    float half = marker_size_cm / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half, half, 0}, {half, half, 0}, {half, -half, 0}, {-half, -half, 0}
    };
    std::vector<cv::Point2f> projected;
    cv::projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs, projected);
    
    float error_sum = 0;
    for (int i = 0; i < 4; ++i) {
        float dx = corners_2d[i].x - projected[i].x;
        float dy = corners_2d[i].y - projected[i].y;
        error_sum += std::sqrt(dx*dx + dy*dy);
    }
    return error_sum / 4.0f;
}

BestDistanceResult ArucoDetector::get_best_distance(const std::vector<ArucoResult>& results, float max_reproj_error) {
    BestDistanceResult best {0, 0, 0, {}};
    if (results.empty()) return best;
    
    std::vector<ArucoResult> valid;
    for (const auto& r : results) {
        if (r.reprojection_error <= max_reproj_error) valid.push_back(r);
    }
    
    best.rejected_count = results.size() - valid.size();
    
    if (valid.empty()) {
        auto min_it = std::min_element(results.begin(), results.end(), [](const ArucoResult& a, const ArucoResult& b) {
            return a.reprojection_error < b.reprojection_error;
        });
        best.distance_cm = min_it->distance_cm;
        best.used_count = 1;
        best.all_distances.push_back(best.distance_cm);
        best.rejected_count = results.size() - 1;
        return best;
    }
    
    for (const auto& r : valid) best.all_distances.push_back(r.distance_cm);
    std::vector<float> sorted_dist = best.all_distances;
    std::sort(sorted_dist.begin(), sorted_dist.end());
    
    if (sorted_dist.size() % 2 == 0) {
        best.distance_cm = (sorted_dist[sorted_dist.size() / 2 - 1] + sorted_dist[sorted_dist.size() / 2]) / 2.0f;
    } else {
        best.distance_cm = sorted_dist[sorted_dist.size() / 2];
    }
    best.used_count = valid.size();
    
    return best;
}
