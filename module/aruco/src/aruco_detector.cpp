/**
 * @file aruco_detector.cpp
 * @brief Implementasi ArucoDetector — ArUco marker detection & pose estimation.
 *
 * Port dari: 06_aruco_marker/aruco_detector.cpp
 * Dipaketkan sebagai shared library (mod_aruco) dengan pola backup_module.
 */

#include <aruco/aruco_detector.h>

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <numeric>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

ArucoDetector::ArucoDetector(float              marker_size_cm,
                             const std::string& dictionary_name,
                             const std::string& params_path)
    : marker_size_cm(marker_size_cm)
    , dictionary_name(dictionary_name)
    , f_pixel_(800.0f)
{
    // Pilih dictionary
    if (dict_map_.find(dictionary_name) == dict_map_.end()) {
        std::cerr << "[mod_aruco] Unknown dictionary: " << dictionary_name
                  << " — fallback ke DICT_4X4_50\n";
        aruco_dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    } else {
        aruco_dict = cv::aruco::getPredefinedDictionary(dict_map_[dictionary_name]);
    }

    // Parameter detector (tuned untuk kondisi lapangan MIDAS)
    aruco_params = cv::aruco::DetectorParameters::create();
    aruco_params->adaptiveThreshConstant       = 7;
    aruco_params->adaptiveThreshWinSizeMin     = 3;
    aruco_params->adaptiveThreshWinSizeMax     = 23;
    aruco_params->adaptiveThreshWinSizeStep    = 10;
    aruco_params->cornerRefinementMethod       = cv::aruco::CORNER_REFINE_SUBPIX;
    aruco_params->cornerRefinementWinSize      = 5;
    aruco_params->cornerRefinementMaxIterations = 30;
    aruco_params->cornerRefinementMinAccuracy  = 0.1;

    load_camera_calibration(params_path);
}

// ─────────────────────────────────────────────────────────────────────────────
// load_camera_calibration
// ─────────────────────────────────────────────────────────────────────────────

void ArucoDetector::load_camera_calibration(const std::string& params_path)
{
    try {
        YAML::Node data = YAML::LoadFile(params_path);

        // Camera matrix (3×3)
        auto K_vec = data["camera_matrix_left"]
                         .as<std::vector<std::vector<float>>>();
        camera_matrix = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                camera_matrix.at<double>(i, j) = K_vec[i][j];

        // Distortion coefficients (1×5)
        auto D_vec = data["dist_coeff_left"]
                         .as<std::vector<std::vector<float>>>();
        dist_coeffs = cv::Mat(1, 5, CV_64F);
        for (int i = 0; i < 5; ++i)
            dist_coeffs.at<double>(0, i) = D_vec[0][i];

        // Focal length rata-rata
        f_pixel_ = static_cast<float>(
            (camera_matrix.at<double>(0, 0) + camera_matrix.at<double>(1, 1)) / 2.0);

        std::cout << "[mod_aruco] Camera calibration loaded from: " << params_path << "\n"
                  << "            f_pixel = " << f_pixel_ << " px\n";

    } catch (const std::exception& e) {
        std::cerr << "[mod_aruco] WARN: Failed to load calibration from '"
                  << params_path << "': " << e.what() << "\n"
                  << "            Using identity camera matrix (fallback).\n";
        camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        dist_coeffs   = cv::Mat::zeros(1, 5, CV_64F);
        f_pixel_       = 800.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// detect
// ─────────────────────────────────────────────────────────────────────────────

std::vector<ArucoResult> ArucoDetector::detect(const cv::Mat& frame)
{
    std::vector<ArucoResult> results;

    // Pastikan gambar grayscale
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame.clone();
    }

    // Deteksi marker
    std::vector<int>                       ids;
    std::vector<std::vector<cv::Point2f>>  corners, rejected;
    cv::aruco::detectMarkers(gray, aruco_dict, corners, ids, aruco_params);

    if (ids.empty()) return results;

    // Estimasi pose
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(
        corners, marker_size_cm, camera_matrix, dist_coeffs, rvecs, tvecs);

    for (size_t i = 0; i < ids.size(); ++i) {
        ArucoResult res;
        res.id             = ids[i];
        res.corners        = corners[i];
        res.rvec           = rvecs[i];
        res.tvec           = tvecs[i];
        res.distance_cm    = static_cast<float>(res.tvec[2]);  // Z-axis = jarak

        // Center point
        float sum_x = 0.0f, sum_y = 0.0f;
        for (const auto& pt : res.corners) { sum_x += pt.x; sum_y += pt.y; }
        res.center = cv::Point2f(sum_x / 4.0f, sum_y / 4.0f);

        // Euler angles
        rvec_to_euler(res.rvec, res.euler_roll, res.euler_pitch, res.euler_yaw);

        // Reprojection error
        res.reprojection_error =
            compute_reprojection_error(res.corners, res.rvec, res.tvec);

        results.push_back(res);
    }

    return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// annotate_frame
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat ArucoDetector::annotate_frame(const cv::Mat&                   frame,
                                      const std::vector<ArucoResult>& results)
{
    cv::Mat annotated = frame.clone();

    if (results.empty()) {
        cv::putText(annotated, "No ArUco marker detected",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 0, 255), 2);
        return annotated;
    }

    for (const auto& r : results) {
        // Gambar tepi marker
        for (int j = 0; j < 4; ++j) {
            cv::line(annotated,
                     r.corners[j], r.corners[(j + 1) % 4],
                     cv::Scalar(0, 255, 0), 2);
        }

        // Gambar axes pose
        cv::drawFrameAxes(annotated,
                          camera_matrix, dist_coeffs,
                          r.rvec, r.tvec,
                          marker_size_cm * 0.5f);

        // Label
        int cx = static_cast<int>(r.center.x);
        int cy = static_cast<int>(r.center.y);
        char buf[64];

        snprintf(buf, sizeof(buf), "ID:%d", r.id);
        cv::putText(annotated, buf, cv::Point(cx - 40, cy - 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

        snprintf(buf, sizeof(buf), "D:%.1fcm", r.distance_cm);
        cv::putText(annotated, buf, cv::Point(cx - 40, cy),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        snprintf(buf, sizeof(buf), "err:%.2fpx", r.reprojection_error);
        cv::putText(annotated, buf, cv::Point(cx - 40, cy + 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        snprintf(buf, sizeof(buf), "R:%.0f P:%.0f Y:%.0f",
                 r.euler_roll, r.euler_pitch, r.euler_yaw);
        cv::putText(annotated, buf, cv::Point(cx - 60, cy + 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180, 180, 255), 1);
    }

    return annotated;
}

// ─────────────────────────────────────────────────────────────────────────────
// get_best_distance
// ─────────────────────────────────────────────────────────────────────────────

BestDistanceResult ArucoDetector::get_best_distance(
    const std::vector<ArucoResult>& results,
    float                           max_reproj_error)
{
    BestDistanceResult best{0.0f, 0, 0, {}};
    if (results.empty()) return best;

    // Filter: buang marker dengan error terlalu besar
    std::vector<ArucoResult> valid;
    for (const auto& r : results) {
        if (r.reprojection_error <= max_reproj_error)
            valid.push_back(r);
    }
    best.rejected_count = static_cast<int>(results.size() - valid.size());

    if (valid.empty()) {
        // Fallback: gunakan marker dengan error terkecil
        auto min_it = std::min_element(
            results.begin(), results.end(),
            [](const ArucoResult& a, const ArucoResult& b) {
                return a.reprojection_error < b.reprojection_error;
            });
        best.distance_cm    = min_it->distance_cm;
        best.used_count     = 1;
        best.rejected_count = static_cast<int>(results.size()) - 1;
        best.all_distances.push_back(best.distance_cm);
        return best;
    }

    // Median dari semua jarak yang valid
    for (const auto& r : valid)
        best.all_distances.push_back(r.distance_cm);

    std::vector<float> sorted_dist = best.all_distances;
    std::sort(sorted_dist.begin(), sorted_dist.end());

    size_t n = sorted_dist.size();
    if (n % 2 == 0) {
        best.distance_cm = (sorted_dist[n / 2 - 1] + sorted_dist[n / 2]) / 2.0f;
    } else {
        best.distance_cm = sorted_dist[n / 2];
    }
    best.used_count = static_cast<int>(n);

    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: rvec_to_euler
// ─────────────────────────────────────────────────────────────────────────────

void ArucoDetector::rvec_to_euler(const cv::Vec3d& rvec,
                                  float& roll, float& pitch, float& yaw)
{
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                          R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;

    if (!singular) {
        roll  = static_cast<float>(std::atan2(R.at<double>(2, 1), R.at<double>(2, 2)));
        pitch = static_cast<float>(std::atan2(-R.at<double>(2, 0), sy));
        yaw   = static_cast<float>(std::atan2(R.at<double>(1, 0), R.at<double>(0, 0)));
    } else {
        roll  = static_cast<float>(std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1)));
        pitch = static_cast<float>(std::atan2(-R.at<double>(2, 0), sy));
        yaw   = 0.0f;
    }

    roll  *= 180.0f / static_cast<float>(CV_PI);
    pitch *= 180.0f / static_cast<float>(CV_PI);
    yaw   *= 180.0f / static_cast<float>(CV_PI);
}

// ─────────────────────────────────────────────────────────────────────────────
// Private: compute_reprojection_error
// ─────────────────────────────────────────────────────────────────────────────

float ArucoDetector::compute_reprojection_error(
    const std::vector<cv::Point2f>& corners_2d,
    const cv::Vec3d&                rvec,
    const cv::Vec3d&                tvec)
{
    float half = marker_size_cm / 2.0f;
    std::vector<cv::Point3f> obj_pts = {
        {-half,  half, 0.0f},
        { half,  half, 0.0f},
        { half, -half, 0.0f},
        {-half, -half, 0.0f}
    };

    std::vector<cv::Point2f> projected;
    cv::projectPoints(obj_pts, rvec, tvec,
                      camera_matrix, dist_coeffs, projected);

    float error_sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float dx = corners_2d[i].x - projected[i].x;
        float dy = corners_2d[i].y - projected[i].y;
        error_sum += std::sqrt(dx * dx + dy * dy);
    }
    return error_sum / 4.0f;
}
