/**
 * @file moil_undistorter.cpp
 * @brief Fisheye undistortion implementation via libmoildev.a
 */

#include "moil_undistorter.h"
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// --- Minimal Moildev Class Declaration for Linker ---
class Moildev {
public:
    Moildev();
    void Config(std::string cameraName, double cameraSensorWidth, double cameraSensorHeight,
                double iCx, double iCy, double ratio, double imageWidth, double imageHeight,
                double calibrationRatio, double parameter0, double parameter1, double parameter2,
                double parameter3, double parameter4, double parameter5);
    void AnyPointM2(float *mapX, float *mapY, double pitch, double yaw, double zoom);
    int getImageWidth();
    int getImageHeight();
};

namespace fusion {

// Internal Moildev instance (using raw pointer to avoid header dependency)
static Moildev* g_moildev = nullptr;

MoilUndistorter::MoilUndistorter(const std::string& camera_params_json,
                                 const std::string& camera_name,
                                 int mode)
    : mode_(mode)
{
    if (!g_moildev) g_moildev = new Moildev();
    load_params_with_name(camera_params_json, camera_name);
    if (focal_length_ > 0) {
        update_maps(0, 0, 0, 1.0);
        maps_ready_ = true;
    }
}

void MoilUndistorter::load_params(const std::string& json_path) {
    load_params_with_name(json_path, "");
}

void MoilUndistorter::load_params_with_name(const std::string& json_path, const std::string& camera_name) {
    try {
        std::ifstream f(json_path);
        if (!f.is_open()) {
            std::cerr << "[Moildev] Cannot open: " << json_path << "\n";
            return;
        }
        nlohmann::json j_all;
        f >> j_all;
        
        // Find a suitable profile
        nlohmann::json j;
        std::string profile = camera_name;
        if (!profile.empty() && j_all.contains(profile)) {
            j = j_all[profile];
        } else if (j_all.contains("entaniya_vr220_1")) {
            profile = "entaniya_vr220_1";
            j = j_all[profile];
        } else {
            for (auto& [key, val] : j_all.items()) { j = val; profile = key; break; }
        }

        focal_length_ = j.value("parameter5", 0.0) / j.value("calibrationRatio", 1.0);
        image_width_  = j.value("imageWidth", 0);
        image_height_ = j.value("imageHeight", 0);
        adjusted_focal_ = focal_length_;

        if (g_moildev) {
            g_moildev->Config(
                profile,
                j.value("cameraSensorWidth", 1.0),
                j.value("cameraSensorHeight", 1.0),
                j.value("iCx", image_width_/2.0),
                j.value("iCy", image_height_/2.0),
                j.value("ratio", 1.0),
                image_width_,
                image_height_,
                j.value("calibrationRatio", 1.0),
                j.value("parameter0", 0.0),
                j.value("parameter1", 0.0),
                j.value("parameter2", 0.0),
                j.value("parameter3", 0.0),
                j.value("parameter4", 0.0),
                j.value("parameter5", 500.0)
            );
        }

        std::cout << "[Moildev] Configured: " << profile << " f=" << focal_length_ << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Moildev] Error loading params: " << e.what() << "\n";
    }
}

void MoilUndistorter::generate_maps(double pitch, double yaw, double roll,
                                     double moil_zoom) {
    (void)roll; // Moildev AnyPointM2 typically ignores roll
    if (image_width_ <= 0 || image_height_ <= 0 || !g_moildev) return;

    map_x_.create(image_height_, image_width_, CV_32FC1);
    map_y_.create(image_height_, image_width_, CV_32FC1);

    // Call actual Moildev library
    g_moildev->AnyPointM2(
        (float*)map_x_.data,
        (float*)map_y_.data,
        pitch,
        yaw,
        moil_zoom
    );

    adjusted_focal_ = focal_length_ * moil_zoom;
}

std::pair<double, double> MoilUndistorter::split_zoom(double total_zoom) {
    if (total_zoom <= MAX_MOIL_ZOOM) {
        return {total_zoom, 1.0};
    }
    return {MAX_MOIL_ZOOM, total_zoom / MAX_MOIL_ZOOM};
}

void MoilUndistorter::rescale_maps(const cv::Size& stream_size) {
    if (map_x_.empty()) return;

    double sx = static_cast<double>(stream_size.width) / map_x_.cols;
    double sy = static_cast<double>(stream_size.height) / map_x_.rows;

    cv::resize(map_x_, scaled_map_x_, stream_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(map_y_, scaled_map_y_, stream_size, 0, 0, cv::INTER_LINEAR);

    scaled_map_x_ *= static_cast<float>(sx);
    scaled_map_y_ *= static_cast<float>(sy);
    stream_size_ = stream_size;
}

cv::Mat MoilUndistorter::digital_crop(const cv::Mat& frame,
                                       double digital_zoom) const {
    if (digital_zoom <= 1.0) return frame;

    int cw = static_cast<int>(frame.cols / digital_zoom);
    int ch = static_cast<int>(frame.rows / digital_zoom);
    int cx = (frame.cols - cw) / 2;
    int cy = (frame.rows - ch) / 2;

    cv::Mat cropped = frame(cv::Range(cy, cy + ch), cv::Range(cx, cx + cw));
    cv::Mat result;
    cv::resize(cropped, result, frame.size(), 0, 0, cv::INTER_LINEAR);
    return result;
}

cv::Mat MoilUndistorter::undistort(const cv::Mat& frame) {
    if (frame.empty() || !maps_ready_) return frame.clone();

    std::lock_guard<std::mutex> lock(map_mutex_);

    cv::Size frame_size(frame.cols, frame.rows);
    if (stream_size_ != frame_size) {
        rescale_maps(frame_size);
    }

    auto [moil_z, digi_z] = split_zoom(zoom_);

    cv::Mat remapped;
    cv::remap(frame, remapped, scaled_map_x_, scaled_map_y_,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    if (digi_z > 1.0) {
        remapped = digital_crop(remapped, digi_z);
    }

    return remapped;
}

void MoilUndistorter::update_maps(double pitch, double yaw,
                                   double roll, double zoom) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    pitch_ = pitch;
    yaw_ = yaw;
    roll_ = roll;
    zoom_ = zoom;

    auto [moil_z, digi_z] = split_zoom(zoom);
    generate_maps(pitch, yaw, roll, moil_z);
    stream_size_ = cv::Size(0, 0);  // Force rescale
}

cv::Matx33d MoilUndistorter::build_aruco_camera_matrix(int w, int h) const {
    double fx = adjusted_focal_;
    double fy = fx;
    double cx = w / 2.0;
    double cy = h / 2.0;
    return cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

} // namespace fusion

