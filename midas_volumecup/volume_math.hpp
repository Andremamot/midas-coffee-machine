#pragma once
#include <opencv2/opencv.hpp>
#include <tuple>
#include "camera_config.hpp"

struct CamFeatures {
    double R_trans;
    double dark_ratio;
    bool valid_R_trans;
};

class VolumeMath {
public:
    static double calculate_z_rim(double m_rim, double m_tray, double a, double b, double c, bool use_inverse = true);
    static double calculate_z_rim_alpha(double m_rim, double m_tray, double z_tray_live, double alpha);
    static std::tuple<double, double, double> calculate_volume(double z_rim, double h_nozzle, double w_pixels, double focal_length);
    static CamFeatures extract_signal_features(const cv::Mat& frame_grayscale, const CameraConfig& cam_config);
    static double measure_nozzle_height(const cv::Mat& frame_grayscale, const CameraConfig& cam_config, double m, double c, double m_b_norm, double c_b_norm, bool& valid);
};
