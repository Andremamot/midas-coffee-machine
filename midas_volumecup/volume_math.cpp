#include "volume_math.hpp"
#include <cmath>
#include <numeric>

double VolumeMath::calculate_z_rim(double m_rim, double m_tray, double a, double b, double c, bool use_inverse) {
    double ratio = m_rim / m_tray;
    
    if (use_inverse) {
        if (ratio + b == 0) return 0.0;
        return (a / (ratio + b)) + c;
    }
    return a * (ratio * ratio) + b * ratio + c;
}

double VolumeMath::calculate_z_rim_alpha(double m_rim, double m_tray, double z_tray_live, double alpha) {
    if (m_tray <= 0 || z_tray_live <= 0) return 0.0;
    double ratio = m_rim / m_tray;
    if (ratio == 0) return 0.0;
    
    return (z_tray_live / ratio) * alpha;
}

std::tuple<double, double, double> VolumeMath::calculate_volume(double z_rim, double h_nozzle, double w_pixels, double focal_length) {
    if (z_rim <= 0 || focal_length <= 0) {
        return {0.0, 0.0, 0.0};
    }
        
    double h_cup = h_nozzle - z_rim;
    double w_real = (w_pixels * z_rim) / focal_length;
    
    double radius_cm = w_real / 2.0;
    double volume = M_PI * (radius_cm * radius_cm) * h_cup;
    
    return {h_cup, w_real, volume};
}

CamFeatures VolumeMath::extract_signal_features(const cv::Mat& frame_grayscale, const CameraConfig& cam_config) {
    CamFeatures feat;
    feat.valid_R_trans = false;
    feat.R_trans = 0.0;
    feat.dark_ratio = 0.0;

    std::vector<double> row_means(cam_config.H, 0.0);
    for (int r = 0; r < cam_config.H; r++) {
        cv::Scalar mean_val = cv::mean(frame_grayscale.row(r));
        row_means[r] = mean_val[0];
    }
    
    double I_max = 0.0;
    for (int r = cam_config.BRIGHT_ROW_START; r < cam_config.BRIGHT_ROW_END; r++) {
        if (row_means[r] > I_max) I_max = row_means[r];
    }
    
    // if bright_zone.size == 0
    if (cam_config.BRIGHT_ROW_END <= cam_config.BRIGHT_ROW_START) return feat;

    double threshold = I_max * 0.60;
    
    for (int r = cam_config.SEARCH_ROW_START; r < cam_config.H - 1; r++) {
        if (row_means[r] >= threshold && threshold > row_means[r + 1]) {
            double denom = row_means[r] - row_means[r + 1];
            if (std::abs(denom) > 1e-6) {
                feat.R_trans = r + (row_means[r] - threshold) / denom;
            } else {
                feat.R_trans = r;
            }
            feat.valid_R_trans = true;
            break;
        }
    }
            
    cv::Mat search_roi = frame_grayscale(cv::Rect(0, cam_config.SEARCH_ROW_START, cam_config.W, cam_config.H - cam_config.SEARCH_ROW_START));
    int dark_count = cv::countNonZero(search_roi < 40);
    feat.dark_ratio = static_cast<double>(dark_count) / search_roi.total();
    
    return feat;
}

double VolumeMath::measure_nozzle_height(const cv::Mat& frame_grayscale, const CameraConfig& cam_config, double m, double c, double m_b_norm, double c_b_norm, bool& valid) {
    CamFeatures feat = extract_signal_features(frame_grayscale, cam_config);
    if (!feat.valid_R_trans) {
        valid = false;
        return 0.0;
    }
        
    double H_A = m * feat.R_trans + c;
    double H_B = m_b_norm * feat.dark_ratio + c_b_norm;
    
    valid = true;
    return 0.70 * H_A + 0.30 * H_B;
}
