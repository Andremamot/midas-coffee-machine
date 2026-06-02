/**
 * @file volume_math.cpp
 * @brief Implementasi fungsi estimasi volume gelas.
 * Port dari: 07_midas_aruco_fusion/core/volume_math.py
 */

#include <volume/volume_math.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace fusion {

double measure_rim_width_px(const cv::Mat& frame, const BBox& bbox) {
    int bbox_w = bbox.x2 - bbox.x1;
    int bbox_h = bbox.y2 - bbox.y1;

    if (bbox_w < 2 || bbox_h < 2)
        return std::max(1.0, static_cast<double>(bbox_w));

    // Rim strip: 10% atas bbox, minimum 4 piksel
    int rim_thickness = std::max(4, bbox_h / 10);

    int ry1 = std::max(0, bbox.y1);
    int ry2 = std::min(frame.rows, bbox.y1 + rim_thickness);
    int rx1 = std::max(0, bbox.x1);
    int rx2 = std::min(frame.cols, bbox.x2);

    if (ry2 <= ry1 || rx2 <= rx1)
        return std::max(1.0, static_cast<double>(bbox_w));

    cv::Mat rim_strip = frame(cv::Range(ry1, ry2), cv::Range(rx1, rx2));
    if (rim_strip.empty())
        return std::max(1.0, static_cast<double>(bbox_w));

    // Grayscale + Otsu threshold
    cv::Mat gray;
    if (rim_strip.channels() == 3)
        cv::cvtColor(rim_strip, gray, cv::COLOR_BGR2GRAY);
    else
        gray = rim_strip;

    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Cari kolom kiri-kanan yang ada piksel
    int first_col = -1, last_col = -1;
    for (int col = 0; col < mask.cols; ++col) {
        for (int row = 0; row < mask.rows; ++row) {
            if (mask.at<uchar>(row, col) > 0) {
                if (first_col < 0) first_col = col;
                last_col = col;
                break;
            }
        }
    }

    if (first_col >= 0 && last_col > first_col)
        return std::max(1.0, static_cast<double>(last_col - first_col));

    // Fallback: 50% bbox width
    return std::max(1.0, static_cast<double>(bbox_w) * 0.5);
}

double calc_diameter(double rim_w_px, double z_rim_cm, double focal_px) {
    if (z_rim_cm <= 0.0 || focal_px <= 0.0 || rim_w_px <= 0.0) return 0.0;
    return (rim_w_px * z_rim_cm) / focal_px;
}

double calc_volume(double h_cup_cm, double diameter_cm) {
    if (h_cup_cm <= 0.0 || diameter_cm <= 0.0) return 0.0;
    double r = diameter_cm / 2.0;
    return M_PI * r * r * h_cup_cm;
}

}  // namespace fusion
