/**
 * @file volume_math.cpp
 * @brief Implementasi fungsi estimasi volume gelas.
 *
 * Port dari: 07_midas_aruco_fusion/core/volume_math.cpp (via backup_module/volume)
 * Dipaketkan sebagai bagian dari mod_volume dengan pola backup_module.
 */

#include <volume/volume_math.h>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace fusion {

// ─────────────────────────────────────────────────────────────────────────────
// measure_rim_width_px
// ─────────────────────────────────────────────────────────────────────────────

double measure_rim_width_px(const cv::Mat& frame, const BBox& bbox)
{
    double bbox_w = static_cast<double>(bbox.x2 - bbox.x1);
    double bbox_h = static_cast<double>(bbox.y2 - bbox.y1);

    if (bbox_w < 2.0 || bbox_h < 2.0)
        return std::max(1.0, bbox_w);

    int h_frame = frame.rows;
    int w_frame = frame.cols;

    // Ketebalan strip rim: 10% tinggi bbox, minimal 4 piksel
    int rim_thickness = std::max(4, static_cast<int>(bbox_h / 10.0));

    // Clamp ke batas frame
    int ry1 = std::max(0, bbox.y1);
    int ry2 = std::min(h_frame, bbox.y1 + rim_thickness);
    int rx1 = std::max(0, bbox.x1);
    int rx2 = std::min(w_frame, bbox.x2);

    if (ry2 <= ry1 || rx2 <= rx1)
        return std::max(1.0, bbox_w);

    // Ambil strip rim dari frame
    cv::Mat rim_strip = frame(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
    if (rim_strip.empty())
        return std::max(1.0, bbox_w);

    // Konversi ke grayscale
    cv::Mat gray_strip;
    if (rim_strip.channels() == 3)
        cv::cvtColor(rim_strip, gray_strip, cv::COLOR_BGR2GRAY);
    else if (rim_strip.channels() == 4)
        cv::cvtColor(rim_strip, gray_strip, cv::COLOR_BGRA2GRAY);
    else
        gray_strip = rim_strip.clone();

    // Otsu threshold — adaptif terhadap kondisi pencahayaan
    cv::Mat mask;
    cv::threshold(gray_strip, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Cari kolom paling kiri dan kanan yang memiliki piksel aktif
    int left_col  = -1;
    int right_col = -1;
    for (int col = 0; col < mask.cols; ++col) {
        cv::Mat col_data = mask.col(col);
        if (cv::countNonZero(col_data) > 0) {
            if (left_col < 0) left_col = col;
            right_col = col;
        }
    }

    if (left_col >= 0 && right_col > left_col) {
        double rim_w = static_cast<double>(right_col - left_col);
        return std::max(1.0, rim_w);
    }

    // Fallback: gunakan 50% tengah bbox sebagai estimasi konservatif
    return std::max(1.0, bbox_w * 0.5);
}

// ─────────────────────────────────────────────────────────────────────────────
// calc_diameter
// ─────────────────────────────────────────────────────────────────────────────

double calc_diameter(double rim_w_px, double z_rim_cm, double focal_px)
{
    if (z_rim_cm <= 0.0 || focal_px <= 0.0 || rim_w_px <= 0.0)
        return 0.0;
    return (rim_w_px * z_rim_cm) / focal_px;
}

// ─────────────────────────────────────────────────────────────────────────────
// calc_volume
// ─────────────────────────────────────────────────────────────────────────────

double calc_volume(double h_cup_cm, double diameter_cm)
{
    if (h_cup_cm <= 0.0 || diameter_cm <= 0.0)
        return 0.0;
    double radius = diameter_cm / 2.0;
    return M_PI * (radius * radius) * h_cup_cm;
}

}  // namespace fusion
