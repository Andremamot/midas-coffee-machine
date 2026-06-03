/*******************************************************************************
 * volume_math.cpp
 * C++ port of 07_midas_aruco_fusion_new/core/volume_math.py
 ******************************************************************************/
#include <volume/volume_math.h>

#include <cmath>
#include <algorithm>
#include <iostream>

namespace VolumeMath {

float measureRimWidthPx(const cv::Mat& frame, const cv::Rect& bbox)
{
    float bbox_w = static_cast<float>(bbox.width);
    float bbox_h = static_cast<float>(bbox.height);

    if (bbox_w < 2.0f || bbox_h < 2.0f)
        return std::max(1.0f, bbox_w);

    int h_frame = frame.rows;
    int w_frame = frame.cols;

    // Ketebalan strip rim: 10% tinggi bbox, minimal 4 piksel
    int rim_thickness = std::max(4, bbox.height / 10);

    // Clamp ke batas frame
    int ry1 = std::max(0, bbox.y);
    int ry2 = std::min(h_frame, bbox.y + rim_thickness);
    int rx1 = std::max(0, bbox.x);
    int rx2 = std::min(w_frame, bbox.x + bbox.width);

    if (ry2 <= ry1 || rx2 <= rx1)
        return std::max(1.0f, bbox_w);

    // Ambil strip rim dari frame
    cv::Mat rim_strip = frame(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
    if (rim_strip.empty())
        return std::max(1.0f, bbox_w);

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

    // Proyeksikan mask ke sumbu kolom (1 baris hasil) dengan satu operasi reduce.
    // cv::reduce + REDUCE_MAX: kolom dengan piksel aktif (>0) → nilai 255.
    // Ini O(W×H) single-pass, jauh lebih cepat dari O(H) × W calls.
    cv::Mat col_max;
    cv::reduce(mask, col_max, 0, cv::REDUCE_MAX, CV_8U);  // 1×W matrix

    int left_col  = -1;
    int right_col = -1;
    const uchar* row = col_max.ptr<uchar>(0);
    for (int col = 0; col < col_max.cols; ++col) {
        if (row[col] > 0) {
            if (left_col < 0) left_col = col;
            right_col = col;
        }
    }

    if (left_col >= 0 && right_col > left_col) {
        float rim_w = static_cast<float>(right_col - left_col);
        return std::max(1.0f, rim_w);
    }

    // Fallback: gunakan 50% tengah bbox sebagai estimasi konservatif
    return std::max(1.0f, bbox_w * 0.5f);
}

double calcDiameter(double rim_w_px, double z_rim_cm, double focal_px)
{
    if (z_rim_cm <= 0.0 || focal_px <= 0.0 || rim_w_px <= 0.0)
        return 0.0;
    return (rim_w_px * z_rim_cm) / focal_px;
}

double calcVolume(double h_cup_cm, double diameter_cm)
{
    if (h_cup_cm <= 0.0 || diameter_cm <= 0.0)
        return 0.0;
    double radius = diameter_cm / 2.0;
    return M_PI * (radius * radius) * h_cup_cm;
}

} // namespace VolumeMath
