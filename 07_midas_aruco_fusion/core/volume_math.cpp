/*******************************************************************************
 * volume_math.cpp
 * C++ port of 07_midas_aruco_fusion_new/core/volume_math.py
 ******************************************************************************/
#include "volume_math.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>

namespace VolumeMath {

float measureRimWidthPx(const cv::Mat& frame, const cv::Rect& bbox)
{
    float bbox_w = static_cast<float>(bbox.width);
    float bbox_h = static_cast<float>(bbox.height);

    if (bbox_w < 2.0f || bbox_h < 2.0f)
        return std::max(1.0f, bbox_w * 0.78f);

    int h_frame = frame.rows;
    int w_frame = frame.cols;

    /* Ukur dari TENGAH bbox (50% height), BUKAN dari atas (5%).
     *
     * ALASAN FUNDAMENTAL: Dengan kamera fisheye pitch=15° melihat ke bawah,
     * cup terlihat sebagai ELIPS dalam frame 2D. Rumus lebar elips pada
     * ketinggian y dari pusat: w(y) = D × √(1 - (y/b)²)
     *
     * Di strip ATAS (y ≈ 0.9b = 90% dari pusat ke tepi):
     *   w_atas = D × √(1 - 0.81) = D × 0.436 ≈ 34% dari diameter penuh
     *   → Otsu mendeteksi ~31-34% → D terukur = 3cm saja dari cup 7.6cm!
     *
     * Di TENGAH bbox (y = 0, pusat elips):
     *   w_tengah = D × √(1 - 0) = D × 1.0 = 100% → diameter PENUH ✓
     *
     * Dengan mengukur di tengah, kita mendapatkan diameter fisik yang benar. */
    int mid_y = bbox.y + bbox.height / 2;
    int strip_half = std::max(3, static_cast<int>(bbox_h * 0.04f));  // ±4% bbox_h
    int ry1 = std::max(0, mid_y - strip_half);
    int ry2 = std::min(h_frame, mid_y + strip_half);

    int rx1 = std::max(0, bbox.x);
    int rx2 = std::min(w_frame, bbox.x + bbox.width);

    if (ry2 <= ry1 || rx2 <= rx1)
        return std::max(1.0f, bbox_w * 0.78f);

    // Ambil strip tengah dari frame
    cv::Mat mid_strip = frame(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
    if (mid_strip.empty())
        return std::max(1.0f, bbox_w * 0.78f);

    // Konversi ke grayscale
    cv::Mat gray_strip;
    if (mid_strip.channels() == 3)
        cv::cvtColor(mid_strip, gray_strip, cv::COLOR_BGR2GRAY);
    else if (mid_strip.channels() == 4)
        cv::cvtColor(mid_strip, gray_strip, cv::COLOR_BGRA2GRAY);
    else
        gray_strip = mid_strip.clone();

    // Gaussian blur untuk mengurangi noise
    cv::Mat blurred;
    cv::GaussianBlur(gray_strip, blurred, cv::Size(5, 5), 0);

    /* THRESH_BINARY_INV: deteksi piksel gelap di strip tengah.
     * Di tengah bbox, cup itu sendiri (berwarna gelap relatif terhadap
     * background terang, atau kontras cup vs tray) terdeteksi. */
    cv::Mat mask;
    cv::threshold(blurred, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // Proyeksikan mask ke sumbu kolom (single-pass O(W×H))
    cv::Mat col_max;
    cv::reduce(mask, col_max, 0, cv::REDUCE_MAX, CV_8U);

    int left_col  = -1;
    int right_col = -1;
    const uchar* row_ptr = col_max.ptr<uchar>(0);
    for (int col = 0; col < col_max.cols; ++col) {
        if (row_ptr[col] > 0) {
            if (left_col < 0) left_col = col;
            right_col = col;
        }
    }

    /* Batas atas 85% bbox_w: cup fisik biasanya mengisi ~78% bbox_w
     * (dari kalibrasi: 7.6cm × 1009.4/13.5cm = 568px / 725px = 78.3%).
     * Cap 85% memberikan toleransi 7% di atas nilai kalibrasi. */
    const float MAX_RIM_FRACTION = 0.85f;
    const float max_rim = bbox_w * MAX_RIM_FRACTION;

    if (left_col >= 0 && right_col > left_col) {
        float rim_w = static_cast<float>(right_col - left_col);
        return std::max(1.0f, std::min(rim_w, max_rim));
    }

    /* Fallback: 78% bbox_w berdasarkan kalibrasi fisik:
     * D_rim=7.6cm, z_rim=13.5cm, focal=1009.4 → 568px = 78% dari 725px bbox */
    return std::max(1.0f, std::min(bbox_w * 0.78f, max_rim));
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

    /* TAPERED_CUP_FACTOR: faktor koreksi untuk cup berbentuk kerucut/trapesium.
     * Formula silinder V = π×r²×h menggunakan r_rim (jari-jari di bibir atas).
     * Untuk cup yang mengecil ke bawah, volume nyata < volume silinder penuh.
     *
     * Kalibrasi dengan cup fisik (D_rim=7.6cm, h=7.6cm, V_nyata=150ml):
     *   V_silinder = π × 3.8² × 7.6 = 344ml
     *   TAPERED_CUP_FACTOR = V_nyata / V_silinder = 150 / 344 = 0.436 ≈ 0.44
     *
     * → Ubah ke 1.0 untuk cup berbentuk silinder sempurna. */
    const double TAPERED_CUP_FACTOR = 0.44;
    return M_PI * (radius * radius) * h_cup_cm * TAPERED_CUP_FACTOR;
}

} // namespace VolumeMath
