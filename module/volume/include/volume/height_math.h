/**
 * @file height_math.h
 * @brief Struct BBox dan fungsi math untuk estimasi tinggi gelas.
 *
 * Digunakan oleh volume_math.h dan cup_volume_estimator.h.
 * Caller (proyek lain) menggunakan ini untuk membentuk BBox dari YOLO output.
 */

#pragma once

#include <vector>
#include <cmath>

namespace fusion {

/**
 * @brief Bounding box dari YOLO detector.
 *
 * Koordinat dalam piksel, origin di pojok kiri-atas frame.
 */
struct BBox {
    int x1, y1, x2, y2;
};

/**
 * @brief Evaluasi polinomial dengan metode Horner (pengganti np.polyval).
 * Koefisien dengan derajat tertinggi pertama (konvensi NumPy).
 */
inline double polyval(const std::vector<double>& coeffs, double x) {
    if (coeffs.empty()) return 0.0;
    double result = coeffs[0];
    for (size_t i = 1; i < coeffs.size(); ++i)
        result = result * x + coeffs[i];
    return result;
}

// ── Height calibration functions (7 types) ───────────────────────────────────
double calc_height_1point(double m_rim, double m_tray, double z_tray, double K);
double calc_height_2point(double m_rim, double m_tray, double z_tray, double m, double c);
double calc_height_zgrid(double m_rim, double m_tray, double z_tray, const std::vector<double>& poly_K);
double calc_height_bbox(double m_rim, double m_tray, double z_tray,
                        const BBox& bbox, double m_ref, double c_ref, double ref_area);
double calc_height_geom(double z_tray, const BBox& bbox, double focal_length_px,
                        const std::vector<double>& poly_Kgeom);
double calc_height_bilateral_zgrid(double m_rim, double m_tray, double z_tray,
                                   const std::vector<double>& poly_m,
                                   const std::vector<double>& poly_c);
double calc_height_analytic(double z_tray, const BBox& bbox, double A, double B);

}  // namespace fusion
