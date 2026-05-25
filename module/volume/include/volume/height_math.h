/**
 * @file height_math.h
 * @brief Struct BBox dan fungsi math untuk estimasi tinggi gelas (7 mode kalibrasi).
 *
 * Include file ini di proyek yang membutuhkan kalkulasi tinggi gelas:
 *   #include <volume/height_math.h>
 *
 * Port dari: 07_midas_aruco_fusion/core/height_math.py
 * Dipaketkan sebagai bagian dari mod_volume dengan pola backup_module.
 */

#pragma once

#include <vector>
#include <cmath>

namespace fusion {

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Bounding box dari YOLO detector.
 *
 * Koordinat dalam piksel, origin di pojok kiri-atas frame.
 */
struct BBox {
    int x1, y1, x2, y2;
};

// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Evaluasi polinomial dengan metode Horner (pengganti np.polyval).
 *
 * Koefisien dengan derajat tertinggi pertama (konvensi NumPy):
 *   coeffs = [a_n, a_{n-1}, ..., a_1, a_0]
 *
 * Sama seperti numpy.polyval(coeffs, x).
 */
inline double polyval(const std::vector<double>& coeffs, double x) {
    if (coeffs.empty()) return 0.0;
    double result = coeffs[0];
    for (size_t i = 1; i < coeffs.size(); ++i)
        result = result * x + coeffs[i];
    return result;
}

// ── Height calibration functions (7 types) ────────────────────────────────────

/**
 * Mode 1 — 1-Point K-Factor calibration.
 * H = z_tray * (1 - K / ratio),  ratio = m_rim / m_tray
 */
double calc_height_1point(double m_rim, double m_tray, double z_tray, double K);

/**
 * Mode 2 — 2-Point Linear calibration.
 * H = z_tray * (m * ratio + c),  ratio = m_rim / m_tray
 */
double calc_height_2point(double m_rim, double m_tray, double z_tray,
                           double m, double c);

/**
 * Mode 3 — Z-Grid Polynomial calibration.
 * K_live = polyval(poly_K, z_tray)
 * H = z_tray * (1 - K_live / ratio)
 */
double calc_height_zgrid(double m_rim, double m_tray, double z_tray,
                          const std::vector<double>& poly_K);

/**
 * Mode 4 — BBox-area compensated calibration.
 * scale = ref_area / live_area
 * H = z_tray * (m_ref * scale * ratio + c_ref)
 */
double calc_height_bbox(double m_rim, double m_tray, double z_tray,
                         const BBox& bbox, double m_ref, double c_ref,
                         double ref_area);

/**
 * Mode 5 — Geometric Projection (Z-Grid) calibration.
 * K_live = polyval(poly_Kgeom, z_tray)
 * H = z_tray * (bbox_h_px / focal_length_px) * K_live
 */
double calc_height_geom(double z_tray, const BBox& bbox,
                         double focal_length_px,
                         const std::vector<double>& poly_Kgeom);

/**
 * Mode 6 — Bilateral Z-Grid calibration.
 * m_live = polyval(poly_m, z_tray)
 * c_live = polyval(poly_c, z_tray)
 * H = z_tray * (m_live * ratio + c_live)
 */
double calc_height_bilateral_zgrid(double m_rim, double m_tray, double z_tray,
                                    const std::vector<double>& poly_m,
                                    const std::vector<double>& poly_c);

/**
 * Mode 7 — Universal Analytic Geometry calibration.
 * H = (bbox_h * z_tray - A) / (bbox_h + B)
 */
double calc_height_analytic(double z_tray, const BBox& bbox,
                              double A, double B);

}  // namespace fusion
