/*******************************************************************************
 * height_math.hpp
 * Port of 07_midas_aruco_fusion/core/height_math.py
 * Cup height computation functions for all calibration modes (1-7)
 ******************************************************************************/
#pragma once

#include <vector>

/**
 * @brief Static helper class with all cup height estimation formulas.
 *
 * Each method corresponds exactly to a Python function in height_math.py.
 * All measurements use centimetres; depth ratios are dimensionless.
 */
class HeightMath {
public:
    /**
     * Mode 1 – 1-Point K-Factor calibration.
     * H = z_tray * (1 - K / ratio)
     */
    static double calc_height_1point(double m_rim, double m_tray,
                                     double z_tray,  double K);

    /**
     * Mode 2 – 2-Point Linear calibration.
     * H = z_tray * (m * ratio + c)
     */
    static double calc_height_2point(double m_rim, double m_tray,
                                     double z_tray,  double m, double c);

    /**
     * Mode 3 – Z-Grid polynomial calibration.
     * K_live = polyval(poly_K, z_tray)
     * H = z_tray * (1 - K_live / ratio)
     */
    static double calc_height_zgrid(double m_rim, double m_tray,
                                    double z_tray,
                                    const std::vector<double>& poly_K);

    /**
     * Mode 4 – BBox-area compensated calibration.
     * scale = ref_area / live_area
     * H = z_tray * (m_ref * scale * ratio + c_ref)
     */
    static double calc_height_bbox(double m_rim, double m_tray,
                                   double z_tray,
                                   int bbox_x1, int bbox_y1,
                                   int bbox_x2, int bbox_y2,
                                   double m_ref, double c_ref,
                                   double ref_area);

    /**
     * Mode 5 – Geometric projection (Z-Grid) calibration.
     * K_live = polyval(poly_Kgeom, z_tray)
     * H = z_tray * (bbox_h_px / focal_length_px) * K_live
     */
    static double calc_height_geom(double z_tray,
                                   int bbox_x1, int bbox_y1,
                                   int bbox_x2, int bbox_y2,
                                   double focal_length_px,
                                   const std::vector<double>& poly_Kgeom);

    /**
     * Mode 6 – Bilateral Z-Grid calibration.
     * m_live = polyval(poly_m, z_tray)
     * c_live = polyval(poly_c, z_tray)
     * H = z_tray * (m_live * ratio + c_live)
     */
    static double calc_height_bilateral_zgrid(double m_rim, double m_tray,
                                              double z_tray,
                                              const std::vector<double>& poly_m,
                                              const std::vector<double>& poly_c);

    /**
     * Mode 7 – Universal Analytic Geometry calibration.
     * H = (bbox_h * z_tray - A) / (bbox_h + B)
     */
    static double calc_height_analytic(double z_tray,
                                       int bbox_x1, int bbox_y1,
                                       int bbox_x2, int bbox_y2,
                                       double A, double B);

    /**
     * Evaluate a polynomial at x: coeffs are [a_n, ..., a_1, a_0]
     * (same convention as numpy.polyval).
     */
    static double polyval(const std::vector<double>& coeffs, double x);
};
