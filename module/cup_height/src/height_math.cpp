/*******************************************************************************
 * src/height_math.cpp
 * Port of 07_midas_aruco_fusion/core/height_math.py
 ******************************************************************************/
#include "cup_height/height_math.h"

#include <algorithm>
#include <cmath>

/*---------------------------------------------------------------------------*/
/* polyval — numpy.polyval equivalent                                         */
/* coeffs = [a_n, a_{n-1}, ..., a_1, a_0]                                   */
/*---------------------------------------------------------------------------*/
double HeightMath::polyval(const std::vector<double>& coeffs, double x)
{
    if (coeffs.empty()) return 0.0;
    double result = 0.0;
    for (const double c : coeffs) {
        result = result * x + c;
    }
    return result;
}

/*---------------------------------------------------------------------------*/
/* Mode 1: 1-Point K-Factor                                                   */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_1point(double m_rim, double m_tray,
                                      double z_tray,  double K)
{
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    const double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    const double h = z_tray * (1.0 - K / ratio);
    return h > 0.0 ? h : 0.0;
}

/*---------------------------------------------------------------------------*/
/* Mode 2: 2-Point Linear                                                     */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_2point(double m_rim, double m_tray,
                                      double z_tray,  double m, double c)
{
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    const double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    const double h = z_tray * (m * ratio + c);
    return h > 0.0 ? h : 0.0;
}

/*---------------------------------------------------------------------------*/
/* Mode 3: Z-Grid Polynomial                                                  */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_zgrid(double m_rim, double m_tray,
                                     double z_tray,
                                     const std::vector<double>& poly_K)
{
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    const double ratio  = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    const double K_live = polyval(poly_K, z_tray);
    const double h      = z_tray * (1.0 - K_live / ratio);
    return h > 0.0 ? h : 0.0;
}

/*---------------------------------------------------------------------------*/
/* Mode 4: BBox Area Compensated                                              */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_bbox(double m_rim, double m_tray,
                                    double z_tray,
                                    int bbox_x1, int bbox_y1,
                                    int bbox_x2, int bbox_y2,
                                    double m_ref, double c_ref,
                                    double ref_area)
{
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    const double ratio     = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    const double live_area = std::max(1.0, static_cast<double>(
                                          (bbox_x2 - bbox_x1) *
                                          (bbox_y2 - bbox_y1)));
    const double scale     = ref_area / live_area;
    const double m_adj     = m_ref * scale;
    const double h         = z_tray * (m_adj * ratio + c_ref);
    return h > 0.0 ? h : 0.0;
}

/*---------------------------------------------------------------------------*/
/* Mode 5: Geometric Projection (Z-Grid)                                      */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_geom(double z_tray,
                                    int bbox_x1, int bbox_y1,
                                    int bbox_x2, int bbox_y2,
                                    double focal_length_px,
                                    const std::vector<double>& poly_Kgeom)
{
    if (z_tray <= 0.0 || focal_length_px <= 0.0) return 0.0;
    const double bbox_h_px = std::max(1.0, static_cast<double>(bbox_y2 - bbox_y1));
    const double K_live    = polyval(poly_Kgeom, z_tray);
    const double h         = z_tray * (bbox_h_px / focal_length_px) * K_live;
    return h > 0.0 ? h : 0.0;
}

/*---------------------------------------------------------------------------*/
/* Mode 6: Bilateral Z-Grid                                                   */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_bilateral_zgrid(double m_rim, double m_tray,
                                               double z_tray,
                                               const std::vector<double>& poly_m,
                                               const std::vector<double>& poly_c)
{
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    const double ratio  = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    const double m_live = polyval(poly_m, z_tray);
    const double c_live = polyval(poly_c, z_tray);
    const double h      = z_tray * (m_live * ratio + c_live);
    return h > 0.0 ? h : 0.0;
}

/*---------------------------------------------------------------------------*/
/* Mode 7: Universal Analytic Geometry                                        */
/*---------------------------------------------------------------------------*/
double HeightMath::calc_height_analytic(double z_tray,
                                        int bbox_x1, int bbox_y1,
                                        int bbox_x2, int bbox_y2,
                                        double A, double B)
{
    if (z_tray <= 0.0) return 0.0;
    const double bbox_h = static_cast<double>(bbox_y2 - bbox_y1);
    const double denom  = bbox_h + B;
    if (std::fabs(denom) < 1e-4) return 0.0;
    const double h = (bbox_h * z_tray - A) / denom;
    return h > 0.0 ? h : 0.0;
}
