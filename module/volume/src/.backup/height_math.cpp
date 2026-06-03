/**
 * @file height_math.cpp
 * @brief Implementasi 7 mode kalibrasi tinggi gelas.
 * Port dari: 07_midas_aruco_fusion/core/height_math.py
 */

#include <volume/height_math.h>
#include <algorithm>
#include <iostream>

namespace fusion {

double calc_height_1point(double m_rim, double m_tray, double z_tray, double K) {
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    double h = z_tray * (1.0 - K / ratio);
    return (h > 0.0) ? h : 0.0;
}

double calc_height_2point(double m_rim, double m_tray, double z_tray,
                          double m, double c) {
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    double h = z_tray * (m * ratio + c);
    return (h > 0.0) ? h : 0.0;
}

double calc_height_zgrid(double m_rim, double m_tray, double z_tray,
                         const std::vector<double>& poly_K) {
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    double K_live = polyval(poly_K, z_tray);
    double h = z_tray * (1.0 - K_live / ratio);
    return (h > 0.0) ? h : 0.0;
}

double calc_height_bbox(double m_rim, double m_tray, double z_tray,
                        const BBox& bbox, double m_ref, double c_ref,
                        double ref_area) {
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    double live_area = std::max(1.0,
        static_cast<double>((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)));
    double scale = ref_area / live_area;
    double h = z_tray * (m_ref * scale * ratio + c_ref);
    return (h > 0.0) ? h : 0.0;
}

double calc_height_geom(double z_tray, const BBox& bbox,
                        double focal_length_px,
                        const std::vector<double>& poly_Kgeom) {
    if (z_tray <= 0.0 || focal_length_px <= 0.0) return 0.0;
    double bbox_h_px = std::max(1.0, static_cast<double>(bbox.y2 - bbox.y1));
    double K_live    = polyval(poly_Kgeom, z_tray);

    /* PENTING: K_live negatif berarti z_tray berada di luar range kalibrasi.
     * Hal ini terjadi jika poly_Kgeom dikalibrasi dengan focal ArUco yang
     * berbeda dari saat inferensi (misal: focal lama 660px vs baru 1009px).
     * Solusi: kalibrasi ulang dengan focal yang sudah dikoreksi. */
    if (K_live <= 0.0) {
        std::cerr << "[HEIGHT-WARN] K_live=" << K_live
                  << " negatif/nol: z_tray=" << z_tray
                  << " di luar range kalibrasi — perlu re-kalibrasi!\n";
        return 0.0;
    }

    double h = z_tray * (bbox_h_px / focal_length_px) * K_live;
    return (h > 0.0) ? h : 0.0;
}


double calc_height_bilateral_zgrid(double m_rim, double m_tray, double z_tray,
                                   const std::vector<double>& poly_m,
                                   const std::vector<double>& poly_c) {
    if (m_tray <= 0.0 || z_tray <= 0.0) return 0.0;
    double ratio = m_rim / m_tray;
    if (ratio <= 0.0) return 0.0;
    double m_live = polyval(poly_m, z_tray);
    double c_live = polyval(poly_c, z_tray);
    double h = z_tray * (m_live * ratio + c_live);
    return (h > 0.0) ? h : 0.0;
}

double calc_height_analytic(double z_tray, const BBox& bbox,
                            double A, double B) {
    if (z_tray <= 0.0) return 0.0;
    double bbox_h = static_cast<double>(bbox.y2 - bbox.y1);
    double denom = bbox_h + B;
    if (std::abs(denom) < 1e-4) return 0.0;
    double h = (bbox_h * z_tray - A) / denom;
    return (h > 0.0) ? h : 0.0;
}

}  // namespace fusion
