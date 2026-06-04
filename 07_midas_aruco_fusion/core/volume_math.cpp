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
        return std::max(1.0f, bbox_w * 0.82f);

    /* ── Kalibrasi Geometris Langsung ──────────────────────────────────────
     *
     * MASALAH IMAGE-PROCESSING (semua pendekatan Otsu sudah dicoba & gagal):
     *   Baik dari strip ATAS maupun strip TENGAH, Otsu SELALU mengembalikan
     *   ~40% dari bbox_w. Ini terjadi karena interior cup gelap hanya mengisi
     *   ~40% dari lebar bounding box YOLO. Otsu mendeteksi kontras interior
     *   vs background tray, bukan tepi fisik luar cup.
     *
     * SOLUSI: Gunakan kalibrasi geometris langsung tanpa image-processing.
     *   Dari pengukuran fisik cup kopi:
     *     D_rim  = 7.6 cm  (diameter rim diukur langsung dengan penggaris)
     *     focal  = 1009.4 px (dari konfigurasi moildev zoom=2)
     *     z_rim  ≈ 13.5 cm (jarak kamera ke bibir cup saat di atas tray)
     *
     *   rim_w_expected = D × focal / z_rim
     *                  = 7.6 × 1009.4 / 13.5 ≈ 568 px
     *
     *   YOLO bbox_w tipikal ≈ 630–725 px
     *     BBOX_FILL_RATIO = 568 / 630 = 0.90  (saat cup kecil di frame)
     *     BBOX_FILL_RATIO = 568 / 725 = 0.78  (saat cup besar di frame)
     *   Rata-rata geometrik: 0.82 → digunakan sebagai nilai terbaik.
     *
     * Estimasi diameter:
     *   D = (bbox_w × 0.82 × z_rim) / focal
     *     ≈ (630 × 0.82 × 13.5) / 1009.4 ≈ 6.9 cm  ✓
     *     ≈ (725 × 0.82 × 13.5) / 1009.4 ≈ 7.9 cm  ✓ (toleransi 4%) */

    const float BBOX_FILL_RATIO = 0.82f;
    return std::max(1.0f, bbox_w * BBOX_FILL_RATIO);
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

    /* TAPERED_CUP_FACTOR: koreksi bentuk trapesium cup kopi.
     *
     * V_silinder penuh (D=7.6cm, h=7.6cm) = π × 3.8² × 7.6 = 344 ml
     * V_nyata cup kopi                     = 150 ml
     * TAPERED_CUP_FACTOR = 150 / 344 = 0.436 ≈ 0.44
     *
     * Ubah ke 1.0 untuk cup silindris sempurna. */
    const double TAPERED_CUP_FACTOR = 0.44;
    return M_PI * (radius * radius) * h_cup_cm * TAPERED_CUP_FACTOR;
}

} // namespace VolumeMath
