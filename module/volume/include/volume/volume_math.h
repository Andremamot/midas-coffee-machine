/**
 * @file volume_math.h
 * @brief Fungsi estimasi volume gelas dari data kamera.
 *
 * Port dari Python: 07_midas_aruco_fusion/core/volume_math.py
 *
 * Pipeline:
 *   measure_rim_width_px() → calc_diameter() → calc_volume()
 */

#pragma once

#include <opencv2/core.hpp>
#include "height_math.h"   // for BBox

namespace fusion {

/**
 * @brief Ukur lebar rim gelas (piksel) via Otsu strip detection.
 *
 * Mengambil strip 10% atas dari bbox, lakukan Otsu threshold,
 * lalu cari kolom kiri-kanan yang ada piksel.
 *
 * @param frame  BGR frame (sudah di-undistort oleh modul)
 * @param bbox   Bounding box gelas dari YOLO
 * @return Lebar rim dalam piksel (selalu > 0, fallback 50% bbox width)
 */
double measure_rim_width_px(const cv::Mat& frame, const BBox& bbox);

/**
 * @brief Hitung diameter fisik gelas (cm) via pinhole camera model.
 *
 * diameter = (rim_w_px × z_rim_cm) / focal_px
 *
 * @param rim_w_px  Lebar rim dalam piksel
 * @param z_rim_cm  Jarak kamera → bibir gelas (cm) = z_tray - h_cup
 * @param focal_px  Focal length efektif dalam piksel
 * @return Diameter dalam cm (0.0 jika input tidak valid)
 */
double calc_diameter(double rim_w_px, double z_rim_cm, double focal_px);

/**
 * @brief Hitung volume gelas (mL) menggunakan model silinder.
 *
 * V = π × (d/2)² × h   (1 cm³ = 1 mL)
 *
 * @param h_cup_cm    Tinggi gelas dalam cm
 * @param diameter_cm Diameter gelas dalam cm
 * @return Volume dalam mL (0.0 jika input tidak valid)
 */
double calc_volume(double h_cup_cm, double diameter_cm);

}  // namespace fusion
