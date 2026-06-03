/*******************************************************************************
 * volume_math.hpp
 * C++ port of 07_midas_aruco_fusion_new/core/volume_math.py
 *
 * Fungsi pure-math untuk estimasi volume gelas dari data sensor:
 *   1. measureRimWidthPx   — lebar rim via Otsu threshold (lebih akurat dari bbox_w)
 *   2. calcDiameter        — diameter fisik (cm) via pinhole model
 *   3. calcVolume          — volume silinder (mL = cm³)
 ******************************************************************************/
#pragma once
/* Synced from 07_midas_aruco_fusion/core/volume_math.hpp */

#include <opencv2/opencv.hpp>

namespace VolumeMath {

/**
 * Ukur lebar gelas di level rim (strip atas bounding box).
 *
 * Menggunakan Otsu threshold pada strip horizontal tipis (10% bbox_h)
 * di bagian atas bbox untuk menemukan tepi kiri-kanan gelas di level bibir.
 * Lebih akurat dari bbox_w penuh karena tidak terpengaruh body yang lebih lebar.
 *
 * @param frame  Frame BGR yang sudah di-undistort.
 * @param bbox   Bounding box gelas dari YOLO (x, y, w, h dalam cv::Rect).
 * @return       Lebar rim dalam piksel. Selalu >= 1.0.
 */
float measureRimWidthPx(const cv::Mat& frame, const cv::Rect& bbox);

/**
 * Hitung diameter fisik gelas (cm) via model pinhole kamera.
 *
 * Formula: diameter = (rim_w_px × z_rim_cm) / focal_px
 *
 * @param rim_w_px   Lebar rim dalam piksel (dari measureRimWidthPx).
 * @param z_rim_cm   Jarak kamera ke bibir gelas dalam cm (z_tray - h_cup).
 * @param focal_px   Focal length efektif dalam piksel (aruco camera_matrix[0,0]).
 * @return           Diameter dalam cm. Return 0.0 jika input tidak valid.
 */
double calcDiameter(double rim_w_px, double z_rim_cm, double focal_px);

/**
 * Hitung volume gelas (mL) menggunakan model silinder.
 *
 * Formula: V = π × (d/2)² × h   (1 cm³ = 1 mL)
 *
 * @param h_cup_cm     Tinggi gelas dalam cm.
 * @param diameter_cm  Diameter gelas dalam cm (dari calcDiameter).
 * @return             Volume dalam mL. Return 0.0 jika input tidak valid.
 */
double calcVolume(double h_cup_cm, double diameter_cm);

} // namespace VolumeMath
